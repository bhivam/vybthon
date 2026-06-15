from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

import litellm
from litellm import CustomStreamWrapper, completion
from litellm.types.utils import ModelResponseStream, StreamingChoices

from .codegen import Dependency, Spec, build_messages, materialize, strip_fences
from .console import Console

litellm.suppress_debug_info = True

# Dependency names recorded in a cache file's header, so a function loaded from
# disk in a fresh process still gets its siblings injected even before any
# spec() calls are re-registered.
_USES_RE = re.compile(r"^# vibe-uses: (.+)$", re.MULTILINE)


def _strip_header(text: str) -> str:
    """Drop the leading ``# vibe-...`` comment header from a cached file,
    leaving just the synthesized source (for showing to the model)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("#")):
        i += 1
    return "\n".join(lines[i:]).strip()

class Vibe:
    """An object whose undefined methods are synthesized by an LLM on first call."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None ,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        packages: list[str] | None = None,
        cache_dir: Path | str | None = None,
        verbose: bool = False,
        retries: int = 2,
        caching: bool = True,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._extra_body = extra_body
        self._packages = packages

        self._cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".vibe_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._retries = retries
        self._console = Console(enabled=verbose)
        self._fns: dict[str, Callable[..., Any]] = {}
        self._sources: dict[str, str] = {}
        self._specs: dict[str, Spec] = {}
        self._caching = caching

    @classmethod
    def openrouter(
        cls,
        model: str,
        *,
        reasoning: bool = False,
        **kwargs: Any,
    ) -> "Vibe":
        OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        if reasoning:
            extra_body = {**kwargs.pop("extra_body", {}), "reasoning": {"enabled": True}}
            kwargs["extra_body"] = extra_body
        return cls(model, api_base=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, **kwargs)

    # -- public surface ----------------------------------------------------

    def spec(
        self,
        name: str,
        *,
        context: str | None = None,
        inputs: str | None = None,
        returns: str | None = None,
        uses: list[str] | None = None,
    ) -> "Vibe":
        """Register first-call guidance for ``name``: free-form context, the
        intended input/output shapes the synthesizer should match, and ``uses``
        — names of other vibe functions ``name`` depends on.

        Each name in ``uses`` is shown to the synthesizer as exact source if it
        has already been synthesized (else as its spec), so the new function
        matches its formats — e.g. a decoder written against the encoder that
        actually exists. They are also injected into the synthesized function's
        namespace, so generated code can call them directly. If a dependency is
        later evicted and regenerated, every function that ``uses`` it is
        evicted too, since it was written against the old version.

        Consulted only when ``name`` is synthesized, so it has no effect once the
        function is cached and is never part of the cache key. Returns ``self``
        so specs can be chained before the first call.
        """
        self._specs[name] = Spec(
            context=context, inputs=inputs, returns=returns, uses=tuple(uses or ())
        )
        return self

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # Never intercept dunder / private lookups (copy, repr, IPython probes,
        # and our own ``self._foo`` access during __init__ all rely on this).
        if name.startswith("_"):
            raise AttributeError(name)

        def caller(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, args, kwargs)

        caller.__name__ = name
        caller.__qualname__ = f"Vibe.{name}"
        return caller

    # -- call / retry ------------------------------------------------------

    def _call(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Resolve and invoke ``name``. If synthesis or the call itself fails,
        evict the bad version and re-synthesize with the exception as context,
        up to ``self._retries`` times."""
        error: Exception | None = None
        for attempt in range(1, self._retries + 2):  # 1 initial try + N retries
            try:
                fn = self._resolve(name, args, kwargs, error)
                return fn(*args, **kwargs)
            except Exception as exc:
                self._evict(name)  # this version is bad — don't keep or reuse it
                error = exc
                self._console.status(f"{name} failed (attempt {attempt}): {exc!r}")
        assert error is not None
        raise error

    def _evict(self, name: str, _seen: set[str] | None = None) -> None:
        """Drop a synthesized function from both caches so it is regenerated.

        Cascades to dependents: any function whose spec ``uses`` this one was
        synthesized against the version being evicted, so it goes too. A seen
        set keeps mutual dependencies from recursing forever.
        """
        seen = _seen if _seen is not None else set()
        if name in seen:
            return
        seen.add(name)
        self._fns.pop(name, None)
        self._sources.pop(name, None)
        self._cache_path(name).unlink(missing_ok=True)
        for dependent, spec in self._specs.items():
            if name in spec.uses:
                self._evict(dependent, seen)

    # -- resolution / caching ---------------------------------------------

    def _resolve(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        error: Exception | None = None,
    ) -> Callable[..., Any]:
        """Return a callable for ``name``: memory cache → disk cache → synthesize.

        On a retry (``error`` set) the caches are skipped and a fresh version is
        synthesized with the failure as context.
        """
        if self._caching and error is None:
            cached = self._fns.get(name)
            if cached is not None:
                return cached

            path = self._cache_path(name)
            if path.exists():
                text = path.read_text()
                fn = materialize(text, name, extras=self._dep_extras(name, text))
                self._sources[name] = _strip_header(text)
                self._fns[name] = fn
                return fn

        source = strip_fences(self._generate(name, args, kwargs, error))
        fn = materialize(source, name, extras=self._dep_extras(name))
        self._sources[name] = source
        if self._caching:
            self._cache_path(name).write_text(self._cache_header(name) + source + "\n")
            self._fns[name] = fn
        return fn

    def _dep_names(self, name: str, cached_text: str | None = None) -> tuple[str, ...]:
        """Names of the vibe functions ``name`` may call: from its spec, or —
        when loading from disk without a re-registered spec — from the
        ``# vibe-uses:`` line in the cache header."""
        spec = self._specs.get(name)
        if spec is not None and spec.uses:
            return spec.uses
        if cached_text is not None:
            match = _USES_RE.search(cached_text)
            if match:
                return tuple(d.strip() for d in match.group(1).split(",") if d.strip())
        return ()

    def _dep_extras(
        self, name: str, cached_text: str | None = None
    ) -> dict[str, Any]:
        """Bind each declared dependency to its lazy vibe caller so synthesized
        code can invoke siblings by name (resolving them on first use)."""
        return {dep: getattr(self, dep) for dep in self._dep_names(name, cached_text)}

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.py"

    def _cache_header(self, name: str) -> str:
        lines = [
            f"# vibe-generated: {name}",
            f"# model: {self._model}",
        ]
        deps = self._dep_names(name)
        if deps:
            lines.append(f"# vibe-uses: {', '.join(deps)}")
        lines.append("# WARNING: machine-written code, review before trusting.")
        return "\n".join(lines) + "\n\n"

    # -- generation --------------------------------------------------------

    def _generate(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        error: Exception | None = None,
    ) -> str:
        """Synthesize the source for ``name`` (one request; ``_call`` handles retries).

        When ``error`` is set, it is passed to the model as context so the new
        version avoids whatever the previous one got wrong.
        """
        self._console.status(f"synthesizing {name}" + (" (retry)" if error else ""))
        messages = build_messages(
            name, args, kwargs, error,
            packages=self._packages,
            spec=self._specs.get(name),
            dependencies=[self._dependency(d) for d in self._dep_names(name)],
        )
        source = self._stream(messages)
        self._console.status(f"synthesized {name}")
        return source

    def _dependency(self, dep: str) -> Dependency:
        """What we can show the model about ``dep``: its exact synthesized
        source (memory, then disk), else just its registered spec."""
        source = self._sources.get(dep)
        if source is None and self._caching:
            path = self._cache_path(dep)
            if path.exists():
                source = _strip_header(path.read_text())
        return Dependency(name=dep, source=source, spec=self._specs.get(dep))

    def _completion_kwargs(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """litellm kwargs; api_base/api_key/extra_body sent only when set."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if self._api_base and not self._model.startswith("openrouter/"):
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        return kwargs

    def _stream(self, messages: list[dict[str, str]]) -> str:
        """Stream a completion and return the concatenated content."""
        resp = completion(**self._completion_kwargs(messages))
        assert isinstance(resp, CustomStreamWrapper)

        parts: list[str] = []
        for chunk in resp:
            assert isinstance(chunk, ModelResponseStream)
            choice = chunk.choices[0]
            assert isinstance(choice, StreamingChoices)
            text = choice.delta.content
            if text:
                parts.append(text)
        return "".join(parts)

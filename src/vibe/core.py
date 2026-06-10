"""The ``Vibe`` object: undefined methods are synthesized by an LLM on call."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import litellm
from litellm import CustomStreamWrapper, completion
from litellm.types.utils import ModelResponseStream, StreamingChoices

from .codegen import build_messages, materialize, strip_fences
from .console import Console

litellm.suppress_debug_info = True  # silence litellm's "Provider List" banner



class Vibe:
    """An object whose undefined methods are synthesized by an LLM on first call.

    Call any method name; if it doesn't exist yet, the model writes a function
    that does what the name + arguments imply, it's executed, and the source is
    cached to disk for reuse on later calls and future runs.

    SECURITY: synthesized code is executed with ``exec``. LLM output is
    untrusted — treat every generated function as arbitrary code. This is a toy,
    not a sandbox; run it only against a local model you trust.
    """

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None ,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
        cache_dir: Path | str | None = None,
        verbose: bool = False,
        retries: int = 2,
        caching: bool = True,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._extra_body = extra_body
        self._cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".vibe_cache"
        self._retries = retries
        self._console = Console(enabled=verbose)
        self._fns: dict[str, Callable[..., Any]] = {}
        self._cache_dir.mkdir(parents=True, exist_ok=True)
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
        OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

        if reasoning:
            extra_body = {**kwargs.pop("extra_body", {}), "reasoning": {"enabled": True}}
            kwargs["extra_body"] = extra_body
        return cls(model, api_base=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, **kwargs)

    # -- public surface ----------------------------------------------------

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

    def _evict(self, name: str) -> None:
        """Drop a synthesized function from both caches so it is regenerated."""
        self._fns.pop(name, None)
        self._cache_path(name).unlink(missing_ok=True)

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
                fn = materialize(path.read_text(), name)
                self._fns[name] = fn
                return fn

        source = strip_fences(self._generate(name, args, kwargs, error))
        fn = materialize(source, name)
        if self._caching:
            self._cache_path(name).write_text(self._cache_header(name) + source + "\n")
            self._fns[name] = fn
        return fn

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.py"

    def _cache_header(self, name: str) -> str:
        return (
            f"# vibe-generated: {name}\n"
            f"# model: {self._model}\n"
            f"# WARNING: machine-written code, review before trusting.\n\n"
        )

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
        source = self._stream(build_messages(name, args, kwargs, error))
        self._console.status(f"synthesized {name}")
        return source

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

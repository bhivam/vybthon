from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\s*\n(.*?)```", re.DOTALL)

# How much of each argument to show the model: enough to convey shape and
# element types without flooding the prompt with large inputs.
_SAMPLE = 3       # elements shown from a collection argument
_MAX_REPR = 200   # characters shown from a single value's repr

SYSTEM_PROMPT = """\
You are a Python function synthesizer. Given a function name and the exact \
arguments it will be called with, you write exactly one Python function \
implementing the behavior implied by the name.

Hard rules:
- Output ONLY the function definition. No prose, no markdown fences, no usage \
examples, no test calls, no print statements.
- The function MUST be named exactly `{name}`.
- The function MUST be callable exactly as the call shown below. Every parameter \
the call does not supply MUST have a default value, so that call succeeds. Extra \
optional parameters are fine; an extra *required* parameter is not. Add precise \
type hints.
- Add one concise docstring line.
{dependencies}
- Be pure and deterministic: no I/O, no network, no global state, and no \
randomness unless the name explicitly implies it.
- Return the result; never just print it.\
"""

STDLIB_RULE = (
    "- Use ONLY the Python standard library. Third-party packages (e.g. "
    "`phonenumbers`, `numpy`, `requests`) are NOT installed and importing them "
    "fails at runtime — implement the behavior yourself instead. Put any `import` "
    "statements INSIDE the function body."
)


def _dependency_rule(packages: list[str] | None) -> str:
    """The import-policy bullet for the system prompt, given allowed packages."""
    if not packages:
        return STDLIB_RULE
    listed = ", ".join(f"`{p}`" for p in packages)
    return (
        f"- You may use the Python standard library and these installed "
        f"third-party packages: {listed}. Prefer them when they fit the task — "
        f"they are available at runtime and you are expected to know them well. "
        f"Do NOT import any third-party package outside that list; it is not "
        f"installed and will fail. Put any `import` statements INSIDE the function "
        f"body."
    )

USER_PROMPT = """\
Write the function `{name}`.

It will be called like:
    {call}

Arguments:
{arguments}

Define `{name}` so this exact call works, then infer the intended behavior from \
the name and the inputs and return the function definition.\
"""

RETRY_PROMPT = """\

The previous version of `{name}` failed when called with these arguments:
    {error}

Write a corrected version that does not raise this error.{hint}\
"""

# Extra, error-specific guidance appended to the retry prompt.
def _import_hint(packages: list[str] | None) -> str:
    if packages:
        listed = ", ".join(f"`{p}`" for p in packages)
        return (
            f" Only the standard library and these packages are installed: "
            f"{listed}. Do not import anything outside that set."
        )
    return (
        " That module is NOT installed and never will be — do not reach for another "
        "third-party package. Implement this using ONLY the Python standard library."
    )


@dataclass(frozen=True)
class Spec:
    """First-call guidance for a function: free-form context and I/O shapes.

    Held in a registry keyed by function name and consulted only at synthesis
    time, so it never affects the cache and is ignored once a function is cached.
    """

    context: str | None = None
    inputs: str | None = None
    returns: str | None = None

    def render(self) -> str:
        parts = []
        if self.context:
            parts.append(self.context)
        if self.inputs:
            parts.append(f"Inputs: {self.inputs}")
        if self.returns:
            parts.append(f"Output: {self.returns}")
        if not parts:
            return ""
        body = "\n".join(parts)
        return (
            "\n\nAuthoritative guidance for this function — follow it and match "
            f"the stated shapes:\n{body}"
        )


class VibeError(RuntimeError):
    """Raised when synthesis produces something we can't turn into a function.

    Carries the offending source; ``str(err)`` appends it with line numbers.
    """

    def __init__(
        self, message: str, *, name: str | None = None, source: str | None = None
    ) -> None:
        super().__init__(message)
        self.name = name
        self.source = source

    def __str__(self) -> str:
        base = super().__str__()
        if not self.source:
            return base
        width = len(str(self.source.count("\n") + 1))
        numbered = "\n".join(
            f"  {i:>{width}} | {line}"
            for i, line in enumerate(self.source.splitlines(), 1)
        )
        label = f" for {self.name!r}" if self.name else ""
        return f"{base}\n\n  synthesized source{label}:\n{numbered}"


def strip_fences(text: str) -> str:
    """Pull code out of a ```fenced``` block if present, else return as-is."""
    match = _FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


def _preview(value: Any) -> str:
    """A bounded repr of ``value``: collections sampled, long values truncated."""
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        shown = ", ".join(repr(x) for x in items[:_SAMPLE])
        more = f", … (+{len(items) - _SAMPLE} more)" if len(items) > _SAMPLE else ""
        brackets = {list: "[]", tuple: "()", set: "{}", frozenset: "{}"}[type(value)]
        return f"{brackets[0]}{shown}{more}{brackets[1]}"
    if isinstance(value, dict):
        items = list(value.items())
        shown = ", ".join(f"{k!r}: {v!r}" for k, v in items[:_SAMPLE])
        more = f", … (+{len(items) - _SAMPLE} more)" if len(items) > _SAMPLE else ""
        return f"{{{shown}{more}}}"
    text = repr(value)
    if len(text) > _MAX_REPR:
        return f"{text[:_MAX_REPR]}… (+{len(text) - _MAX_REPR} chars)"
    return text


def render_call(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Render the call expression, e.g. ``greet('Sam', loud=True)`` (values bounded)."""
    parts = [_preview(a) for a in args]
    parts += [f"{k}={_preview(v)}" for k, v in kwargs.items()]
    return f"{name}({', '.join(parts)})"


def render_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """One line per argument: position/name, type, and a bounded value preview."""
    lines = [
        f"  - positional #{i}: type {type(a).__name__}, value {_preview(a)}"
        for i, a in enumerate(args, 1)
    ]
    lines += [
        f"  - keyword {k!r}: type {type(v).__name__}, value {_preview(v)}"
        for k, v in kwargs.items()
    ]
    return "\n".join(lines) if lines else "  (none)"


def build_messages(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    error: Exception | None = None,
    *,
    packages: list[str] | None = None,
    spec: Spec | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages that ask the model to synthesize ``name``.

    ``packages`` are third-party deps the model is allowed to import. ``spec`` is
    optional first-call guidance (context and I/O shapes). If ``error`` is given
    (a previous attempt that raised), its type and message are appended so the
    model can correct course on the retry.
    """
    user = USER_PROMPT.format(
        name=name,
        call=render_call(name, args, kwargs),
        arguments=render_arguments(args, kwargs),
    )
    if spec is not None:
        user += spec.render()
    if error is not None:
        hint = _import_hint(packages) if isinstance(error, ImportError) else ""
        user += RETRY_PROMPT.format(
            name=name, error=f"{type(error).__name__}: {error}", hint=hint
        )
    system = SYSTEM_PROMPT.format(name=name, dependencies=_dependency_rule(packages))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def materialize(source: str, name: str) -> Callable[..., Any]:
    """exec ``source`` in a fresh namespace and pull out the function ``name``."""
    namespace: dict[str, Any] = {}
    try:
        exec(compile(source, f"<vibe:{name}>", "exec"), namespace)
    except SyntaxError as exc:
        raise VibeError(
            f"synthesized code for {name!r} is not valid Python: {exc}",
            name=name,
            source=source,
        ) from exc
    fn = namespace.get(name)
    if not callable(fn):
        raise VibeError(
            f"synthesized code did not define a callable named {name!r}; "
            f"got {type(fn).__name__}",
            name=name,
            source=source,
        )
    return fn

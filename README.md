# vibethon

Objects whose methods are written by a local LLM the moment you call them.

Call any method name on a `Vibe` instance. If it doesn't exist yet, a local
model (via [litellm](https://github.com/BerriAI/litellm) + [Ollama](https://ollama.com))
synthesizes a Python function that does what the name and arguments imply, runs
it, and caches the source to disk so the next call — this run or a future one —
reuses it instantly.

```python
from vibe import Vibe

v = Vibe()
v.reverse_string("hello")        # LLM writes reverse_string, runs it -> "olleh"
v.nth_prime(10)                  # synthesized once, then cached -> 29
v.celsius_to_fahrenheit(100)     # -> 212.0
```

## Providers

Defaults to a local Ollama model. Any litellm-supported provider works by
passing a `model` id and `api_key`. For OpenRouter there's a helper:

```python
v = Vibe.openrouter(
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",  # explicit prefix required
    api_key=...,        # or set OPENROUTER_API_KEY (e.g. in a gitignored .env)
    reasoning=True,     # forwards {"reasoning": {"enabled": True}}
)
```

The `openrouter/` prefix is required — we don't guess the provider. `api_base`
is only sent for local/self-hosted models; hosted providers resolve their own
endpoint. See `examples/openrouter.py`.

## Layout

```
src/vibe/          the package
  core.py          the Vibe object: __getattr__ → synthesize → exec → cache
  codegen.py       prompts, fence stripping, exec-into-namespace
  console.py       dim status lines + streamed code on stderr
examples/
  demo.py          call several non-existent methods
  ask_ollama.py    minimal single-shot litellm call
```

## How it works

1. `Vibe.__getattr__(name)` returns a closure (dunder/private names are passed
   through so `copy`/`repr`/IPython probes don't get synthesized).
2. On call, `_resolve` checks the in-memory cache, then `./.vibe_cache/<name>.py`
   on disk, then asks the model.
3. The call site's `*args, **kwargs` are rendered into the prompt so the model
   writes the correct signature; the response is stripped of markdown fences,
   `exec`'d into a fresh namespace, and the function is pulled out by name.
4. Validated source is written to the cache and reused thereafter.

## Running

This is a Nix flake project. Precompiled wheels (litellm's `tokenizers`) need a
standard `libstdc++`, which the dev shell puts on `LD_LIBRARY_PATH`:

```bash
nix develop                       # enter the dev shell
uv sync                           # install deps + the vibe package (editable)
ollama serve                      # and `ollama pull gemma4:12b`
uv run python examples/demo.py
```

## Security

Synthesized code is executed with `exec`. **LLM output is untrusted — treat
every generated function as arbitrary code.** This is a toy / "vibe coding"
tool, not a sandbox. Run it only against a local model you trust, and review the
contents of `.vibe_cache/`.

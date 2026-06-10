# vibethon

Objects whose methods are written by an LLM the moment you call them. Call any
method; if it doesn't exist, a model writes it from the name + arguments, runs
it, and caches the source to `./.vibe_cache/` for instant reuse.

```python
from vibe import Vibe

v = Vibe.openrouter("openrouter/meta-llama/llama-3.1-8b-instruct")  # set OPENROUTER_API_KEY
v.reverse_string("hello")   # -> "olleh"
v.nth_prime(10)             # cached after first call -> 29
```

## Install

Needs Python ≥ 3.10 and an LLM provider (hosted, or a local Ollama).

```bash
pip install "git+https://github.com/bhivam/vibethon"   # or: uv add "git+..."
```

The distribution is `vibethon`; the import is `vibe`. On NixOS, run inside
`nix develop` (litellm's wheels need `libstdc++`).

## Security

Synthesized code runs via `exec`. LLM output is untrusted — treat every
generated function as arbitrary code. This is a toy, not a sandbox.

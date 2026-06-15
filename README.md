# vybthon

Objects whose methods are written by an LLM the moment you call them. Call any
method; if it doesn't exist, a model writes it from the name + arguments, runs
it, and caches the source to `./.vibe_cache/` for instant reuse.

```python
from vibe import Vibe

v = Vibe.openrouter("openrouter/meta-llama/llama-3.1-8b-instruct")  # set OPENROUTER_API_KEY
v.reverse_string("hello")   # -> "olleh"
v.nth_prime(10)             # cached after first call -> 29
```

## Functions that depend on each other

Declare dependencies with `uses=` and the synthesizer is shown the *exact
source* of each dependency (or its spec, if not yet written) — so a decoder is
written against the encoder that actually exists, not a guess at one:

```python
v.spec("huffman_encode", inputs="str", returns="(dict[str, int], str): counts + compressed string")
v.spec("huffman_decode", uses=["huffman_encode"],
       inputs="the tuple produced by huffman_encode", returns="str: the original string")

payload = v.huffman_encode("hello world")
v.huffman_decode(payload)   # -> "hello world"
```

Dependencies are also in scope inside the generated function, so synthesized
code may call them directly. If a dependency fails and is regenerated, every
function that `uses` it is evicted and re-synthesized against the new version.
See `examples/huffman.py`.

## Install

Needs Python ≥ 3.10 and an LLM provider (hosted, or a local Ollama).

```bash
pip install "git+https://github.com/bhivam/vybthon"   # or: uv add "git+..."
```

The distribution is `vybthon`; the import is `vibe`. On NixOS, run inside
`nix develop` (litellm's wheels need `libstdc++`).

## Security

Synthesized code runs via `exec`. LLM output is untrusted — treat every
generated function as arbitrary code. This is a toy, not a sandbox.

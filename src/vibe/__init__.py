"""vibe: objects whose methods are written by an LLM the moment you call them

    >>> from vibe import Vibe
    >>> v = Vibe()
    >>> v.reverse_string("hello")      
    'olleh'
    >>> v.nth_prime(10)               
    29

vibed code is executed with minimal checks. LLM output is untrusted.
"""

from .codegen import VibeError
from .core import Vibe

__all__ = ["Vibe", "VibeError"]

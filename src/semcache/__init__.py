"""SemCache — semantic caching layer for LLM APIs.

Returns cached answers for semantically-similar prompts (embedding cosine
similarity above a threshold), cutting spend and latency on repetitive traffic.
"""

__version__ = "0.1.0"

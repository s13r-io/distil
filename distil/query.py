"""Read layer — retrieve → relevance gate → grounded synthesis. ARCHITECTURE.md §9.

Built incrementally across Phase 10. This module starts with the vector math shared by
retrieval; ranking, the abstention gate, and grounded synthesis are added in 10.4–10.7.
"""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (0 if either is zero-length)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

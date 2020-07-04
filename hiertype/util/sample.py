from typing import *
import numpy as np


def sample_except(n: int, k: int, excluded: Set[int]) -> np.ndarray:
    """Sample k numbers in {0 ... n - 1} except excluded."""

    samples = np.random.choice(n - len(excluded), k)
    for x in sorted(excluded):
        samples[samples >= x] += 1

    return samples


def sample_in_range_except(lo: int, hi: int, k: int, excluded: Set[int]) -> List[int]:
    exc = {x for x in excluded if lo <= x < hi}
    try:
        return list(sample_except(hi - lo, k, {x - lo for x in exc}) + lo)
    except:
        return []


def sample_multiple_from(xs: List[int], n: int) -> List[int]:
    indices = np.random.choice(len(xs), n)
    return [xs[i] for i in indices]

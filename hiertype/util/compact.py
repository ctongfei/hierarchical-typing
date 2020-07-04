from typing import *
import numpy as np


def compact2(
        xss: List[List[int]],
        pad: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pads and constructs a compact representation for a List[List[int]].
    """
    len1 = [len(xs) for xs in xss]
    max_len1 = max(len1)

    c = np.stack([
        np.pad(
            np.array(xs, dtype=np.int64),
            (0, max_len1 - len(xs)),
            mode='constant',
            constant_values=pad
        )
        for xs in xss
    ])

    l1 = np.array(len1)
    return l1, c


def compact3(
        xsss: List[List[List[int]]],
        pad: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pads and constructs a compact representation for a List[List[List[int]]].
    """
    for xss in xsss:
        if len(xss) == 0:
            xss.append([pad])

    len1 = [len(xss) for xss in xsss]
    max_len1 = max(len1)
    len2 = [[len(xs) for xs in xss] for xss in xsss]
    max_len2 = max(max(l, default=0) for l in len2)

    c = np.stack([
        np.pad(
            np.stack([
                np.pad(
                    np.array(xs, dtype=np.int64),
                    (0, max_len2 - len(xs)),
                    mode='constant',
                    constant_values=pad
                )
                for xs in xss
            ]),
            [(0, max_len1 - len(xss)), (0, 0)],
            mode='constant',
            constant_values=pad
        )
        for xss in xsss
    ])

    l1, l2 = compact2(len2, 0)

    return l1, l2, c

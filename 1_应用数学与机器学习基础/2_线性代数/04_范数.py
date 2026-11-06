# 04_范数
"""Lecture: 1_应用数学与机器学习基础/2_线性代数.

A dependency-free, numerically stable Euclidean norm example.
"""

import math
from collections.abc import Iterable


def l2_norm(values: Iterable[float]) -> float:
    """Return sqrt(sum(x*x)) without avoidable intermediate overflow.

    The empty vector has norm zero. Non-numeric values raise TypeError;
    NaN and infinity follow math.hypot's standard floating-point behavior.
    """
    return math.hypot(*(float(value) for value in values))


if __name__ == "__main__":
    print(f"||[3, 4]||_2 = {l2_norm([3, 4]):g}")

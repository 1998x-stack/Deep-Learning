"""Regression tests for the standalone norm lesson."""
import importlib.util
import math
import pathlib
import unittest

MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "1_应用数学与机器学习基础" / "2_线性代数" / "04_范数.py"
spec = importlib.util.spec_from_file_location("norm_lesson", MODULE_PATH)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)


class L2NormTests(unittest.TestCase):
    def test_three_four_five(self):
        self.assertEqual(lesson.l2_norm([3, 4]), 5.0)

    def test_empty_vector(self):
        self.assertEqual(lesson.l2_norm([]), 0.0)

    def test_large_components_do_not_overflow(self):
        self.assertTrue(math.isclose(lesson.l2_norm([1e200, 1e200]), math.sqrt(2) * 1e200))

    def test_small_components_do_not_underflow(self):
        self.assertGreater(lesson.l2_norm([1e-200, 1e-200]), 0.0)

    def test_non_numeric_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            lesson.l2_norm(["not a number"])


if __name__ == "__main__":
    unittest.main()

import unittest
from calc import emissionPoint, centerDistance
import math

class TestFunctions(unittest.TestCase):
    
    def test_center_dist(self):
        dist1 = centerDistance(
            {"x1": 1, "y1": 1, "z1": 1, "x2": -1, "y2": -1, "z2": -1, "dt": 0}
        )
        dist2 = centerDistance(
            {"x1": 3, "y1": 3, "z1": 2, "x2": -1, "y2": -1, "z2": 2, "dt": 2/300*math.sqrt(2)}
        )
        self.assertEqual(dist1, 0)
        self.assertEqual(dist2, 2)

if __name__ == '__main__':
    unittest.main()
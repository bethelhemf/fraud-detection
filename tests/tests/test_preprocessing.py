import unittest
import pandas as pd
from src.preprocessing import robust_ip_to_int

class TestPreprocessing(unittest.TestCase):
    def test_ip_conversion(self):
        # Test string IP
        self.assertEqual(robust_ip_to_int("1.1.1.1"), 16843009.0)
        # Test float IP
        self.assertEqual(robust_ip_to_int(16843009.0), 16843009.0)
        # Test invalid IP
        self.assertEqual(robust_ip_to_int("invalid"), 0.0)

if __name__ == '__main__':
    unittest.main()
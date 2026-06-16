import unittest
import os

class TestModelArtifacts(unittest.TestCase):
    def test_model_files_exist(self):
        self.assertTrue(os.path.exists('models/random_forest_model.pkl'))
        self.assertTrue(os.path.exists('models/random_forest_bank_model.pkl'))

if __name__ == '__main__':
    unittest.main()
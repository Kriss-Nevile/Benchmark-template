import unittest
from pathlib import Path

from summbench.dataset import load_dataset_from_csv


class DatasetTests(unittest.TestCase):
    def test_load_dataset_from_csv_reads_expected_columns(self):
        project_root = Path(__file__).resolve().parents[1]
        examples = load_dataset_from_csv(project_root / "data" / "sample_dataset.csv")

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].sample_id, 0)
        self.assertIn("library", examples[0].article.lower())
        self.assertIn("reading confidence", examples[0].reference_summary.lower())


if __name__ == "__main__":
    unittest.main()

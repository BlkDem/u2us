import tempfile
import unittest
from pathlib import Path

from scripts.run_real_baseline import prepare, rows_write


class PreparationTest(unittest.TestCase):
    def test_priority_normalization_and_cleaning(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {s: Path(directory) / (s + '.jsonl') for s in ('test', 'valid', 'train')}
            rows_write(paths['test'], [{'src': ' Привет ', 'tgt': 'Hello'}])
            rows_write(paths['valid'], [{'src': 'ПРИВЕТ', 'tgt': 'Different'}, {'src': 'Мир', 'tgt': 'World'}])
            rows_write(paths['train'], [{'src': 'Другой', 'tgt': ' WORLD '}, {'src': 'Хорошо\u0001 да', 'tgt': 'All right'}, {'src': '', 'tgt': 'Bad'}])
            parts, report = prepare(paths, 8000)
            self.assertEqual({k: len(v) for k,v in parts.items()}, {'test': 1, 'valid': 1, 'train': 1})
            self.assertEqual(parts['train'][0]['src'], 'Хорошо да')
            self.assertEqual(report['valid']['normalized_side_overlap_removed'], 1)
            self.assertEqual(report['train']['normalized_side_overlap_removed'], 1)


if __name__ == '__main__':
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from check_data import audit


def fixture(root):
    for folder in ('JPEGImages', 'PointAnnotations', 'ImageSets'):
        (root/folder).mkdir(parents=True)
    for name, color in [('中文 train', 'red'), ('val', 'blue')]:
        Image.new('RGB', (512, 512), color).save(root/'JPEGImages'/f'{name}.png')
        (root/'PointAnnotations'/f'{name}.txt').write_text('10 20\n', encoding='utf-8')
    (root/'ImageSets/train.txt').write_text('\ufeff中文 train\n', encoding='utf-8')
    (root/'ImageSets/val.txt').write_text('val\n', encoding='utf-8')


class AuditTests(unittest.TestCase):
    def test_fast_mode_never_converts_or_loads_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture(root)
            with patch('PIL.Image.Image.convert', side_effect=AssertionError('RGB decoding')), \
                 patch('PIL.PngImagePlugin.PngImageFile.load', side_effect=AssertionError('pixel loading')):
                report = audit(root, show_progress=False)
            self.assertEqual(report['train']['points'], 1)
            self.assertEqual(report['val']['sizes'], {'512x512': 1})

    def test_full_decode_and_header_reports_agree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture(root)
            self.assertEqual(audit(root, show_progress=False), audit(root, decode=True, show_progress=False))

    def test_duplicate_pixels_with_different_names_are_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture(root)
            (root/'JPEGImages/val.png').write_bytes((root/'JPEGImages/中文 train.png').read_bytes())
            audit(root, show_progress=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate decoded image'):
                audit(root, True, show_progress=False)

    def test_truncated_payload_requires_decode_check(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture(root)
            path = root/'JPEGImages/val.png'
            path.write_bytes(path.read_bytes()[:100])
            audit(root, show_progress=False)
            with self.assertRaisesRegex(ValueError, 'val/val'):
                audit(root, decode=True, show_progress=False)

    def test_outside_points_still_fail_with_image_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture(root)
            (root/'PointAnnotations/val.txt').write_text('512 20\n')
            with self.assertRaisesRegex(ValueError, 'val/val:.*outside'):
                audit(root, show_progress=False)

    def test_split_overlap_checked_before_image_io(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); fixture(root)
            (root/'ImageSets/val.txt').write_text('中文 train\n', encoding='utf-8')
            with patch('check_data.Image.open', side_effect=AssertionError('unnecessary I/O')):
                with self.assertRaisesRegex(ValueError, 'IDs overlap'):
                    audit(root, show_progress=False)


if __name__ == '__main__':
    unittest.main()

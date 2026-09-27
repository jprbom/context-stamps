"""Input-only partitions and bounded image identity, using authored fixtures."""

import io
import tempfile
import unittest
from unittest.mock import patch

from chartqa_prepare import image_identity, normalized_question, select_groups, shards


def cohort_rows():
    result = []
    for split in ("train", "val", "test"):
        for group in ("shared", split + "-own"):
            for index in range(6):
                result.append(dict(id=f"{split}-{group}-{index}", split=split, pixel_group=group,
                                   question=f"Question {index}?"))
    return result


class PreparationTests(unittest.TestCase):
    def test_shared_images_are_withheld_from_earlier_cohorts(self):
        plan = select_groups(cohort_rows(), dict(train=1, val=1, test=1))
        groups = {s: {g["group"] for g in rows} for s, rows in plan["selected"].items()}
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])
        self.assertEqual(groups["train"], {"train-own"})
        self.assertEqual(groups["val"], {"val-own"})
        self.assertTrue(any(e["reason"] == "image_in_later_native_split" for e in plan["exclusions"]))

    def test_selection_ignores_reference_labels_and_iteration_order(self):
        rows = cohort_rows()
        counts = dict(train=1, val=1, test=1)
        original = select_groups(rows, counts)
        changed = [dict(r, answer="ignored reference", label=["ignored label"]) for r in reversed(rows)]
        self.assertEqual(original, select_groups(changed, counts))
        self.assertTrue(all(len(g["ids"]) == 4 for groups in original["selected"].values() for g in groups))

    def test_duplicate_questions_do_not_create_a_repeated_task_group(self):
        rows = cohort_rows()
        for row in rows:
            if row["split"] == "train":
                row["question"] = " SAME  question " if row["id"].endswith("0") else "same question"
        with self.assertRaisesRegex(ValueError, "insufficient repeated-chart groups in train"):
            select_groups(rows, dict(train=1, val=1, test=1))
        self.assertEqual(normalized_question(" ＡＢＣ\tchart "), "abc chart")

    def test_invalid_counts_and_duplicate_ids_fail(self):
        for counts in (dict(train=True, val=1, test=1), dict(train=1, test=1), dict(train=257, val=1, test=1)):
            with self.assertRaises(ValueError):
                select_groups(cohort_rows(), counts)
        rows = cohort_rows()
        with self.assertRaises(ValueError):
            select_groups(rows + [rows[0]], dict(train=1, val=1, test=1))

    def png(self, compression):
        from PIL import Image
        buffer = io.BytesIO()
        image = Image.new("RGBA", (8, 8), color=(11, 22, 33, 255))
        image.putpixel((1, 2), (80, 10, 90, 100))
        image.save(buffer, format="PNG", compress_level=compression)
        return buffer.getvalue()

    def test_encoding_changes_do_not_hide_identical_pixels(self):
        a, b = image_identity(self.png(0)), image_identity(self.png(9))
        self.assertNotEqual(a["image_sha256"], b["image_sha256"])
        self.assertEqual(a["pixel_group"], b["pixel_group"])
        self.assertEqual((a["width"], a["height"], a["media_type"]), (8, 8, "image/png"))

    def test_malformed_images_and_resource_bounds_fail(self):
        from PIL import UnidentifiedImageError

        with self.assertRaises(UnidentifiedImageError):
            image_identity(b"not an image")
        with patch("chartqa_prepare.MAX_PIXELS", 63):
            with self.assertRaisesRegex(ValueError, "pixel limit"):
                image_identity(self.png(0))
        with patch("chartqa_prepare.MAX_IMAGE_BYTES", 2):
            with self.assertRaises(ValueError):
                image_identity(self.png(0))
        with self.assertRaises(ValueError):
            image_identity(bytearray(self.png(0)))

    def test_member_allowlist_rejects_traversal_before_file_access(self):
        from pathlib import Path

        from chartqa_prepare import DATA_REVISION, DATASET
        with self.assertRaisesRegex(ValueError, "dataset card and all five shards required"):
            shards(Path("unused"), dict(dataset=DATASET, revision=DATA_REVISION, files={"../../private": {}}))

    def test_native_png_names_are_url_encoded_and_hash_bound(self):
        from pathlib import Path

        from chartqa_native_prepare import fetch_png, git_identity

        raw = self.png(0)
        row = dict(source_image="ChartQA Dataset/train/png/OECD_R&D_(M3),x.png", declared_bytes=len(raw), git_blob=git_identity(raw))
        with tempfile.TemporaryDirectory() as directory:
            with patch("chartqa_native_prepare.urllib.request.urlopen", return_value=io.BytesIO(raw)) as request:
                result = fetch_png(row, Path(directory))
                self.assertIn("OECD_R%26D_%28M3%29%2Cx.png", request.call_args.args[0])
                self.assertEqual(result["byte_length"], len(raw))
            with patch("chartqa_native_prepare.urllib.request.urlopen") as request:
                with self.assertRaises(ValueError):
                    fetch_png(dict(row, source_image="ChartQA Dataset/train/png/../../private.png"), Path(directory))
                request.assert_not_called()
            with patch("chartqa_native_prepare.urllib.request.urlopen", return_value=io.BytesIO(raw)):
                with self.assertRaisesRegex(ValueError, "differs from pinned Git source"):
                    fetch_png(dict(row, git_blob="0" * 40), Path(directory))


if __name__ == "__main__":
    unittest.main()

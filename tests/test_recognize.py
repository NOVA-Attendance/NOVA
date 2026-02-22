#!/usr/bin/env python3
"""
tests/test_recognize.py - Unit tests for compare_against_embedding()

Run from the repository root:
    python -m pytest tests/test_recognize.py -v

Real-image tests are skipped automatically if no images exist in data/image_database/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from pathlib import Path
import numpy as np
import pytest

from recognize_image import compare_against_embedding, compute_embedding


EMBEDDING_SIZE = 512


def random_embedding(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBEDDING_SIZE).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def find_test_image():
    candidates = (
        list(Path("data/image_database").rglob("*.jpg"))
        + list(Path("data/image_database").rglob("*.png"))
    )
    return candidates[0] if candidates else None


class TestCompareEdgeCases:

    def test_nonexistent_image_returns_error(self, tmp_path):
        result = compare_against_embedding(tmp_path / "missing.jpg", random_embedding())
        assert "error" in result
        assert result["matched"] is False
        assert result["confidence"] == 0.0

    def test_empty_embedding_returns_error(self, tmp_path):
        import cv2
        img_path = tmp_path / "blank.jpg"
        cv2.imwrite(str(img_path), np.ones((100, 100, 3), dtype=np.uint8) * 255)
        result = compare_against_embedding(img_path, [])
        assert "error" in result
        assert result["confidence"] == 0.0

    def test_result_always_has_required_keys(self, tmp_path):
        result = compare_against_embedding(tmp_path / "missing.jpg", random_embedding())
        for key in ("confidence", "distance", "matched", "model", "threshold", "metric"):
            assert key in result

    def test_confidence_never_negative(self, tmp_path):
        result = compare_against_embedding(tmp_path / "missing.jpg", random_embedding())
        assert result["confidence"] >= 0.0


@pytest.mark.skipif(find_test_image() is None, reason="No test image in data/image_database/")
class TestCompareRealImage:

    image = find_test_image()

    def test_self_match_high_confidence(self):
        own_embedding = compute_embedding(self.image, "Facenet512").tolist()
        result = compare_against_embedding(self.image, own_embedding, threshold=0.4)
        assert not result.get("error"), result.get("error")
        assert result["confidence"] >= 0.85
        assert result["matched"] is True

    def test_random_noise_low_confidence(self):
        result = compare_against_embedding(self.image, random_embedding(seed=42), threshold=0.4)
        assert not result.get("error"), result.get("error")
        assert result["confidence"] <= 0.5
        assert result["matched"] is False

    def test_threshold_zero_always_matches(self):
        own_embedding = compute_embedding(self.image, "Facenet512").tolist()
        result = compare_against_embedding(self.image, own_embedding, threshold=0.0)
        assert result["matched"] is True

    def test_threshold_one_never_matches(self):
        own_embedding = compute_embedding(self.image, "Facenet512").tolist()
        result = compare_against_embedding(self.image, own_embedding, threshold=1.0)
        assert result["matched"] is False

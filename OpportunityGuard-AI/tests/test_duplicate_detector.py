"""
tests/test_duplicate_detector.py — Tests for similarity search.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.duplicate_detector import load_duplicate_index, find_similar


@pytest.fixture(scope="module")
def dup_index():
    return load_duplicate_index()


class TestLoadDuplicateIndex:
    def test_index_loads(self, dup_index):
        assert dup_index is not None

    def test_index_has_required_keys(self, dup_index):
        # Artifact uses 'vectorizer'/'reference_texts' keys
        assert ("tfidf" in dup_index or "vectorizer" in dup_index)
        assert "matrix" in dup_index
        assert ("descriptions" in dup_index or "reference_texts" in dup_index)

    def test_index_not_empty(self, dup_index):
        descriptions = dup_index.get("descriptions") or dup_index.get("reference_texts", [])
        assert len(descriptions) > 0

    def test_missing_index_raises(self, tmp_path):
        from src.duplicate_detector import load_duplicate_index
        import unittest.mock as mock
        with mock.patch("src.duplicate_detector.INDEX_PATH", tmp_path / "missing.joblib"):
            with pytest.raises(FileNotFoundError):
                load_duplicate_index()


class TestFindSimilar:
    def test_returns_list(self, dup_index):
        results = find_similar("software developer internship", dup_index, top_k=3, threshold=0.05)
        assert isinstance(results, list)

    def test_result_has_required_fields(self, dup_index):
        results = find_similar("software engineer position at tech company", dup_index, top_k=1, threshold=0.01)
        if results:
            r = results[0]
            assert "similarity" in r
            assert "reference_index" in r
            assert "description_preview" in r

    def test_similarity_in_range(self, dup_index):
        results = find_similar("web developer remote intern", dup_index, top_k=5, threshold=0.01)
        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0

    def test_empty_description_returns_empty(self, dup_index):
        assert find_similar("", dup_index) == []

    def test_whitespace_only_returns_empty(self, dup_index):
        assert find_similar("   ", dup_index) == []

    def test_top_k_limit_respected(self, dup_index):
        results = find_similar("software developer job", dup_index, top_k=3, threshold=0.01)
        assert len(results) <= 3

    def test_threshold_filters_low_similarity(self, dup_index):
        # With a very high threshold, fewer or no results should be returned
        results_high = find_similar("software", dup_index, top_k=10, threshold=0.99)
        results_low = find_similar("software", dup_index, top_k=10, threshold=0.01)
        assert len(results_high) <= len(results_low)

    def test_results_sorted_by_similarity_desc(self, dup_index):
        results = find_similar("software engineer internship remote", dup_index, top_k=5, threshold=0.01)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i]["similarity"] >= results[i + 1]["similarity"]

    def test_description_preview_truncated(self, dup_index):
        results = find_similar("job description test", dup_index, top_k=3, threshold=0.01)
        for r in results:
            assert len(r["description_preview"]) <= 310  # small buffer over 300

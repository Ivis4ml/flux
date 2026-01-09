"""
Tests for rollout components (length predictor, partial buffer).
"""

import pytest
import torch
from datetime import datetime

from flux.core.trajectory import PartialTrajectory
from flux.core.types import PolicyVersion
from flux.rollout.length_predictor import (
    LengthObservation,
    LengthPrediction,
    LengthPredictor,
)
from flux.rollout.partial_buffer import (
    PartialEntry,
    PartialTrajectoryBuffer,
)


class TestLengthPrediction:
    """Tests for LengthPrediction dataclass."""

    def test_creation(self):
        """Test prediction creation."""
        pred = LengthPrediction(
            prompt="Hello, world!",
            predicted_length=100,
            confidence=0.8,
        )
        assert pred.prompt == "Hello, world!"
        assert pred.predicted_length == 100
        assert pred.confidence == 0.8


class TestLengthObservation:
    """Tests for LengthObservation dataclass."""

    def test_creation(self):
        """Test observation creation."""
        obs = LengthObservation(
            prompt="Hello",
            actual_length=50,
            prompt_length=5,
        )
        assert obs.prompt == "Hello"
        assert obs.actual_length == 50


class TestLengthPredictor:
    """Tests for LengthPredictor."""

    def test_creation_defaults(self):
        """Test predictor creation with defaults."""
        predictor = LengthPredictor()
        assert predictor.default_length == 256
        assert predictor.min_length == 16
        assert predictor.max_length == 4096

    def test_predict_basic(self):
        """Test basic prediction."""
        predictor = LengthPredictor()
        pred = predictor.predict("What is 2 + 2?")

        assert isinstance(pred, LengthPrediction)
        assert predictor.min_length <= pred.predicted_length <= predictor.max_length
        assert 0 <= pred.confidence <= 1

    def test_predict_batch(self):
        """Test batch prediction."""
        predictor = LengthPredictor()
        prompts = [
            "Hello",
            "What is the meaning of life?",
            "```python\ndef foo():\n    pass```",
        ]
        predictions = predictor.predict_batch(prompts)

        assert len(predictions) == 3
        assert all(isinstance(p, LengthPrediction) for p in predictions)

    def test_observe_updates_statistics(self):
        """Test that observations update statistics."""
        predictor = LengthPredictor()

        initial_mean = predictor._mean_length

        # Add observations
        predictor.observe("Hello", 100)
        predictor.observe("World", 200)

        # Mean should be updated
        assert predictor._total_observed == 2
        assert predictor._mean_length != initial_mean

    def test_observe_batch(self):
        """Test batch observation."""
        predictor = LengthPredictor()

        prompts = ["a", "b", "c"]
        lengths = [50, 100, 150]

        predictor.observe_batch(prompts, lengths)
        assert predictor._total_observed == 3

    def test_feature_extraction(self):
        """Test feature extraction from prompts."""
        predictor = LengthPredictor()

        # Question prompt
        features = predictor._extract_features("What is Python?")
        assert features["is_question"] == 1.0

        # Code prompt
        features = predictor._extract_features("```python\nprint('hello')```")
        assert features["has_code"] == 1.0

        # List request
        features = predictor._extract_features("List 5 things")
        assert features["wants_list"] == 1.0

        # Brief request
        features = predictor._extract_features("Briefly explain")
        assert features["wants_brief"] == 1.0

    def test_sort_by_length(self):
        """Test sorting prompts by predicted length."""
        predictor = LengthPredictor()

        prompts = [
            "This is a longer prompt that should produce more output",
            "Short",
            "Medium length prompt here",
        ]

        sorted_pairs = predictor.sort_by_length(prompts, ascending=True)

        # Check that we get all prompts back
        sorted_prompts = [p for p, _ in sorted_pairs]
        assert len(sorted_prompts) == 3
        assert set(sorted_prompts) == set(prompts)

    def test_bucket_by_length(self):
        """Test bucketing prompts by length."""
        predictor = LengthPredictor()

        prompts = ["A", "B", "C", "D", "E"]
        buckets = predictor.bucket_by_length(
            prompts,
            bucket_boundaries=(100, 200, 300),
        )

        # Should have all bucket categories
        assert "<100" in buckets
        assert "100-200" in buckets
        assert "200-300" in buckets
        assert ">300" in buckets

        # All prompts should be in some bucket
        total = sum(len(b) for b in buckets.values())
        assert total == 5

    def test_get_statistics(self):
        """Test getting predictor statistics."""
        predictor = LengthPredictor()
        predictor.observe("test", 100)

        stats = predictor.get_statistics()

        assert "mean_length" in stats
        assert "std_length" in stats
        assert "total_observed" in stats
        assert stats["total_observed"] == 1


class TestPartialEntry:
    """Tests for PartialEntry dataclass."""

    def test_creation(self):
        """Test entry creation."""
        traj = PartialTrajectory(
            id="test-1",
            prompt="Hello",
            response="Hi there",
            tokens=[1, 2, 3],
            log_probs=[-0.1, -0.2, -0.1],
            version=PolicyVersion(version_id=5),
        )
        entry = PartialEntry(trajectory=traj)

        assert entry.trajectory == traj
        assert entry.access_count == 0
        assert entry.prompt_hash != ""

    def test_touch_updates_access(self):
        """Test touch method updates access stats."""
        traj = PartialTrajectory(
            id="test-1",
            prompt="Hello",
            response="",
            tokens=[],
            log_probs=[],
            version=PolicyVersion(version_id=0),
        )
        entry = PartialEntry(trajectory=traj)

        assert entry.access_count == 0
        entry.touch()
        assert entry.access_count == 1
        assert entry.last_accessed is not None

    def test_compute_priority(self):
        """Test priority computation."""
        traj = PartialTrajectory(
            id="test-1",
            prompt="Hello",
            response="Response",
            tokens=list(range(100)),  # 100 tokens
            log_probs=[-0.1] * 100,
            version=PolicyVersion(version_id=5),
        )
        entry = PartialEntry(trajectory=traj)

        priority = entry.compute_priority(current_version=5)
        assert priority > 0


class TestPartialTrajectoryBuffer:
    """Tests for PartialTrajectoryBuffer."""

    def _make_partial(
        self, id: str, prompt: str, num_tokens: int, version: int
    ) -> PartialTrajectory:
        """Helper to create partial trajectory."""
        return PartialTrajectory(
            id=id,
            prompt=prompt,
            response="x" * num_tokens,
            tokens=list(range(num_tokens)),
            log_probs=[-0.1] * num_tokens,
            version=PolicyVersion(version_id=version),
        )

    def test_creation_defaults(self):
        """Test buffer creation with defaults."""
        buffer = PartialTrajectoryBuffer()
        assert buffer.max_size == 1000
        assert buffer.size == 0
        assert buffer.is_empty

    def test_add_partial(self):
        """Test adding partial trajectory."""
        buffer = PartialTrajectoryBuffer()
        partial = self._make_partial("test-1", "Hello", 100, 0)

        result = buffer.add(partial)
        assert result is True
        assert buffer.size == 1

    def test_add_too_short_rejected(self):
        """Test that too-short partials are rejected."""
        buffer = PartialTrajectoryBuffer(min_length_threshold=50)
        partial = self._make_partial("test-1", "Hello", 10, 0)

        result = buffer.add(partial)
        assert result is False
        assert buffer.size == 0

    def test_find_match(self):
        """Test finding matching partial."""
        buffer = PartialTrajectoryBuffer()
        partial = self._make_partial("test-1", "Hello, world!", 100, 0)
        buffer.add(partial)

        match = buffer.find_match("Hello, world!")
        assert match is not None
        assert match.id == "test-1"

    def test_find_match_not_found(self):
        """Test no match found."""
        buffer = PartialTrajectoryBuffer()
        partial = self._make_partial("test-1", "Hello", 100, 0)
        buffer.add(partial)

        match = buffer.find_match("Different prompt")
        assert match is None

    def test_find_match_staleness_filter(self):
        """Test that stale partials are filtered out."""
        buffer = PartialTrajectoryBuffer(max_staleness=3)
        partial = self._make_partial("test-1", "Hello", 100, 0)
        buffer.add(partial)

        # Within staleness
        match = buffer.find_match("Hello", current_version=2)
        assert match is not None

        # Beyond staleness
        match = buffer.find_match("Hello", current_version=10)
        assert match is None

    def test_find_match_removes_entry(self):
        """Test find_match with remove=True."""
        buffer = PartialTrajectoryBuffer()
        partial = self._make_partial("test-1", "Hello", 100, 0)
        buffer.add(partial)

        match = buffer.find_match("Hello", remove=True)
        assert match is not None
        assert buffer.size == 0

    def test_remove_entry(self):
        """Test removing entry by ID."""
        buffer = PartialTrajectoryBuffer()
        partial = self._make_partial("test-1", "Hello", 100, 0)
        buffer.add(partial)

        result = buffer.remove("test-1")
        assert result is True
        assert buffer.size == 0

        # Second remove should fail
        result = buffer.remove("test-1")
        assert result is False

    def test_max_size_eviction(self):
        """Test LRU-style eviction at capacity."""
        buffer = PartialTrajectoryBuffer(max_size=3, min_length_threshold=10)

        for i in range(5):
            partial = self._make_partial(f"test-{i}", f"Prompt {i}", 50, i)
            buffer.add(partial)

        assert buffer.size == 3
        # Oldest entries should be evicted

    def test_clear(self):
        """Test clearing buffer."""
        buffer = PartialTrajectoryBuffer()
        for i in range(5):
            partial = self._make_partial(f"test-{i}", f"Prompt {i}", 100, i)
            buffer.add(partial)

        count = buffer.clear()
        assert count == 5
        assert buffer.size == 0

    def test_cleanup_stale(self):
        """Test cleaning up stale entries."""
        buffer = PartialTrajectoryBuffer(max_staleness=3)

        # Add entries at different versions
        for i in range(5):
            partial = self._make_partial(f"test-{i}", f"Prompt {i}", 100, i)
            buffer.add(partial)

        # Cleanup at version 5 (entries 0, 1 are stale)
        removed = buffer.cleanup_stale(current_version=5)
        assert removed == 2

    def test_get_by_version(self):
        """Test getting entries by version."""
        buffer = PartialTrajectoryBuffer()

        for i in range(3):
            partial = self._make_partial(f"test-{i}", f"Prompt {i}", 100, 5)
            buffer.add(partial)

        entries = buffer.get_by_version(5)
        assert len(entries) == 3

        entries = buffer.get_by_version(10)
        assert len(entries) == 0

    def test_get_statistics(self):
        """Test getting buffer statistics."""
        buffer = PartialTrajectoryBuffer()
        partial = self._make_partial("test-1", "Hello", 100, 0)
        buffer.add(partial)
        buffer.find_match("Hello")

        stats = buffer.get_statistics()

        assert stats["size"] == 1
        assert stats["total_added"] == 1
        assert stats["total_matched"] == 1
        assert stats["match_rate"] == 1.0

    def test_update_priorities(self):
        """Test updating priorities."""
        buffer = PartialTrajectoryBuffer()

        for i in range(3):
            partial = self._make_partial(f"test-{i}", f"Prompt {i}", 100, i)
            buffer.add(partial)

        # Update priorities for version 2
        buffer.update_priorities(current_version=2)

        # Entries at version 2 should have highest priority
        # (This is mostly a smoke test)

    def test_thread_safety(self):
        """Test thread safety of buffer operations."""
        import threading

        buffer = PartialTrajectoryBuffer(max_size=100, min_length_threshold=10)
        errors = []

        def add_worker(start_id):
            try:
                for i in range(20):
                    partial = PartialTrajectoryBuffer
                    partial = self._make_partial(
                        f"test-{start_id}-{i}",
                        f"Prompt {start_id} {i}",
                        50,
                        i,
                    )
                    buffer.add(partial)
            except Exception as e:
                errors.append(e)

        def find_worker():
            try:
                for i in range(20):
                    buffer.find_match(f"Prompt 0 {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_worker, args=(0,)),
            threading.Thread(target=add_worker, args=(1,)),
            threading.Thread(target=find_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

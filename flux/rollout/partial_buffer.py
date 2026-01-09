"""
Partial trajectory buffer for APRIL strategy.

Stores incomplete/aborted trajectories for reuse in future generations.
Enables efficient continuation of long-running generations that were
aborted to meet batch timing requirements.
"""

from __future__ import annotations

import hashlib
import heapq
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator

from flux.core.trajectory import PartialTrajectory
from flux.core.types import PolicyVersion


@dataclass
class PartialEntry:
    """Entry in the partial buffer."""

    trajectory: PartialTrajectory
    added_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime | None = None
    prompt_hash: str = ""
    priority: float = 0.0  # Higher = more likely to be reused

    def __post_init__(self) -> None:
        if not self.prompt_hash:
            self.prompt_hash = self._compute_hash(self.trajectory.prompt)

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute hash of prompt for fast lookup."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def compute_priority(self, current_version: int) -> float:
        """Compute priority score for this entry.

        Higher priority = more valuable for reuse.
        Factors:
        - Response length (longer = more valuable)
        - Version freshness (newer = more valuable)
        - Log prob availability (has log probs = more valuable)
        """
        traj = self.trajectory
        length_score = min(1.0, len(traj.tokens) / 512)
        version_gap = current_version - traj.version.version_id
        version_score = max(0.0, 1.0 - version_gap / 10)
        logprob_score = 1.0 if traj.log_probs else 0.5

        self.priority = 0.4 * length_score + 0.4 * version_score + 0.2 * logprob_score
        return self.priority

    def __lt__(self, other: "PartialEntry") -> bool:
        """For heap ordering (lower priority = lower in heap)."""
        return self.priority < other.priority


class PartialTrajectoryBuffer:
    """Buffer for storing and retrieving partial trajectories.

    Enables the APRIL reuse strategy by storing incomplete trajectories
    and matching them with future prompts for continuation.

    Features:
    - Fast lookup by prompt hash
    - LRU-style eviction when capacity reached
    - Priority-based retrieval (fresher, longer partials first)
    - Thread-safe operations

    Example:
        buffer = PartialTrajectoryBuffer(max_size=1000)

        # Store partial trajectory
        buffer.add(partial_trajectory)

        # Find matching partial for a prompt
        partial = buffer.find_match(prompt)
        if partial:
            # Continue generation from partial
            continuation_prompt = prompt + partial.response

        # Remove used partial
        buffer.remove(partial.id)
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_staleness: int = 5,
        min_length_threshold: int = 32,
    ) -> None:
        """Initialize the partial buffer.

        Args:
            max_size: Maximum number of entries to store.
            max_staleness: Maximum version gap before entry is considered stale.
            min_length_threshold: Minimum response length to store.
        """
        self.max_size = max_size
        self.max_staleness = max_staleness
        self.min_length_threshold = min_length_threshold

        # Storage
        self._entries: dict[str, PartialEntry] = {}  # id -> entry
        self._hash_index: dict[str, list[str]] = defaultdict(list)  # hash -> [ids]

        # Priority heap for eviction
        self._priority_heap: list[PartialEntry] = []

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._total_added = 0
        self._total_matched = 0
        self._total_evicted = 0

    @property
    def size(self) -> int:
        """Current number of entries."""
        with self._lock:
            return len(self._entries)

    @property
    def is_empty(self) -> bool:
        """Whether buffer is empty."""
        return self.size == 0

    def add(
        self,
        trajectory: PartialTrajectory,
        priority: float | None = None,
    ) -> bool:
        """Add a partial trajectory to the buffer.

        Args:
            trajectory: The partial trajectory to store.
            priority: Optional priority override.

        Returns:
            True if added successfully.
        """
        # Check minimum length
        if len(trajectory.tokens) < self.min_length_threshold:
            return False

        with self._lock:
            # Create entry
            entry = PartialEntry(trajectory=trajectory)
            if priority is not None:
                entry.priority = priority

            # Evict if at capacity
            while len(self._entries) >= self.max_size:
                self._evict_lowest_priority()

            # Add to storage
            self._entries[trajectory.id] = entry
            self._hash_index[entry.prompt_hash].append(trajectory.id)
            heapq.heappush(self._priority_heap, entry)

            self._total_added += 1
            return True

    def add_batch(self, trajectories: list[PartialTrajectory]) -> int:
        """Add multiple partial trajectories.

        Args:
            trajectories: List of partial trajectories.

        Returns:
            Number successfully added.
        """
        count = 0
        for traj in trajectories:
            if self.add(traj):
                count += 1
        return count

    def find_match(
        self,
        prompt: str,
        current_version: int | None = None,
        remove: bool = False,
    ) -> PartialTrajectory | None:
        """Find a matching partial trajectory for a prompt.

        Args:
            prompt: The prompt to match.
            current_version: Current policy version for staleness check.
            remove: If True, remove the matched entry from buffer.

        Returns:
            Matching PartialTrajectory or None if no match.
        """
        prompt_hash = PartialEntry._compute_hash(prompt)

        with self._lock:
            if prompt_hash not in self._hash_index:
                return None

            # Find best match among entries with same hash
            best_entry: PartialEntry | None = None
            best_priority = -1.0

            for entry_id in self._hash_index[prompt_hash]:
                if entry_id not in self._entries:
                    continue

                entry = self._entries[entry_id]

                # Verify exact prompt match (hash collision possible)
                if entry.trajectory.prompt != prompt:
                    continue

                # Check staleness
                if current_version is not None:
                    version_gap = current_version - entry.trajectory.version.version_id
                    if version_gap > self.max_staleness:
                        continue

                # Update priority
                entry.compute_priority(current_version or 0)

                if entry.priority > best_priority:
                    best_entry = entry
                    best_priority = entry.priority

            if best_entry is None:
                return None

            best_entry.touch()
            self._total_matched += 1

            if remove:
                self._remove_entry(best_entry.trajectory.id)

            return best_entry.trajectory

    def find_matches(
        self,
        prompts: list[str],
        current_version: int | None = None,
        remove: bool = False,
    ) -> dict[str, PartialTrajectory]:
        """Find matching partials for multiple prompts.

        Args:
            prompts: List of prompts to match.
            current_version: Current policy version.
            remove: If True, remove matched entries.

        Returns:
            Dict mapping prompt to matched PartialTrajectory.
        """
        matches = {}
        for prompt in prompts:
            match = self.find_match(prompt, current_version, remove)
            if match is not None:
                matches[prompt] = match
        return matches

    def remove(self, trajectory_id: str) -> bool:
        """Remove a partial trajectory by ID.

        Args:
            trajectory_id: The trajectory ID to remove.

        Returns:
            True if removed successfully.
        """
        with self._lock:
            return self._remove_entry(trajectory_id)

    def _remove_entry(self, trajectory_id: str) -> bool:
        """Remove an entry from storage (internal, no lock)."""
        if trajectory_id not in self._entries:
            return False

        entry = self._entries.pop(trajectory_id)

        # Remove from hash index
        if entry.prompt_hash in self._hash_index:
            ids = self._hash_index[entry.prompt_hash]
            if trajectory_id in ids:
                ids.remove(trajectory_id)
            if not ids:
                del self._hash_index[entry.prompt_hash]

        return True

    def _evict_lowest_priority(self) -> bool:
        """Evict the lowest priority entry."""
        while self._priority_heap:
            entry = heapq.heappop(self._priority_heap)
            # Check if entry is still in storage (may have been removed)
            if entry.trajectory.id in self._entries:
                self._remove_entry(entry.trajectory.id)
                self._total_evicted += 1
                return True
        return False

    def clear(self) -> int:
        """Clear all entries from buffer.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._hash_index.clear()
            self._priority_heap.clear()
            return count

    def cleanup_stale(self, current_version: int) -> int:
        """Remove entries that are too stale.

        Args:
            current_version: Current policy version.

        Returns:
            Number of entries removed.
        """
        removed = 0
        with self._lock:
            stale_ids = []
            for entry_id, entry in self._entries.items():
                version_gap = current_version - entry.trajectory.version.version_id
                if version_gap > self.max_staleness:
                    stale_ids.append(entry_id)

            for entry_id in stale_ids:
                self._remove_entry(entry_id)
                removed += 1

        return removed

    def update_priorities(self, current_version: int) -> None:
        """Update priorities for all entries.

        Call periodically to keep priorities fresh.

        Args:
            current_version: Current policy version.
        """
        with self._lock:
            for entry in self._entries.values():
                entry.compute_priority(current_version)

            # Rebuild heap with updated priorities
            self._priority_heap = list(self._entries.values())
            heapq.heapify(self._priority_heap)

    def get_by_version(
        self, version_id: int
    ) -> list[PartialTrajectory]:
        """Get all partials generated at a specific version.

        Args:
            version_id: The policy version ID.

        Returns:
            List of partial trajectories.
        """
        with self._lock:
            return [
                entry.trajectory
                for entry in self._entries.values()
                if entry.trajectory.version.version_id == version_id
            ]

    def iter_entries(self) -> Iterator[PartialEntry]:
        """Iterate over all entries.

        Yields:
            PartialEntry objects.
        """
        with self._lock:
            yield from list(self._entries.values())

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dict with buffer statistics.
        """
        with self._lock:
            lengths = [
                len(e.trajectory.tokens)
                for e in self._entries.values()
            ]
            priorities = [e.priority for e in self._entries.values()]

            return {
                "size": len(self._entries),
                "max_size": self.max_size,
                "total_added": self._total_added,
                "total_matched": self._total_matched,
                "total_evicted": self._total_evicted,
                "match_rate": (
                    self._total_matched / max(1, self._total_added)
                ),
                "mean_length": (
                    sum(lengths) / max(1, len(lengths))
                ) if lengths else 0,
                "mean_priority": (
                    sum(priorities) / max(1, len(priorities))
                ) if priorities else 0,
                "unique_prompts": len(self._hash_index),
            }

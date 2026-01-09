"""
Tests for StreamingRolloutManager helpers.
"""

import pytest

from flux.core.config import RolloutConfig
from flux.core.types import PolicyVersion, TrajectoryStatus
from flux.rollout.manager import RolloutRequest, StreamingRolloutManager
from flux.rollout.sglang_client import GenerationResult, GenerationStatus


class DummyClient:
    """Minimal client stub for manager construction."""


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (GenerationStatus.ABORTED, TrajectoryStatus.ABORTED),
        (GenerationStatus.FAILED, TrajectoryStatus.FAILED),
    ],
)
def test_create_trajectory_propagates_aborted_failed(status, expected):
    manager = StreamingRolloutManager(client=DummyClient(), config=RolloutConfig())
    request = RolloutRequest(request_id="req-1", prompt="Hello")
    version = PolicyVersion(version_id=3)
    result = GenerationResult(
        request_id="req-1",
        status=status,
        prompt="Hello",
        response="World",
        tokens=[1, 2, 3],
        log_probs=[0.0, 0.0, 0.0],
        prompt_tokens=1,
        completion_tokens=2,
    )

    trajectory = manager._create_trajectory(request, result, version)

    assert trajectory.status == expected

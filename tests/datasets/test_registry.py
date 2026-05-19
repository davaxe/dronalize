import pytest

from dronalize.config.models import RequireFramesSpec
from dronalize.datasets import DatasetDescriptor, get_dataset, list_datasets


@pytest.mark.parametrize("name", list_datasets())
def test_all_available_builtin_datasets_resolve_to_matching_descriptors(name: str) -> None:
    descriptor = get_dataset(name)
    assert isinstance(descriptor, DatasetDescriptor)
    assert descriptor.name == name


@pytest.mark.parametrize("name", list_datasets())
def test_builtin_default_screening_requires_agent_at_prediction_frame(name: str) -> None:
    descriptor = get_dataset(name)
    screening = descriptor.default_config.screening

    assert screening is not None
    assert "min_samples" in screening.cleanup
    assert "require_frames" in screening.agent
    rule = screening.agent["require_frames"]
    assert isinstance(rule, RequireFramesSpec)
    assert rule.frames == (descriptor.default_config.scenes.history_frames - 1,)
    assert rule.require is not None
    assert rule.require.absolute == 1
    assert rule.require.relative is None

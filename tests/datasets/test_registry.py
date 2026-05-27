import pytest

from dronalize.config.models import RequireFramesSpec
from dronalize.datasets import DatasetDescriptor, get_dataset, list_datasets


@pytest.mark.parametrize("name", list_datasets())
def test_builtin_datasets_resolve(name: str) -> None:
    descriptor = get_dataset(name)
    assert isinstance(descriptor, DatasetDescriptor)
    assert descriptor.name == name


@pytest.mark.parametrize("name", list_datasets())
def test_builtin_screening_requires_observation_end(name: str) -> None:
    descriptor = get_dataset(name)
    screening = descriptor.default_config.screening

    assert screening is not None
    assert "min_samples" in screening.cleanup
    assert "require_frames" in screening.agent
    rule = screening.agent["require_frames"]
    assert isinstance(rule, RequireFramesSpec)
    assert descriptor.default_config.scenes.default_observation_length is not None
    assert rule.frames == (descriptor.default_config.scenes.default_observation_length - 1,)
    assert rule.require is not None
    assert rule.require.absolute == 1
    assert rule.require.relative is None


@pytest.mark.parametrize("name", list_datasets())
def test_builtin_datasets_have_temporal_support(name: str) -> None:
    descriptor = get_dataset(name)
    temporal = descriptor.temporal_support

    if descriptor.name == "a43":
        assert temporal is None
        return

    assert temporal is not None
    assert temporal.source_frame_bounds.min_frames is not None
    assert temporal.source_frame_bounds.max_frames is not None
    assert temporal.source_frame_bounds.min_frames <= temporal.source_frame_bounds.max_frames
    assert temporal.windowing.max_window_frames == temporal.source_frame_bounds.max_frames
    assert temporal.windowing.default_policy in temporal.windowing.supported_policies

import pytest

from dronalize.datasets import DatasetDescriptor, get_dataset, list_datasets


@pytest.mark.parametrize("name", list_datasets())
def test_all_available_builtin_datasets_resolve_to_matching_descriptors(name: str) -> None:
    descriptor = get_dataset(name)
    assert isinstance(descriptor, DatasetDescriptor)
    assert descriptor.name == name

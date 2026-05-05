import pytest

from dronalize.datasets import DatasetSpec, available, get


@pytest.mark.parametrize("name", available())
def test_all_available_builtin_datasets_resolve_to_matching_descriptors(name: str) -> None:
    descriptor = get(name)
    assert isinstance(descriptor, DatasetSpec)
    assert descriptor.name == name

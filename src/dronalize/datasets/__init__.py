"""Dataset registry and explicit dataset-spec surface."""

from dronalize.datasets.registry import DatasetSpec, ResourcesFactory, available, get, register

__all__ = ["DatasetSpec", "ResourcesFactory", "available", "get", "register"]

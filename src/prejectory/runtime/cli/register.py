"""Provides functionality to register custom datasets from CLI."""

import importlib
from collections.abc import Iterable, Iterator
from types import ModuleType
from typing import cast

from prejectory.core.errors import CliError, cli_usage_error
from prejectory.datasets import DatasetDescriptor, register_dataset

_REGISTER_HOOK_NAME = "register_prejectory_datasets"


def register_custom_datasets(dataset_modules: list[str] | None) -> None:
    """Registers custom datasets from the specified dataset modules.

    Parameters
    ----------
    dataset_modules : list[str] | None
        List of dataset module names to import and register datasets from. If
        None, no modules are imported and no datasets are registered.


    Notes
    -----
    There are two ways the modules specified in `dataset_modules` can register
    datasets:
        1. Define a function named `register_prejectory_datasets` that returns a
        single `DatasetDescriptor` or an iterable of `DatasetDescriptor` objects.
        2. Importing the module has side effects that register datasets directly
        with the `register` function.

    Option 1 is the recommended approach for new dataset modules, as it provides
    a clear and explicit way to register datasets.

    """  # ruff: ignore[non-imperative-mood]
    if not dataset_modules:
        return

    for module_name in dataset_modules:
        module = _import_dataset_module(module_name)
        hook = getattr(module, _REGISTER_HOOK_NAME, None)
        if hook is None:
            continue
        if not callable(hook):
            msg = f"Dataset module '{module_name}' defines non-callable {_REGISTER_HOOK_NAME}."
            raise cli_usage_error(msg)
        try:
            descriptors = cast("DatasetDescriptor | Iterable[object] | None", hook())
        except CliError:
            raise
        except Exception as exc:
            msg = f"Dataset module '{module_name}' failed while registering datasets."
            raise CliError(msg) from exc
        for descriptor in _normalize_dataset_descriptors(descriptors, module_name):
            register_dataset(descriptor)


def _import_dataset_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        requested_root = module_name.partition(".")[0]

        if exc.name in {module_name, requested_root}:
            msg = f"Could not import dataset module '{module_name}'."
        else:
            msg = (
                f"Could not import dataset module '{module_name}' because "
                f"dependency '{exc.name}' is missing."
            )

        raise cli_usage_error(msg) from exc


def _normalize_dataset_descriptors(
    descriptors: DatasetDescriptor | Iterable[DatasetDescriptor | object] | None,
    module_name: str,
) -> Iterator[DatasetDescriptor]:
    if descriptors is None:
        return

    if isinstance(descriptors, DatasetDescriptor):
        yield descriptors
        return

    try:
        iterator: Iterator[DatasetDescriptor | object] = iter(descriptors)
    except TypeError as exc:
        msg = (
            f"Dataset module '{module_name}' returned unsupported value from {_REGISTER_HOOK_NAME}."
        )
        raise cli_usage_error(msg) from exc

    for descriptor in iterator:
        if not isinstance(descriptor, DatasetDescriptor):
            msg = (
                f"Dataset module '{module_name}' returned "
                f"{type(descriptor).__name__}, expected DatasetDescriptor."
            )
            raise cli_usage_error(msg)

        yield descriptor

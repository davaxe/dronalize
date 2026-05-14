"""Provides functionality to register custom datasets from CLI."""

import importlib
from collections.abc import Callable, Iterable, Iterator
from types import ModuleType
from typing import cast

import click

from dronalize.datasets import DatasetDescriptor, register_dataset

_REGISTER_HOOK_NAME = "register_dronalize_datasets"

DatasetHook = Callable[[], DatasetDescriptor | Iterable[DatasetDescriptor] | None]


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
        1. Define a function named `register_dronalize_datasets` that returns a
        single `DatasetDescriptor` or an iterable of `DatasetDescriptor` objects.
        2. Importing the module has side effects that register datasets directly
        with the `register` function.

    Option 1 is the recommended approach for new dataset modules, as it provides
    a clear and explicit way to register datasets.

    """  # noqa: D401
    if not dataset_modules:
        return

    for module_name in dataset_modules:
        module = _import_dataset_module(module_name)
        hook = _get_dataset_register_hook(module, module_name)

        if hook is None:
            continue

        specs = _call_dataset_register_hook(hook, module_name)

        for spec in _normalize_dataset_descriptors(specs, module_name):
            register_dataset(spec)


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

        raise click.UsageError(msg) from exc


def _get_dataset_register_hook(module: ModuleType, module_name: str) -> DatasetHook | None:
    hook = getattr(module, _REGISTER_HOOK_NAME, None)

    if hook is None:
        return None

    if not callable(hook):
        msg = f"Dataset module '{module_name}' defines non-callable {_REGISTER_HOOK_NAME}."
        raise click.UsageError(msg)

    # Assume it follows the expected signature, this will be validated when it
    # is used.
    return cast("DatasetHook", hook)


def _call_dataset_register_hook(
    hook: DatasetHook, module_name: str
) -> DatasetDescriptor | Iterable[DatasetDescriptor] | None:
    try:
        return hook()
    except click.ClickException:
        raise
    except Exception as exc:
        msg = f"Dataset module '{module_name}' failed while registering datasets."
        raise click.ClickException(msg) from exc


def _normalize_dataset_descriptors(
    specs: DatasetDescriptor | Iterable[DatasetDescriptor | object] | None, module_name: str
) -> Iterable[DatasetDescriptor]:
    if specs is None:
        return ()

    if isinstance(specs, DatasetDescriptor):
        return (specs,)

    try:
        iterator: Iterator[DatasetDescriptor | object] = iter(specs)
    except TypeError as exc:
        msg = (
            f"Dataset module '{module_name}' returned unsupported value from {_REGISTER_HOOK_NAME}."
        )
        raise click.UsageError(msg) from exc

    normalized_specs: list[DatasetDescriptor] = []

    for spec in iterator:
        if not isinstance(spec, DatasetDescriptor):
            msg = (
                f"Dataset module '{module_name}' returned "
                f"{type(spec).__name__}, expected DatasetDescriptor."
            )
            raise click.UsageError(msg)

        normalized_specs.append(spec)

    return normalized_specs

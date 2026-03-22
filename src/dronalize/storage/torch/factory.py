from __future__ import annotations

from typing import Any


def open_torch_dataset(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """Raise until torch dataset loading is implemented."""
    _ = (args, kwargs)
    msg = "Torch dataset loading is not implemented yet."
    raise NotImplementedError(msg)

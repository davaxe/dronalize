from __future__ import annotations

from dronalize.datasets import (
    a43,
    ad4che,
    apolloscape,
    argoverse1,
    argoverse2,
    eth_ucy,
    exid,
    highd,
    i80,
    ind,
    interact,
    lyft,
    nuscenes,
    opendd,
    round,
    sind,
    unid,
    us101,
    vod,
    waymo,
)
from dronalize.datasets.registry import available, get

# Import built-in dataset packages for registration side effects.
_ = (
    a43,
    ad4che,
    apolloscape,
    argoverse1,
    argoverse2,
    eth_ucy,
    exid,
    highd,
    i80,
    ind,
    interact,
    lyft,
    nuscenes,
    opendd,
    round,
    sind,
    unid,
    us101,
    vod,
    waymo,
)

__all__ = ["available", "get"]

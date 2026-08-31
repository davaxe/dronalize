from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LeanObjectState(_message.Message):
    __slots__ = ["center_x", "center_y", "heading", "valid", "velocity_x", "velocity_y"]
    CENTER_X_FIELD_NUMBER: _ClassVar[int]
    CENTER_Y_FIELD_NUMBER: _ClassVar[int]
    HEADING_FIELD_NUMBER: _ClassVar[int]
    VALID_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_X_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_Y_FIELD_NUMBER: _ClassVar[int]
    center_x: float
    center_y: float
    heading: float
    valid: bool
    velocity_x: float
    velocity_y: float
    def __init__(self, center_x: _Optional[float] = ..., center_y: _Optional[float] = ..., heading: _Optional[float] = ..., velocity_x: _Optional[float] = ..., velocity_y: _Optional[float] = ..., valid: bool = ...) -> None: ...

class LeanScenario(_message.Message):
    __slots__ = ["sdc_track_index", "tracks"]
    SDC_TRACK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    sdc_track_index: int
    tracks: _containers.RepeatedCompositeFieldContainer[LeanTrack]
    def __init__(self, sdc_track_index: _Optional[int] = ..., tracks: _Optional[_Iterable[_Union[LeanTrack, _Mapping]]] = ...) -> None: ...

class LeanTrack(_message.Message):
    __slots__ = ["id", "object_type", "states"]
    ID_FIELD_NUMBER: _ClassVar[int]
    OBJECT_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATES_FIELD_NUMBER: _ClassVar[int]
    id: int
    object_type: int
    states: _containers.RepeatedCompositeFieldContainer[LeanObjectState]
    def __init__(self, id: _Optional[int] = ..., object_type: _Optional[int] = ..., states: _Optional[_Iterable[_Union[LeanObjectState, _Mapping]]] = ...) -> None: ...

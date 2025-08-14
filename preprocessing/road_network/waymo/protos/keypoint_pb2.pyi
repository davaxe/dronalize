from protos import vector_pb2 as _vector_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
KEYPOINT_TYPE_FOREHEAD: KeypointType
KEYPOINT_TYPE_HEAD_CENTER: KeypointType
KEYPOINT_TYPE_LEFT_ANKLE: KeypointType
KEYPOINT_TYPE_LEFT_ELBOW: KeypointType
KEYPOINT_TYPE_LEFT_HIP: KeypointType
KEYPOINT_TYPE_LEFT_KNEE: KeypointType
KEYPOINT_TYPE_LEFT_SHOULDER: KeypointType
KEYPOINT_TYPE_LEFT_WRIST: KeypointType
KEYPOINT_TYPE_NOSE: KeypointType
KEYPOINT_TYPE_RIGHT_ANKLE: KeypointType
KEYPOINT_TYPE_RIGHT_ELBOW: KeypointType
KEYPOINT_TYPE_RIGHT_HIP: KeypointType
KEYPOINT_TYPE_RIGHT_KNEE: KeypointType
KEYPOINT_TYPE_RIGHT_SHOULDER: KeypointType
KEYPOINT_TYPE_RIGHT_WRIST: KeypointType
KEYPOINT_TYPE_UNSPECIFIED: KeypointType

class CameraKeypoint(_message.Message):
    __slots__ = ["keypoint_2d", "keypoint_3d", "type"]
    KEYPOINT_2D_FIELD_NUMBER: _ClassVar[int]
    KEYPOINT_3D_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    keypoint_2d: Keypoint2d
    keypoint_3d: Keypoint3d
    type: KeypointType
    def __init__(self, type: _Optional[_Union[KeypointType, str]] = ..., keypoint_2d: _Optional[_Union[Keypoint2d, _Mapping]] = ..., keypoint_3d: _Optional[_Union[Keypoint3d, _Mapping]] = ...) -> None: ...

class CameraKeypoints(_message.Message):
    __slots__ = ["keypoint"]
    KEYPOINT_FIELD_NUMBER: _ClassVar[int]
    keypoint: _containers.RepeatedCompositeFieldContainer[CameraKeypoint]
    def __init__(self, keypoint: _Optional[_Iterable[_Union[CameraKeypoint, _Mapping]]] = ...) -> None: ...

class Keypoint2d(_message.Message):
    __slots__ = ["location_px", "visibility"]
    Extensions: _python_message._ExtensionDict
    LOCATION_PX_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    location_px: _vector_pb2.Vector2d
    visibility: KeypointVisibility
    def __init__(self, location_px: _Optional[_Union[_vector_pb2.Vector2d, _Mapping]] = ..., visibility: _Optional[_Union[KeypointVisibility, _Mapping]] = ...) -> None: ...

class Keypoint3d(_message.Message):
    __slots__ = ["location_m", "visibility"]
    Extensions: _python_message._ExtensionDict
    LOCATION_M_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    location_m: _vector_pb2.Vector3d
    visibility: KeypointVisibility
    def __init__(self, location_m: _Optional[_Union[_vector_pb2.Vector3d, _Mapping]] = ..., visibility: _Optional[_Union[KeypointVisibility, _Mapping]] = ...) -> None: ...

class KeypointVisibility(_message.Message):
    __slots__ = ["is_occluded"]
    IS_OCCLUDED_FIELD_NUMBER: _ClassVar[int]
    is_occluded: bool
    def __init__(self, is_occluded: bool = ...) -> None: ...

class LaserKeypoint(_message.Message):
    __slots__ = ["keypoint_3d", "type"]
    KEYPOINT_3D_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    keypoint_3d: Keypoint3d
    type: KeypointType
    def __init__(self, type: _Optional[_Union[KeypointType, str]] = ..., keypoint_3d: _Optional[_Union[Keypoint3d, _Mapping]] = ...) -> None: ...

class LaserKeypoints(_message.Message):
    __slots__ = ["keypoint"]
    KEYPOINT_FIELD_NUMBER: _ClassVar[int]
    keypoint: _containers.RepeatedCompositeFieldContainer[LaserKeypoint]
    def __init__(self, keypoint: _Optional[_Iterable[_Union[LaserKeypoint, _Mapping]]] = ...) -> None: ...

class KeypointType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

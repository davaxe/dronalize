from protos import camera_tokens_pb2 as _camera_tokens_pb2
from protos import compressed_lidar_pb2 as _compressed_lidar_pb2
from protos import map_pb2 as _map_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DynamicMapState(_message.Message):
    __slots__ = ["lane_states"]
    LANE_STATES_FIELD_NUMBER: _ClassVar[int]
    lane_states: _containers.RepeatedCompositeFieldContainer[_map_pb2.TrafficSignalLaneState]
    def __init__(self, lane_states: _Optional[_Iterable[_Union[_map_pb2.TrafficSignalLaneState, _Mapping]]] = ...) -> None: ...

class ObjectState(_message.Message):
    __slots__ = ["center_x", "center_y", "center_z", "heading", "height", "length", "valid", "velocity_x", "velocity_y", "width"]
    CENTER_X_FIELD_NUMBER: _ClassVar[int]
    CENTER_Y_FIELD_NUMBER: _ClassVar[int]
    CENTER_Z_FIELD_NUMBER: _ClassVar[int]
    HEADING_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    VALID_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_X_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_Y_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    center_x: float
    center_y: float
    center_z: float
    heading: float
    height: float
    length: float
    valid: bool
    velocity_x: float
    velocity_y: float
    width: float
    def __init__(self, center_x: _Optional[float] = ..., center_y: _Optional[float] = ..., center_z: _Optional[float] = ..., length: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ..., heading: _Optional[float] = ..., velocity_x: _Optional[float] = ..., velocity_y: _Optional[float] = ..., valid: bool = ...) -> None: ...

class RequiredPrediction(_message.Message):
    __slots__ = ["difficulty", "track_index"]
    class DifficultyLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    DIFFICULTY_FIELD_NUMBER: _ClassVar[int]
    LEVEL_1: RequiredPrediction.DifficultyLevel
    LEVEL_2: RequiredPrediction.DifficultyLevel
    NONE: RequiredPrediction.DifficultyLevel
    TRACK_INDEX_FIELD_NUMBER: _ClassVar[int]
    difficulty: RequiredPrediction.DifficultyLevel
    track_index: int
    def __init__(self, track_index: _Optional[int] = ..., difficulty: _Optional[_Union[RequiredPrediction.DifficultyLevel, str]] = ...) -> None: ...

class Scenario(_message.Message):
    __slots__ = ["compressed_frame_laser_data", "current_time_index", "dynamic_map_states", "frame_camera_tokens", "map_features", "objects_of_interest", "scenario_id", "sdc_track_index", "timestamps_seconds", "tracks", "tracks_to_predict"]
    COMPRESSED_FRAME_LASER_DATA_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TIME_INDEX_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_MAP_STATES_FIELD_NUMBER: _ClassVar[int]
    FRAME_CAMERA_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAP_FEATURES_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_OF_INTEREST_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_ID_FIELD_NUMBER: _ClassVar[int]
    SDC_TRACK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    TRACKS_TO_PREDICT_FIELD_NUMBER: _ClassVar[int]
    compressed_frame_laser_data: _containers.RepeatedCompositeFieldContainer[_compressed_lidar_pb2.CompressedFrameLaserData]
    current_time_index: int
    dynamic_map_states: _containers.RepeatedCompositeFieldContainer[DynamicMapState]
    frame_camera_tokens: _containers.RepeatedCompositeFieldContainer[_camera_tokens_pb2.FrameCameraTokens]
    map_features: _containers.RepeatedCompositeFieldContainer[_map_pb2.MapFeature]
    objects_of_interest: _containers.RepeatedScalarFieldContainer[int]
    scenario_id: str
    sdc_track_index: int
    timestamps_seconds: _containers.RepeatedScalarFieldContainer[float]
    tracks: _containers.RepeatedCompositeFieldContainer[Track]
    tracks_to_predict: _containers.RepeatedCompositeFieldContainer[RequiredPrediction]
    def __init__(self, scenario_id: _Optional[str] = ..., timestamps_seconds: _Optional[_Iterable[float]] = ..., current_time_index: _Optional[int] = ..., tracks: _Optional[_Iterable[_Union[Track, _Mapping]]] = ..., dynamic_map_states: _Optional[_Iterable[_Union[DynamicMapState, _Mapping]]] = ..., map_features: _Optional[_Iterable[_Union[_map_pb2.MapFeature, _Mapping]]] = ..., sdc_track_index: _Optional[int] = ..., objects_of_interest: _Optional[_Iterable[int]] = ..., tracks_to_predict: _Optional[_Iterable[_Union[RequiredPrediction, _Mapping]]] = ..., compressed_frame_laser_data: _Optional[_Iterable[_Union[_compressed_lidar_pb2.CompressedFrameLaserData, _Mapping]]] = ..., frame_camera_tokens: _Optional[_Iterable[_Union[_camera_tokens_pb2.FrameCameraTokens, _Mapping]]] = ...) -> None: ...

class Track(_message.Message):
    __slots__ = ["id", "object_type", "states"]
    class ObjectType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ID_FIELD_NUMBER: _ClassVar[int]
    OBJECT_TYPE_FIELD_NUMBER: _ClassVar[int]
    STATES_FIELD_NUMBER: _ClassVar[int]
    TYPE_CYCLIST: Track.ObjectType
    TYPE_OTHER: Track.ObjectType
    TYPE_PEDESTRIAN: Track.ObjectType
    TYPE_UNSET: Track.ObjectType
    TYPE_VEHICLE: Track.ObjectType
    id: int
    object_type: Track.ObjectType
    states: _containers.RepeatedCompositeFieldContainer[ObjectState]
    def __init__(self, id: _Optional[int] = ..., object_type: _Optional[_Union[Track.ObjectType, str]] = ..., states: _Optional[_Iterable[_Union[ObjectState, _Mapping]]] = ...) -> None: ...

from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Crosswalk(_message.Message):
    __slots__ = ["polygon"]
    POLYGON_FIELD_NUMBER: _ClassVar[int]
    polygon: _containers.RepeatedCompositeFieldContainer[MapPoint]
    def __init__(self, polygon: _Optional[_Iterable[_Union[MapPoint, _Mapping]]] = ...) -> None: ...

class Driveway(_message.Message):
    __slots__ = ["polygon"]
    POLYGON_FIELD_NUMBER: _ClassVar[int]
    polygon: _containers.RepeatedCompositeFieldContainer[MapPoint]
    def __init__(self, polygon: _Optional[_Iterable[_Union[MapPoint, _Mapping]]] = ...) -> None: ...

class LaneCenter(_message.Message):
    __slots__ = ["polyline", "speed_limit_mph"]
    POLYLINE_FIELD_NUMBER: _ClassVar[int]
    SPEED_LIMIT_MPH_FIELD_NUMBER: _ClassVar[int]
    polyline: _containers.RepeatedCompositeFieldContainer[MapPoint]
    speed_limit_mph: float
    def __init__(self, speed_limit_mph: _Optional[float] = ..., polyline: _Optional[_Iterable[_Union[MapPoint, _Mapping]]] = ...) -> None: ...

class LeanMapContainer(_message.Message):
    __slots__ = ["map_features"]
    MAP_FEATURES_FIELD_NUMBER: _ClassVar[int]
    map_features: _containers.RepeatedCompositeFieldContainer[MapFeature]
    def __init__(self, map_features: _Optional[_Iterable[_Union[MapFeature, _Mapping]]] = ...) -> None: ...

class MapFeature(_message.Message):
    __slots__ = ["crosswalk", "driveway", "id", "lane", "road_edge", "road_line", "speed_bump", "stop_sign"]
    CROSSWALK_FIELD_NUMBER: _ClassVar[int]
    DRIVEWAY_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LANE_FIELD_NUMBER: _ClassVar[int]
    ROAD_EDGE_FIELD_NUMBER: _ClassVar[int]
    ROAD_LINE_FIELD_NUMBER: _ClassVar[int]
    SPEED_BUMP_FIELD_NUMBER: _ClassVar[int]
    STOP_SIGN_FIELD_NUMBER: _ClassVar[int]
    crosswalk: Crosswalk
    driveway: Driveway
    id: int
    lane: LaneCenter
    road_edge: RoadEdge
    road_line: RoadLine
    speed_bump: SpeedBump
    stop_sign: StopSign
    def __init__(self, id: _Optional[int] = ..., lane: _Optional[_Union[LaneCenter, _Mapping]] = ..., road_line: _Optional[_Union[RoadLine, _Mapping]] = ..., road_edge: _Optional[_Union[RoadEdge, _Mapping]] = ..., stop_sign: _Optional[_Union[StopSign, _Mapping]] = ..., crosswalk: _Optional[_Union[Crosswalk, _Mapping]] = ..., speed_bump: _Optional[_Union[SpeedBump, _Mapping]] = ..., driveway: _Optional[_Union[Driveway, _Mapping]] = ...) -> None: ...

class MapPoint(_message.Message):
    __slots__ = ["x", "y", "z"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class RoadEdge(_message.Message):
    __slots__ = ["polyline", "type"]
    class RoadEdgeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    POLYLINE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TYPE_ROAD_EDGE_BOUNDARY: RoadEdge.RoadEdgeType
    TYPE_ROAD_EDGE_MEDIAN: RoadEdge.RoadEdgeType
    TYPE_UNKNOWN: RoadEdge.RoadEdgeType
    polyline: _containers.RepeatedCompositeFieldContainer[MapPoint]
    type: RoadEdge.RoadEdgeType
    def __init__(self, type: _Optional[_Union[RoadEdge.RoadEdgeType, str]] = ..., polyline: _Optional[_Iterable[_Union[MapPoint, _Mapping]]] = ...) -> None: ...

class RoadLine(_message.Message):
    __slots__ = ["polyline", "type"]
    class RoadLineType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    POLYLINE_FIELD_NUMBER: _ClassVar[int]
    TYPE_BROKEN_DOUBLE_YELLOW: RoadLine.RoadLineType
    TYPE_BROKEN_SINGLE_WHITE: RoadLine.RoadLineType
    TYPE_BROKEN_SINGLE_YELLOW: RoadLine.RoadLineType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TYPE_PASSING_DOUBLE_YELLOW: RoadLine.RoadLineType
    TYPE_SOLID_DOUBLE_WHITE: RoadLine.RoadLineType
    TYPE_SOLID_DOUBLE_YELLOW: RoadLine.RoadLineType
    TYPE_SOLID_SINGLE_WHITE: RoadLine.RoadLineType
    TYPE_SOLID_SINGLE_YELLOW: RoadLine.RoadLineType
    TYPE_UNKNOWN: RoadLine.RoadLineType
    polyline: _containers.RepeatedCompositeFieldContainer[MapPoint]
    type: RoadLine.RoadLineType
    def __init__(self, type: _Optional[_Union[RoadLine.RoadLineType, str]] = ..., polyline: _Optional[_Iterable[_Union[MapPoint, _Mapping]]] = ...) -> None: ...

class SpeedBump(_message.Message):
    __slots__ = ["polygon"]
    POLYGON_FIELD_NUMBER: _ClassVar[int]
    polygon: _containers.RepeatedCompositeFieldContainer[MapPoint]
    def __init__(self, polygon: _Optional[_Iterable[_Union[MapPoint, _Mapping]]] = ...) -> None: ...

class StopSign(_message.Message):
    __slots__ = ["lane", "position"]
    LANE_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    lane: _containers.RepeatedScalarFieldContainer[int]
    position: MapPoint
    def __init__(self, lane: _Optional[_Iterable[int]] = ..., position: _Optional[_Union[MapPoint, _Mapping]] = ...) -> None: ...

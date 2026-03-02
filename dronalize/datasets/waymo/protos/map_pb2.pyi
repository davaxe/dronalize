from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoundarySegment(_message.Message):
    __slots__ = ["boundary_feature_id", "boundary_type", "lane_end_index", "lane_start_index"]
    BOUNDARY_FEATURE_ID_FIELD_NUMBER: _ClassVar[int]
    BOUNDARY_TYPE_FIELD_NUMBER: _ClassVar[int]
    LANE_END_INDEX_FIELD_NUMBER: _ClassVar[int]
    LANE_START_INDEX_FIELD_NUMBER: _ClassVar[int]
    boundary_feature_id: int
    boundary_type: RoadLine.RoadLineType
    lane_end_index: int
    lane_start_index: int
    def __init__(self, lane_start_index: _Optional[int] = ..., lane_end_index: _Optional[int] = ..., boundary_feature_id: _Optional[int] = ..., boundary_type: _Optional[_Union[RoadLine.RoadLineType, str]] = ...) -> None: ...

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

class DynamicState(_message.Message):
    __slots__ = ["lane_states", "timestamp_seconds"]
    LANE_STATES_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_SECONDS_FIELD_NUMBER: _ClassVar[int]
    lane_states: _containers.RepeatedCompositeFieldContainer[TrafficSignalLaneState]
    timestamp_seconds: float
    def __init__(self, timestamp_seconds: _Optional[float] = ..., lane_states: _Optional[_Iterable[_Union[TrafficSignalLaneState, _Mapping]]] = ...) -> None: ...

class LaneCenter(_message.Message):
    __slots__ = ["entry_lanes", "exit_lanes", "interpolating", "left_boundaries", "left_neighbors", "polyline", "right_boundaries", "right_neighbors", "speed_limit_mph", "type"]
    class LaneType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ENTRY_LANES_FIELD_NUMBER: _ClassVar[int]
    EXIT_LANES_FIELD_NUMBER: _ClassVar[int]
    INTERPOLATING_FIELD_NUMBER: _ClassVar[int]
    LEFT_BOUNDARIES_FIELD_NUMBER: _ClassVar[int]
    LEFT_NEIGHBORS_FIELD_NUMBER: _ClassVar[int]
    POLYLINE_FIELD_NUMBER: _ClassVar[int]
    RIGHT_BOUNDARIES_FIELD_NUMBER: _ClassVar[int]
    RIGHT_NEIGHBORS_FIELD_NUMBER: _ClassVar[int]
    SPEED_LIMIT_MPH_FIELD_NUMBER: _ClassVar[int]
    TYPE_BIKE_LANE: LaneCenter.LaneType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FREEWAY: LaneCenter.LaneType
    TYPE_SURFACE_STREET: LaneCenter.LaneType
    TYPE_UNDEFINED: LaneCenter.LaneType
    entry_lanes: _containers.RepeatedScalarFieldContainer[int]
    exit_lanes: _containers.RepeatedScalarFieldContainer[int]
    interpolating: bool
    left_boundaries: _containers.RepeatedCompositeFieldContainer[BoundarySegment]
    left_neighbors: _containers.RepeatedCompositeFieldContainer[LaneNeighbor]
    polyline: _containers.RepeatedCompositeFieldContainer[MapPoint]
    right_boundaries: _containers.RepeatedCompositeFieldContainer[BoundarySegment]
    right_neighbors: _containers.RepeatedCompositeFieldContainer[LaneNeighbor]
    speed_limit_mph: float
    type: LaneCenter.LaneType
    def __init__(self, speed_limit_mph: _Optional[float] = ..., type: _Optional[_Union[LaneCenter.LaneType, str]] = ..., interpolating: bool = ..., polyline: _Optional[_Iterable[_Union[MapPoint, _Mapping]]] = ..., entry_lanes: _Optional[_Iterable[int]] = ..., exit_lanes: _Optional[_Iterable[int]] = ..., left_boundaries: _Optional[_Iterable[_Union[BoundarySegment, _Mapping]]] = ..., right_boundaries: _Optional[_Iterable[_Union[BoundarySegment, _Mapping]]] = ..., left_neighbors: _Optional[_Iterable[_Union[LaneNeighbor, _Mapping]]] = ..., right_neighbors: _Optional[_Iterable[_Union[LaneNeighbor, _Mapping]]] = ...) -> None: ...

class LaneNeighbor(_message.Message):
    __slots__ = ["boundaries", "feature_id", "neighbor_end_index", "neighbor_start_index", "self_end_index", "self_start_index"]
    BOUNDARIES_FIELD_NUMBER: _ClassVar[int]
    FEATURE_ID_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_END_INDEX_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_START_INDEX_FIELD_NUMBER: _ClassVar[int]
    SELF_END_INDEX_FIELD_NUMBER: _ClassVar[int]
    SELF_START_INDEX_FIELD_NUMBER: _ClassVar[int]
    boundaries: _containers.RepeatedCompositeFieldContainer[BoundarySegment]
    feature_id: int
    neighbor_end_index: int
    neighbor_start_index: int
    self_end_index: int
    self_start_index: int
    def __init__(self, feature_id: _Optional[int] = ..., self_start_index: _Optional[int] = ..., self_end_index: _Optional[int] = ..., neighbor_start_index: _Optional[int] = ..., neighbor_end_index: _Optional[int] = ..., boundaries: _Optional[_Iterable[_Union[BoundarySegment, _Mapping]]] = ...) -> None: ...

class Map(_message.Message):
    __slots__ = ["dynamic_states", "map_features"]
    DYNAMIC_STATES_FIELD_NUMBER: _ClassVar[int]
    MAP_FEATURES_FIELD_NUMBER: _ClassVar[int]
    dynamic_states: _containers.RepeatedCompositeFieldContainer[DynamicState]
    map_features: _containers.RepeatedCompositeFieldContainer[MapFeature]
    def __init__(self, map_features: _Optional[_Iterable[_Union[MapFeature, _Mapping]]] = ..., dynamic_states: _Optional[_Iterable[_Union[DynamicState, _Mapping]]] = ...) -> None: ...

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

class TrafficSignalLaneState(_message.Message):
    __slots__ = ["lane", "state", "stop_point"]
    class State(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    LANE_FIELD_NUMBER: _ClassVar[int]
    LANE_STATE_ARROW_CAUTION: TrafficSignalLaneState.State
    LANE_STATE_ARROW_GO: TrafficSignalLaneState.State
    LANE_STATE_ARROW_STOP: TrafficSignalLaneState.State
    LANE_STATE_CAUTION: TrafficSignalLaneState.State
    LANE_STATE_FLASHING_CAUTION: TrafficSignalLaneState.State
    LANE_STATE_FLASHING_STOP: TrafficSignalLaneState.State
    LANE_STATE_GO: TrafficSignalLaneState.State
    LANE_STATE_STOP: TrafficSignalLaneState.State
    LANE_STATE_UNKNOWN: TrafficSignalLaneState.State
    STATE_FIELD_NUMBER: _ClassVar[int]
    STOP_POINT_FIELD_NUMBER: _ClassVar[int]
    lane: int
    state: TrafficSignalLaneState.State
    stop_point: MapPoint
    def __init__(self, lane: _Optional[int] = ..., state: _Optional[_Union[TrafficSignalLaneState.State, str]] = ..., stop_point: _Optional[_Union[MapPoint, _Mapping]] = ...) -> None: ...

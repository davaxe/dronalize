from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AccessRestriction(_message.Message):
    __slots__ = ["type"]
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    NO_RESTRICTION: AccessRestriction.Type
    ONLY_BIKE: AccessRestriction.Type
    ONLY_BUS: AccessRestriction.Type
    ONLY_HOV: AccessRestriction.Type
    ONLY_TURN: AccessRestriction.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN: AccessRestriction.Type
    type: AccessRestriction.Type
    def __init__(self, type: _Optional[_Union[AccessRestriction.Type, str]] = ...) -> None: ...

class AnnotatedShape(_message.Message):
    __slots__ = ["administrative_boundary", "building", "drivable_surface_prior", "multipolygon", "name", "park", "region", "venue"]
    class Building(_message.Message):
        __slots__ = ["above_ground_floors", "height_meters", "type"]
        class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        ABOVE_GROUND_FLOORS_FIELD_NUMBER: _ClassVar[int]
        COMMERCIAL: AnnotatedShape.Building.Type
        EDUCATIONAL: AnnotatedShape.Building.Type
        EVENT: AnnotatedShape.Building.Type
        GOVERNMENTAL: AnnotatedShape.Building.Type
        HEIGHT_METERS_FIELD_NUMBER: _ClassVar[int]
        HOSPITAL: AnnotatedShape.Building.Type
        PARKING_STRUCTURE: AnnotatedShape.Building.Type
        RELIGIOUS: AnnotatedShape.Building.Type
        RESIDENTIAL: AnnotatedShape.Building.Type
        TRANSPORTATION: AnnotatedShape.Building.Type
        TYPE_FIELD_NUMBER: _ClassVar[int]
        UNKNOWN: AnnotatedShape.Building.Type
        above_ground_floors: int
        height_meters: int
        type: AnnotatedShape.Building.Type
        def __init__(self, type: _Optional[_Union[AnnotatedShape.Building.Type, str]] = ..., height_meters: _Optional[int] = ..., above_ground_floors: _Optional[int] = ...) -> None: ...
    class Region(_message.Message):
        __slots__ = ["source"]
        SOURCE_FIELD_NUMBER: _ClassVar[int]
        source: str
        def __init__(self, source: _Optional[str] = ...) -> None: ...
    ADMINISTRATIVE_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    BUILDING_FIELD_NUMBER: _ClassVar[int]
    DRIVABLE_SURFACE_PRIOR_FIELD_NUMBER: _ClassVar[int]
    MULTIPOLYGON_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARK_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    VENUE_FIELD_NUMBER: _ClassVar[int]
    administrative_boundary: _empty_pb2.Empty
    building: AnnotatedShape.Building
    drivable_surface_prior: _empty_pb2.Empty
    multipolygon: MultiPolygon
    name: _containers.RepeatedCompositeFieldContainer[LocalizedString]
    park: _empty_pb2.Empty
    region: AnnotatedShape.Region
    venue: _empty_pb2.Empty
    def __init__(self, name: _Optional[_Iterable[_Union[LocalizedString, _Mapping]]] = ..., multipolygon: _Optional[_Union[MultiPolygon, _Mapping]] = ..., building: _Optional[_Union[AnnotatedShape.Building, _Mapping]] = ..., region: _Optional[_Union[AnnotatedShape.Region, _Mapping]] = ..., venue: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., administrative_boundary: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., park: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., drivable_surface_prior: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ...) -> None: ...

class Condition(_message.Message):
    __slots__ = ["daily_temporal_condition"]
    DAILY_TEMPORAL_CONDITION_FIELD_NUMBER: _ClassVar[int]
    daily_temporal_condition: DailyTimeInterval
    def __init__(self, daily_temporal_condition: _Optional[_Union[DailyTimeInterval, _Mapping]] = ...) -> None: ...

class DailyTimeInterval(_message.Message):
    __slots__ = ["day_of_the_week", "end_local_time_seconds", "start_local_time_seconds", "timezone_database_region_name"]
    class DayOfTheWeek(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    DAY_OF_THE_WEEK_FIELD_NUMBER: _ClassVar[int]
    END_LOCAL_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    FRIDAY: DailyTimeInterval.DayOfTheWeek
    MONDAY: DailyTimeInterval.DayOfTheWeek
    SATURDAY: DailyTimeInterval.DayOfTheWeek
    START_LOCAL_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SUNDAY: DailyTimeInterval.DayOfTheWeek
    THURSDAY: DailyTimeInterval.DayOfTheWeek
    TIMEZONE_DATABASE_REGION_NAME_FIELD_NUMBER: _ClassVar[int]
    TUESDAY: DailyTimeInterval.DayOfTheWeek
    WEDNESDAY: DailyTimeInterval.DayOfTheWeek
    day_of_the_week: DailyTimeInterval.DayOfTheWeek
    end_local_time_seconds: int
    start_local_time_seconds: int
    timezone_database_region_name: str
    def __init__(self, day_of_the_week: _Optional[_Union[DailyTimeInterval.DayOfTheWeek, str]] = ..., start_local_time_seconds: _Optional[int] = ..., end_local_time_seconds: _Optional[int] = ..., timezone_database_region_name: _Optional[str] = ...) -> None: ...

class GeoFrame(_message.Message):
    __slots__ = ["bearing_degrees", "origin"]
    BEARING_DEGREES_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_FIELD_NUMBER: _ClassVar[int]
    bearing_degrees: float
    origin: GeoLocation
    def __init__(self, origin: _Optional[_Union[GeoLocation, _Mapping]] = ..., bearing_degrees: _Optional[float] = ...) -> None: ...

class GeoLocation(_message.Message):
    __slots__ = ["altitude_cm", "lat_e7", "lng_e7"]
    ALTITUDE_CM_FIELD_NUMBER: _ClassVar[int]
    LAT_E7_FIELD_NUMBER: _ClassVar[int]
    LNG_E7_FIELD_NUMBER: _ClassVar[int]
    altitude_cm: int
    lat_e7: int
    lng_e7: int
    def __init__(self, lat_e7: _Optional[int] = ..., lng_e7: _Optional[int] = ..., altitude_cm: _Optional[int] = ...) -> None: ...

class GlobalId(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: bytes
    def __init__(self, id: _Optional[bytes] = ...) -> None: ...

class Junction(_message.Message):
    __slots__ = ["is_non_trivial_intersection", "lanes", "road_network_nodes", "traffic_control_elements"]
    IS_NON_TRIVIAL_INTERSECTION_FIELD_NUMBER: _ClassVar[int]
    LANES_FIELD_NUMBER: _ClassVar[int]
    ROAD_NETWORK_NODES_FIELD_NUMBER: _ClassVar[int]
    TRAFFIC_CONTROL_ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    is_non_trivial_intersection: bool
    lanes: _containers.RepeatedCompositeFieldContainer[GlobalId]
    road_network_nodes: _containers.RepeatedCompositeFieldContainer[GlobalId]
    traffic_control_elements: _containers.RepeatedCompositeFieldContainer[GlobalId]
    def __init__(self, is_non_trivial_intersection: bool = ..., road_network_nodes: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., lanes: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., traffic_control_elements: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ...) -> None: ...

class Lane(_message.Message):
    __slots__ = ["access_restriction", "adjacent_lane_change_left", "adjacent_lane_change_right", "can_have_parked_cars", "geo_frame", "lanes_ahead", "left_boundary", "orientation_in_parent_segment", "parent_segment_or_junction", "right_boundary", "tolls", "traffic_controls", "turn_type_in_parent_junction", "yield_to_lanes"]
    class TurnType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class Boundary(_message.Message):
        __slots__ = ["divider_type", "type_change_point_cm", "vertex_deltas_x_cm", "vertex_deltas_y_cm", "vertex_deltas_z_cm"]
        class DividerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        CURB: Lane.Boundary.DividerType
        CURB_RED: Lane.Boundary.DividerType
        CURB_YELLOW: Lane.Boundary.DividerType
        DIVIDER_TYPE_FIELD_NUMBER: _ClassVar[int]
        DOUBLE_WHITE_SOLID: Lane.Boundary.DividerType
        DOUBLE_YELLOW_DASHED_FAR_SOLID_NEAR: Lane.Boundary.DividerType
        DOUBLE_YELLOW_SOLID: Lane.Boundary.DividerType
        DOUBLE_YELLOW_SOLID_FAR_DASHED_NEAR: Lane.Boundary.DividerType
        NONE: Lane.Boundary.DividerType
        SINGLE_WHITE_DASHED: Lane.Boundary.DividerType
        SINGLE_WHITE_SOLID: Lane.Boundary.DividerType
        SINGLE_YELLOW_DASHED: Lane.Boundary.DividerType
        SINGLE_YELLOW_SOLID: Lane.Boundary.DividerType
        TYPE_CHANGE_POINT_CM_FIELD_NUMBER: _ClassVar[int]
        UNKNOWN: Lane.Boundary.DividerType
        VERTEX_DELTAS_X_CM_FIELD_NUMBER: _ClassVar[int]
        VERTEX_DELTAS_Y_CM_FIELD_NUMBER: _ClassVar[int]
        VERTEX_DELTAS_Z_CM_FIELD_NUMBER: _ClassVar[int]
        divider_type: _containers.RepeatedScalarFieldContainer[Lane.Boundary.DividerType]
        type_change_point_cm: _containers.RepeatedScalarFieldContainer[int]
        vertex_deltas_x_cm: _containers.RepeatedScalarFieldContainer[int]
        vertex_deltas_y_cm: _containers.RepeatedScalarFieldContainer[int]
        vertex_deltas_z_cm: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, vertex_deltas_x_cm: _Optional[_Iterable[int]] = ..., vertex_deltas_y_cm: _Optional[_Iterable[int]] = ..., vertex_deltas_z_cm: _Optional[_Iterable[int]] = ..., divider_type: _Optional[_Iterable[_Union[Lane.Boundary.DividerType, str]]] = ..., type_change_point_cm: _Optional[_Iterable[int]] = ...) -> None: ...
    ACCESS_RESTRICTION_FIELD_NUMBER: _ClassVar[int]
    ADJACENT_LANE_CHANGE_LEFT_FIELD_NUMBER: _ClassVar[int]
    ADJACENT_LANE_CHANGE_RIGHT_FIELD_NUMBER: _ClassVar[int]
    CAN_HAVE_PARKED_CARS_FIELD_NUMBER: _ClassVar[int]
    GEO_FRAME_FIELD_NUMBER: _ClassVar[int]
    LANES_AHEAD_FIELD_NUMBER: _ClassVar[int]
    LEFT: Lane.TurnType
    LEFT_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_IN_PARENT_SEGMENT_FIELD_NUMBER: _ClassVar[int]
    PARENT_SEGMENT_OR_JUNCTION_FIELD_NUMBER: _ClassVar[int]
    RIGHT: Lane.TurnType
    RIGHT_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    SHARP_LEFT: Lane.TurnType
    SHARP_RIGHT: Lane.TurnType
    THROUGH: Lane.TurnType
    TOLLS_FIELD_NUMBER: _ClassVar[int]
    TRAFFIC_CONTROLS_FIELD_NUMBER: _ClassVar[int]
    TURN_TYPE_IN_PARENT_JUNCTION_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN: Lane.TurnType
    U_TURN: Lane.TurnType
    YIELD_TO_LANES_FIELD_NUMBER: _ClassVar[int]
    access_restriction: AccessRestriction
    adjacent_lane_change_left: GlobalId
    adjacent_lane_change_right: GlobalId
    can_have_parked_cars: bool
    geo_frame: GeoFrame
    lanes_ahead: _containers.RepeatedCompositeFieldContainer[GlobalId]
    left_boundary: Lane.Boundary
    orientation_in_parent_segment: RoadNetworkSegment.TravelDirection
    parent_segment_or_junction: GlobalId
    right_boundary: Lane.Boundary
    tolls: _containers.RepeatedCompositeFieldContainer[GlobalId]
    traffic_controls: _containers.RepeatedCompositeFieldContainer[GlobalId]
    turn_type_in_parent_junction: Lane.TurnType
    yield_to_lanes: _containers.RepeatedCompositeFieldContainer[GlobalId]
    def __init__(self, parent_segment_or_junction: _Optional[_Union[GlobalId, _Mapping]] = ..., access_restriction: _Optional[_Union[AccessRestriction, _Mapping]] = ..., orientation_in_parent_segment: _Optional[_Union[RoadNetworkSegment.TravelDirection, str]] = ..., turn_type_in_parent_junction: _Optional[_Union[Lane.TurnType, str]] = ..., geo_frame: _Optional[_Union[GeoFrame, _Mapping]] = ..., left_boundary: _Optional[_Union[Lane.Boundary, _Mapping]] = ..., right_boundary: _Optional[_Union[Lane.Boundary, _Mapping]] = ..., lanes_ahead: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., adjacent_lane_change_left: _Optional[_Union[GlobalId, _Mapping]] = ..., adjacent_lane_change_right: _Optional[_Union[GlobalId, _Mapping]] = ..., traffic_controls: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., yield_to_lanes: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., can_have_parked_cars: bool = ..., tolls: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ...) -> None: ...

class LaneSequence(_message.Message):
    __slots__ = ["lanes"]
    LANES_FIELD_NUMBER: _ClassVar[int]
    lanes: _containers.RepeatedCompositeFieldContainer[GlobalId]
    def __init__(self, lanes: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ...) -> None: ...

class LaneSequences(_message.Message):
    __slots__ = ["lane_sequences"]
    LANE_SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    lane_sequences: _containers.RepeatedCompositeFieldContainer[LaneSequence]
    def __init__(self, lane_sequences: _Optional[_Iterable[_Union[LaneSequence, _Mapping]]] = ...) -> None: ...

class LatLngBox(_message.Message):
    __slots__ = ["north_east", "south_west"]
    NORTH_EAST_FIELD_NUMBER: _ClassVar[int]
    SOUTH_WEST_FIELD_NUMBER: _ClassVar[int]
    north_east: GeoLocation
    south_west: GeoLocation
    def __init__(self, south_west: _Optional[_Union[GeoLocation, _Mapping]] = ..., north_east: _Optional[_Union[GeoLocation, _Mapping]] = ...) -> None: ...

class LocalizedString(_message.Message):
    __slots__ = ["language_code", "value"]
    LANGUAGE_CODE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    language_code: str
    value: str
    def __init__(self, language_code: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class MapElement(_message.Message):
    __slots__ = ["associated_conditions", "bounding_box", "element", "id"]
    class AssociatedConditions(_message.Message):
        __slots__ = ["conditions", "overrides"]
        CONDITIONS_FIELD_NUMBER: _ClassVar[int]
        OVERRIDES_FIELD_NUMBER: _ClassVar[int]
        conditions: _containers.RepeatedCompositeFieldContainer[Condition]
        overrides: MapElement.Element
        def __init__(self, conditions: _Optional[_Iterable[_Union[Condition, _Mapping]]] = ..., overrides: _Optional[_Union[MapElement.Element, _Mapping]] = ...) -> None: ...
    class Element(_message.Message):
        __slots__ = ["annotated_shape", "junction", "lane", "node", "segment", "segment_sequence", "traffic_control_element"]
        ANNOTATED_SHAPE_FIELD_NUMBER: _ClassVar[int]
        JUNCTION_FIELD_NUMBER: _ClassVar[int]
        LANE_FIELD_NUMBER: _ClassVar[int]
        NODE_FIELD_NUMBER: _ClassVar[int]
        SEGMENT_FIELD_NUMBER: _ClassVar[int]
        SEGMENT_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
        TRAFFIC_CONTROL_ELEMENT_FIELD_NUMBER: _ClassVar[int]
        annotated_shape: AnnotatedShape
        junction: Junction
        lane: Lane
        node: RoadNetworkNode
        segment: RoadNetworkSegment
        segment_sequence: SegmentSequence
        traffic_control_element: TrafficControlElement
        def __init__(self, segment: _Optional[_Union[RoadNetworkSegment, _Mapping]] = ..., node: _Optional[_Union[RoadNetworkNode, _Mapping]] = ..., lane: _Optional[_Union[Lane, _Mapping]] = ..., traffic_control_element: _Optional[_Union[TrafficControlElement, _Mapping]] = ..., junction: _Optional[_Union[Junction, _Mapping]] = ..., segment_sequence: _Optional[_Union[SegmentSequence, _Mapping]] = ..., annotated_shape: _Optional[_Union[AnnotatedShape, _Mapping]] = ...) -> None: ...
    ASSOCIATED_CONDITIONS_FIELD_NUMBER: _ClassVar[int]
    BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    associated_conditions: _containers.RepeatedCompositeFieldContainer[MapElement.AssociatedConditions]
    bounding_box: LatLngBox
    element: MapElement.Element
    id: GlobalId
    def __init__(self, id: _Optional[_Union[GlobalId, _Mapping]] = ..., element: _Optional[_Union[MapElement.Element, _Mapping]] = ..., bounding_box: _Optional[_Union[LatLngBox, _Mapping]] = ..., associated_conditions: _Optional[_Iterable[_Union[MapElement.AssociatedConditions, _Mapping]]] = ...) -> None: ...

class MapFragment(_message.Message):
    __slots__ = ["elements", "name"]
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedCompositeFieldContainer[MapElement]
    name: str
    def __init__(self, name: _Optional[str] = ..., elements: _Optional[_Iterable[_Union[MapElement, _Mapping]]] = ...) -> None: ...

class MultiPolygon(_message.Message):
    __slots__ = ["polygons"]
    POLYGONS_FIELD_NUMBER: _ClassVar[int]
    polygons: _containers.RepeatedCompositeFieldContainer[Polygon]
    def __init__(self, polygons: _Optional[_Iterable[_Union[Polygon, _Mapping]]] = ...) -> None: ...

class Polygon(_message.Message):
    __slots__ = ["holes", "shell_vertices"]
    HOLES_FIELD_NUMBER: _ClassVar[int]
    SHELL_VERTICES_FIELD_NUMBER: _ClassVar[int]
    holes: _containers.RepeatedCompositeFieldContainer[Polygon]
    shell_vertices: _containers.RepeatedCompositeFieldContainer[GeoLocation]
    def __init__(self, shell_vertices: _Optional[_Iterable[_Union[GeoLocation, _Mapping]]] = ..., holes: _Optional[_Iterable[_Union[Polygon, _Mapping]]] = ...) -> None: ...

class RoadNetworkNode(_message.Message):
    __slots__ = ["junction", "location", "road_segments", "z_level"]
    JUNCTION_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    ROAD_SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    Z_LEVEL_FIELD_NUMBER: _ClassVar[int]
    junction: GlobalId
    location: GeoLocation
    road_segments: _containers.RepeatedCompositeFieldContainer[GlobalId]
    z_level: int
    def __init__(self, location: _Optional[_Union[GeoLocation, _Mapping]] = ..., road_segments: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., z_level: _Optional[int] = ..., junction: _Optional[_Union[GlobalId, _Mapping]] = ...) -> None: ...

class RoadNetworkSegment(_message.Message):
    __slots__ = ["backward_bikeable", "backward_direction_speed_limit_meters_per_second", "backward_lane_set", "driveable", "end_node", "forward_bikeable", "forward_lane_set", "is_private", "is_toll_road", "lanes", "name", "num_bidirectional_lanes", "restrictions", "road_class", "speed_limit_meters_per_second", "start_node", "travel_direction", "vertices", "walkable", "z_level"]
    class BikeAccess(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class RoadClass(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class SideOfSegment(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class TravelDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class LaneSet(_message.Message):
        __slots__ = ["bike_lane_access", "num_driving_lanes", "num_left_turn_driving_lanes", "num_right_turn_driving_lanes", "turn_descriptions_for_driving_lanes"]
        class BikeLaneAccess(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        BIKE_LANE_ACCESS_FIELD_NUMBER: _ClassVar[int]
        DESIGNATED: RoadNetworkSegment.LaneSet.BikeLaneAccess
        DESIGNATED_BACKWARDS: RoadNetworkSegment.LaneSet.BikeLaneAccess
        DESIGNATED_SHARED: RoadNetworkSegment.LaneSet.BikeLaneAccess
        NO: RoadNetworkSegment.LaneSet.BikeLaneAccess
        NUM_DRIVING_LANES_FIELD_NUMBER: _ClassVar[int]
        NUM_LEFT_TURN_DRIVING_LANES_FIELD_NUMBER: _ClassVar[int]
        NUM_RIGHT_TURN_DRIVING_LANES_FIELD_NUMBER: _ClassVar[int]
        SHARED: RoadNetworkSegment.LaneSet.BikeLaneAccess
        TURN_DESCRIPTIONS_FOR_DRIVING_LANES_FIELD_NUMBER: _ClassVar[int]
        UNKNOWN_BIKE_ACCESS: RoadNetworkSegment.LaneSet.BikeLaneAccess
        bike_lane_access: _containers.RepeatedScalarFieldContainer[RoadNetworkSegment.LaneSet.BikeLaneAccess]
        num_driving_lanes: int
        num_left_turn_driving_lanes: int
        num_right_turn_driving_lanes: int
        turn_descriptions_for_driving_lanes: str
        def __init__(self, num_driving_lanes: _Optional[int] = ..., num_left_turn_driving_lanes: _Optional[int] = ..., num_right_turn_driving_lanes: _Optional[int] = ..., turn_descriptions_for_driving_lanes: _Optional[str] = ..., bike_lane_access: _Optional[_Iterable[_Union[RoadNetworkSegment.LaneSet.BikeLaneAccess, str]]] = ...) -> None: ...
    ALLOWED: RoadNetworkSegment.BikeAccess
    BACKWARD_BIKEABLE_FIELD_NUMBER: _ClassVar[int]
    BACKWARD_DIRECTION_SPEED_LIMIT_METERS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    BACKWARD_LANE_SET_FIELD_NUMBER: _ClassVar[int]
    CYCLEWAY: RoadNetworkSegment.RoadClass
    DEDICATED: RoadNetworkSegment.BikeAccess
    DRIVEABLE_FIELD_NUMBER: _ClassVar[int]
    EITHER_SIDE_OF_SEGMENT: RoadNetworkSegment.SideOfSegment
    END_NODE_FIELD_NUMBER: _ClassVar[int]
    FORWARD_BIKEABLE_FIELD_NUMBER: _ClassVar[int]
    FORWARD_LANE_SET_FIELD_NUMBER: _ClassVar[int]
    IS_PRIVATE_FIELD_NUMBER: _ClassVar[int]
    IS_TOLL_ROAD_FIELD_NUMBER: _ClassVar[int]
    LANES_FIELD_NUMBER: _ClassVar[int]
    MOTORWAY: RoadNetworkSegment.RoadClass
    MOTORWAY_LINK: RoadNetworkSegment.RoadClass
    NAME_FIELD_NUMBER: _ClassVar[int]
    NEITHER_SIDE_OF_SEGMENT: RoadNetworkSegment.SideOfSegment
    NOT_ALLOWED: RoadNetworkSegment.BikeAccess
    NUM_BIDIRECTIONAL_LANES_FIELD_NUMBER: _ClassVar[int]
    ONE_WAY_BACKWARD: RoadNetworkSegment.TravelDirection
    ONE_WAY_FORWARD: RoadNetworkSegment.TravelDirection
    ONE_WAY_REVERSIBLE: RoadNetworkSegment.TravelDirection
    PATH: RoadNetworkSegment.RoadClass
    PEDESTRIAN: RoadNetworkSegment.RoadClass
    PRIMARY: RoadNetworkSegment.RoadClass
    PRIMARY_LINK: RoadNetworkSegment.RoadClass
    PROTECTED: RoadNetworkSegment.BikeAccess
    RESIDENTIAL: RoadNetworkSegment.RoadClass
    RESTRICTIONS_FIELD_NUMBER: _ClassVar[int]
    ROAD_CLASS_FIELD_NUMBER: _ClassVar[int]
    SECONDARY: RoadNetworkSegment.RoadClass
    SECONDARY_LINK: RoadNetworkSegment.RoadClass
    SEGMENT_LEFT: RoadNetworkSegment.SideOfSegment
    SEGMENT_RIGHT: RoadNetworkSegment.SideOfSegment
    SERVICE: RoadNetworkSegment.RoadClass
    SERVICE_ALLEY: RoadNetworkSegment.RoadClass
    SERVICE_DRIVEWAY: RoadNetworkSegment.RoadClass
    SERVICE_DRIVE_THROUGH: RoadNetworkSegment.RoadClass
    SERVICE_EMERGENCY_ACCESS: RoadNetworkSegment.RoadClass
    SERVICE_LIVING_STREET: RoadNetworkSegment.RoadClass
    SERVICE_PARKING_AISLE: RoadNetworkSegment.RoadClass
    SHARED: RoadNetworkSegment.BikeAccess
    SPEED_LIMIT_METERS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    START_NODE_FIELD_NUMBER: _ClassVar[int]
    STEPS: RoadNetworkSegment.RoadClass
    TERTIARY: RoadNetworkSegment.RoadClass
    TERTIARY_LINK: RoadNetworkSegment.RoadClass
    TRAVEL_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    TRUNK: RoadNetworkSegment.RoadClass
    TRUNK_LINK: RoadNetworkSegment.RoadClass
    TWO_WAY: RoadNetworkSegment.TravelDirection
    UNCLASSIFIED: RoadNetworkSegment.RoadClass
    UNKNOWN: RoadNetworkSegment.BikeAccess
    UNKNOWN_ROAD_CLASS: RoadNetworkSegment.RoadClass
    UNKNOWN_SIDE_OF_SEGMENT: RoadNetworkSegment.SideOfSegment
    UNKNOWN_TRAVEL_DIRECTION: RoadNetworkSegment.TravelDirection
    VERTICES_FIELD_NUMBER: _ClassVar[int]
    WALKABLE_FIELD_NUMBER: _ClassVar[int]
    Z_LEVEL_FIELD_NUMBER: _ClassVar[int]
    backward_bikeable: RoadNetworkSegment.BikeAccess
    backward_direction_speed_limit_meters_per_second: float
    backward_lane_set: RoadNetworkSegment.LaneSet
    driveable: bool
    end_node: GlobalId
    forward_bikeable: RoadNetworkSegment.BikeAccess
    forward_lane_set: RoadNetworkSegment.LaneSet
    is_private: bool
    is_toll_road: bool
    lanes: _containers.RepeatedCompositeFieldContainer[GlobalId]
    name: _containers.RepeatedCompositeFieldContainer[LocalizedString]
    num_bidirectional_lanes: int
    restrictions: _containers.RepeatedCompositeFieldContainer[GlobalId]
    road_class: RoadNetworkSegment.RoadClass
    speed_limit_meters_per_second: float
    start_node: GlobalId
    travel_direction: RoadNetworkSegment.TravelDirection
    vertices: _containers.RepeatedCompositeFieldContainer[GeoLocation]
    walkable: RoadNetworkSegment.SideOfSegment
    z_level: int
    def __init__(self, vertices: _Optional[_Iterable[_Union[GeoLocation, _Mapping]]] = ..., start_node: _Optional[_Union[GlobalId, _Mapping]] = ..., end_node: _Optional[_Union[GlobalId, _Mapping]] = ..., forward_lane_set: _Optional[_Union[RoadNetworkSegment.LaneSet, _Mapping]] = ..., backward_lane_set: _Optional[_Union[RoadNetworkSegment.LaneSet, _Mapping]] = ..., num_bidirectional_lanes: _Optional[int] = ..., lanes: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., name: _Optional[_Iterable[_Union[LocalizedString, _Mapping]]] = ..., road_class: _Optional[_Union[RoadNetworkSegment.RoadClass, str]] = ..., is_private: bool = ..., is_toll_road: bool = ..., travel_direction: _Optional[_Union[RoadNetworkSegment.TravelDirection, str]] = ..., z_level: _Optional[int] = ..., speed_limit_meters_per_second: _Optional[float] = ..., backward_direction_speed_limit_meters_per_second: _Optional[float] = ..., restrictions: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., driveable: bool = ..., walkable: _Optional[_Union[RoadNetworkSegment.SideOfSegment, str]] = ..., forward_bikeable: _Optional[_Union[RoadNetworkSegment.BikeAccess, str]] = ..., backward_bikeable: _Optional[_Union[RoadNetworkSegment.BikeAccess, str]] = ...) -> None: ...

class Schedule(_message.Message):
    __slots__ = ["daily_schedule"]
    DAILY_SCHEDULE_FIELD_NUMBER: _ClassVar[int]
    daily_schedule: _containers.RepeatedCompositeFieldContainer[DailyTimeInterval]
    def __init__(self, daily_schedule: _Optional[_Iterable[_Union[DailyTimeInterval, _Mapping]]] = ...) -> None: ...

class SegmentSequence(_message.Message):
    __slots__ = ["segments", "type"]
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class SegmentWithOrientation(_message.Message):
        __slots__ = ["orientation", "segment"]
        class Orientation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        BACKWARD: SegmentSequence.SegmentWithOrientation.Orientation
        FORWARD: SegmentSequence.SegmentWithOrientation.Orientation
        ORIENTATION_FIELD_NUMBER: _ClassVar[int]
        SEGMENT_FIELD_NUMBER: _ClassVar[int]
        UNKNOWN: SegmentSequence.SegmentWithOrientation.Orientation
        orientation: SegmentSequence.SegmentWithOrientation.Orientation
        segment: GlobalId
        def __init__(self, segment: _Optional[_Union[GlobalId, _Mapping]] = ..., orientation: _Optional[_Union[SegmentSequence.SegmentWithOrientation.Orientation, str]] = ...) -> None: ...
    FORBIDDEN: SegmentSequence.Type
    MANDATORY: SegmentSequence.Type
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN: SegmentSequence.Type
    segments: _containers.RepeatedCompositeFieldContainer[SegmentSequence.SegmentWithOrientation]
    type: SegmentSequence.Type
    def __init__(self, type: _Optional[_Union[SegmentSequence.Type, str]] = ..., segments: _Optional[_Iterable[_Union[SegmentSequence.SegmentWithOrientation, _Mapping]]] = ...) -> None: ...

class TrafficControlElement(_message.Message):
    __slots__ = ["construction_zone", "construction_zone_sign", "controlled_paths", "do_not_enter_sign", "geo_frame", "geometry_type", "keep_clear_zone", "left_turn_only_sign", "left_turn_yield_on_green_sign", "no_left_and_u_turn_sign", "no_left_turn_on_red_sign", "no_left_turn_sign", "no_parking_sign", "no_right_turn_on_red_sign", "no_right_turn_sign", "no_straight_through_sign", "no_turn_on_red_sign", "no_u_turn_sign", "one_way_sign", "other_turn_restriction_sign", "parking_zone", "pedestrian_crossing_sign", "pedestrian_crosswalk", "points_x_deltas_cm", "points_y_deltas_cm", "points_z_deltas_cm", "railroad_crossing_other_sign", "railroad_crossing_regulatory_sign", "railroad_crossing_warning_sign", "right_turn_only_sign", "roundabout_circulation_sign", "roundabout_directional_sign", "roundabout_other_sign", "school_zone_sign", "signal_ahead_sign", "signal_flashing_red", "signal_flashing_yellow", "signal_green_face", "signal_green_u_turn", "signal_left_arrow_green_face", "signal_left_arrow_red_face", "signal_left_arrow_yellow_face", "signal_red_face", "signal_red_u_turn", "signal_right_arrow_green_face", "signal_right_arrow_red_face", "signal_right_arrow_yellow_face", "signal_upper_left_arrow_green_face", "signal_upper_left_arrow_red_face", "signal_upper_left_arrow_yellow_face", "signal_upper_right_arrow_green_face", "signal_upper_right_arrow_red_face", "signal_upper_right_arrow_yellow_face", "signal_yellow_face", "signal_yellow_u_turn", "speed_bump", "speed_bump_sign", "speed_hump", "state_law_stop_for_pedestrian_in_crosswalk_sign", "state_law_yield_for_pedestrian_sign", "stop_here_for_pedestrians_sign", "stop_line", "stop_sign", "straight_through_only_sign", "traffic_light", "turning_vehicles_yield_to_pedestrians_sign", "u_turn_only_sign", "yield_here_for_pedestrians_sign", "yield_sign"]
    class GeometryType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class AuxiliaryElement(_message.Message):
        __slots__ = ["primary_traffic_control_elements"]
        PRIMARY_TRAFFIC_CONTROL_ELEMENTS_FIELD_NUMBER: _ClassVar[int]
        primary_traffic_control_elements: _containers.RepeatedCompositeFieldContainer[GlobalId]
        def __init__(self, primary_traffic_control_elements: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ...) -> None: ...
    class PedestrianCrosswalk(_message.Message):
        __slots__ = ["traffic_lights", "yield_lines"]
        TRAFFIC_LIGHTS_FIELD_NUMBER: _ClassVar[int]
        YIELD_LINES_FIELD_NUMBER: _ClassVar[int]
        traffic_lights: _containers.RepeatedCompositeFieldContainer[GlobalId]
        yield_lines: _containers.RepeatedCompositeFieldContainer[GlobalId]
        def __init__(self, traffic_lights: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., yield_lines: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ...) -> None: ...
    class TrafficLight(_message.Message):
        __slots__ = ["face_states"]
        FACE_STATES_FIELD_NUMBER: _ClassVar[int]
        face_states: _containers.RepeatedCompositeFieldContainer[GlobalId]
        def __init__(self, face_states: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ...) -> None: ...
    class TrafficLightFaceState(_message.Message):
        __slots__ = ["no_right_turn_on_red", "yield_rules_when_on"]
        class YieldSet(_message.Message):
            __slots__ = ["lane", "yield_to_crosswalks", "yield_to_lanes"]
            LANE_FIELD_NUMBER: _ClassVar[int]
            YIELD_TO_CROSSWALKS_FIELD_NUMBER: _ClassVar[int]
            YIELD_TO_LANES_FIELD_NUMBER: _ClassVar[int]
            lane: GlobalId
            yield_to_crosswalks: _containers.RepeatedCompositeFieldContainer[GlobalId]
            yield_to_lanes: _containers.RepeatedCompositeFieldContainer[GlobalId]
            def __init__(self, lane: _Optional[_Union[GlobalId, _Mapping]] = ..., yield_to_lanes: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ..., yield_to_crosswalks: _Optional[_Iterable[_Union[GlobalId, _Mapping]]] = ...) -> None: ...
        NO_RIGHT_TURN_ON_RED_FIELD_NUMBER: _ClassVar[int]
        YIELD_RULES_WHEN_ON_FIELD_NUMBER: _ClassVar[int]
        no_right_turn_on_red: bool
        yield_rules_when_on: _containers.RepeatedCompositeFieldContainer[TrafficControlElement.TrafficLightFaceState.YieldSet]
        def __init__(self, yield_rules_when_on: _Optional[_Iterable[_Union[TrafficControlElement.TrafficLightFaceState.YieldSet, _Mapping]]] = ..., no_right_turn_on_red: bool = ...) -> None: ...
    CONSTRUCTION_ZONE_FIELD_NUMBER: _ClassVar[int]
    CONSTRUCTION_ZONE_SIGN_FIELD_NUMBER: _ClassVar[int]
    CONTROLLED_PATHS_FIELD_NUMBER: _ClassVar[int]
    DO_NOT_ENTER_SIGN_FIELD_NUMBER: _ClassVar[int]
    GEOMETRY_TYPE_FIELD_NUMBER: _ClassVar[int]
    GEO_FRAME_FIELD_NUMBER: _ClassVar[int]
    KEEP_CLEAR_ZONE_FIELD_NUMBER: _ClassVar[int]
    LEFT_TURN_ONLY_SIGN_FIELD_NUMBER: _ClassVar[int]
    LEFT_TURN_YIELD_ON_GREEN_SIGN_FIELD_NUMBER: _ClassVar[int]
    LINESTRING: TrafficControlElement.GeometryType
    MULTI_POINT: TrafficControlElement.GeometryType
    NO_LEFT_AND_U_TURN_SIGN_FIELD_NUMBER: _ClassVar[int]
    NO_LEFT_TURN_ON_RED_SIGN_FIELD_NUMBER: _ClassVar[int]
    NO_LEFT_TURN_SIGN_FIELD_NUMBER: _ClassVar[int]
    NO_PARKING_SIGN_FIELD_NUMBER: _ClassVar[int]
    NO_RIGHT_TURN_ON_RED_SIGN_FIELD_NUMBER: _ClassVar[int]
    NO_RIGHT_TURN_SIGN_FIELD_NUMBER: _ClassVar[int]
    NO_STRAIGHT_THROUGH_SIGN_FIELD_NUMBER: _ClassVar[int]
    NO_TURN_ON_RED_SIGN_FIELD_NUMBER: _ClassVar[int]
    NO_U_TURN_SIGN_FIELD_NUMBER: _ClassVar[int]
    ONE_WAY_SIGN_FIELD_NUMBER: _ClassVar[int]
    OTHER_TURN_RESTRICTION_SIGN_FIELD_NUMBER: _ClassVar[int]
    PARKING_ZONE_FIELD_NUMBER: _ClassVar[int]
    PEDESTRIAN_CROSSING_SIGN_FIELD_NUMBER: _ClassVar[int]
    PEDESTRIAN_CROSSWALK_FIELD_NUMBER: _ClassVar[int]
    POINT: TrafficControlElement.GeometryType
    POINTS_X_DELTAS_CM_FIELD_NUMBER: _ClassVar[int]
    POINTS_Y_DELTAS_CM_FIELD_NUMBER: _ClassVar[int]
    POINTS_Z_DELTAS_CM_FIELD_NUMBER: _ClassVar[int]
    POLYGON: TrafficControlElement.GeometryType
    RAILROAD_CROSSING_OTHER_SIGN_FIELD_NUMBER: _ClassVar[int]
    RAILROAD_CROSSING_REGULATORY_SIGN_FIELD_NUMBER: _ClassVar[int]
    RAILROAD_CROSSING_WARNING_SIGN_FIELD_NUMBER: _ClassVar[int]
    RIGHT_TURN_ONLY_SIGN_FIELD_NUMBER: _ClassVar[int]
    ROUNDABOUT_CIRCULATION_SIGN_FIELD_NUMBER: _ClassVar[int]
    ROUNDABOUT_DIRECTIONAL_SIGN_FIELD_NUMBER: _ClassVar[int]
    ROUNDABOUT_OTHER_SIGN_FIELD_NUMBER: _ClassVar[int]
    SCHOOL_ZONE_SIGN_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_AHEAD_SIGN_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_FLASHING_RED_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_FLASHING_YELLOW_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_GREEN_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_GREEN_U_TURN_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_LEFT_ARROW_GREEN_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_LEFT_ARROW_RED_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_LEFT_ARROW_YELLOW_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_RED_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_RED_U_TURN_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_RIGHT_ARROW_GREEN_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_RIGHT_ARROW_RED_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_RIGHT_ARROW_YELLOW_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_UPPER_LEFT_ARROW_GREEN_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_UPPER_LEFT_ARROW_RED_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_UPPER_LEFT_ARROW_YELLOW_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_UPPER_RIGHT_ARROW_GREEN_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_UPPER_RIGHT_ARROW_RED_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_UPPER_RIGHT_ARROW_YELLOW_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_YELLOW_FACE_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_YELLOW_U_TURN_FIELD_NUMBER: _ClassVar[int]
    SPEED_BUMP_FIELD_NUMBER: _ClassVar[int]
    SPEED_BUMP_SIGN_FIELD_NUMBER: _ClassVar[int]
    SPEED_HUMP_FIELD_NUMBER: _ClassVar[int]
    STATE_LAW_STOP_FOR_PEDESTRIAN_IN_CROSSWALK_SIGN_FIELD_NUMBER: _ClassVar[int]
    STATE_LAW_YIELD_FOR_PEDESTRIAN_SIGN_FIELD_NUMBER: _ClassVar[int]
    STOP_HERE_FOR_PEDESTRIANS_SIGN_FIELD_NUMBER: _ClassVar[int]
    STOP_LINE_FIELD_NUMBER: _ClassVar[int]
    STOP_SIGN_FIELD_NUMBER: _ClassVar[int]
    STRAIGHT_THROUGH_ONLY_SIGN_FIELD_NUMBER: _ClassVar[int]
    TRAFFIC_LIGHT_FIELD_NUMBER: _ClassVar[int]
    TURNING_VEHICLES_YIELD_TO_PEDESTRIANS_SIGN_FIELD_NUMBER: _ClassVar[int]
    UKNOWN: TrafficControlElement.GeometryType
    U_TURN_ONLY_SIGN_FIELD_NUMBER: _ClassVar[int]
    YIELD_HERE_FOR_PEDESTRIANS_SIGN_FIELD_NUMBER: _ClassVar[int]
    YIELD_SIGN_FIELD_NUMBER: _ClassVar[int]
    construction_zone: _empty_pb2.Empty
    construction_zone_sign: _empty_pb2.Empty
    controlled_paths: _containers.RepeatedCompositeFieldContainer[LaneSequence]
    do_not_enter_sign: _empty_pb2.Empty
    geo_frame: GeoFrame
    geometry_type: TrafficControlElement.GeometryType
    keep_clear_zone: _empty_pb2.Empty
    left_turn_only_sign: _empty_pb2.Empty
    left_turn_yield_on_green_sign: _empty_pb2.Empty
    no_left_and_u_turn_sign: _empty_pb2.Empty
    no_left_turn_on_red_sign: _empty_pb2.Empty
    no_left_turn_sign: _empty_pb2.Empty
    no_parking_sign: _empty_pb2.Empty
    no_right_turn_on_red_sign: _empty_pb2.Empty
    no_right_turn_sign: _empty_pb2.Empty
    no_straight_through_sign: _empty_pb2.Empty
    no_turn_on_red_sign: _empty_pb2.Empty
    no_u_turn_sign: _empty_pb2.Empty
    one_way_sign: _empty_pb2.Empty
    other_turn_restriction_sign: _empty_pb2.Empty
    parking_zone: _empty_pb2.Empty
    pedestrian_crossing_sign: _empty_pb2.Empty
    pedestrian_crosswalk: TrafficControlElement.PedestrianCrosswalk
    points_x_deltas_cm: _containers.RepeatedScalarFieldContainer[int]
    points_y_deltas_cm: _containers.RepeatedScalarFieldContainer[int]
    points_z_deltas_cm: _containers.RepeatedScalarFieldContainer[int]
    railroad_crossing_other_sign: _empty_pb2.Empty
    railroad_crossing_regulatory_sign: _empty_pb2.Empty
    railroad_crossing_warning_sign: _empty_pb2.Empty
    right_turn_only_sign: _empty_pb2.Empty
    roundabout_circulation_sign: _empty_pb2.Empty
    roundabout_directional_sign: _empty_pb2.Empty
    roundabout_other_sign: _empty_pb2.Empty
    school_zone_sign: _empty_pb2.Empty
    signal_ahead_sign: _empty_pb2.Empty
    signal_flashing_red: TrafficControlElement.TrafficLightFaceState
    signal_flashing_yellow: TrafficControlElement.TrafficLightFaceState
    signal_green_face: TrafficControlElement.TrafficLightFaceState
    signal_green_u_turn: TrafficControlElement.TrafficLightFaceState
    signal_left_arrow_green_face: TrafficControlElement.TrafficLightFaceState
    signal_left_arrow_red_face: TrafficControlElement.TrafficLightFaceState
    signal_left_arrow_yellow_face: TrafficControlElement.TrafficLightFaceState
    signal_red_face: TrafficControlElement.TrafficLightFaceState
    signal_red_u_turn: TrafficControlElement.TrafficLightFaceState
    signal_right_arrow_green_face: TrafficControlElement.TrafficLightFaceState
    signal_right_arrow_red_face: TrafficControlElement.TrafficLightFaceState
    signal_right_arrow_yellow_face: TrafficControlElement.TrafficLightFaceState
    signal_upper_left_arrow_green_face: TrafficControlElement.TrafficLightFaceState
    signal_upper_left_arrow_red_face: TrafficControlElement.TrafficLightFaceState
    signal_upper_left_arrow_yellow_face: TrafficControlElement.TrafficLightFaceState
    signal_upper_right_arrow_green_face: TrafficControlElement.TrafficLightFaceState
    signal_upper_right_arrow_red_face: TrafficControlElement.TrafficLightFaceState
    signal_upper_right_arrow_yellow_face: TrafficControlElement.TrafficLightFaceState
    signal_yellow_face: TrafficControlElement.TrafficLightFaceState
    signal_yellow_u_turn: TrafficControlElement.TrafficLightFaceState
    speed_bump: _empty_pb2.Empty
    speed_bump_sign: _empty_pb2.Empty
    speed_hump: _empty_pb2.Empty
    state_law_stop_for_pedestrian_in_crosswalk_sign: _empty_pb2.Empty
    state_law_yield_for_pedestrian_sign: _empty_pb2.Empty
    stop_here_for_pedestrians_sign: _empty_pb2.Empty
    stop_line: TrafficControlElement.AuxiliaryElement
    stop_sign: _empty_pb2.Empty
    straight_through_only_sign: _empty_pb2.Empty
    traffic_light: TrafficControlElement.TrafficLight
    turning_vehicles_yield_to_pedestrians_sign: _empty_pb2.Empty
    u_turn_only_sign: _empty_pb2.Empty
    yield_here_for_pedestrians_sign: _empty_pb2.Empty
    yield_sign: _empty_pb2.Empty
    def __init__(self, stop_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., yield_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., stop_line: _Optional[_Union[TrafficControlElement.AuxiliaryElement, _Mapping]] = ..., traffic_light: _Optional[_Union[TrafficControlElement.TrafficLight, _Mapping]] = ..., signal_flashing_yellow: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_flashing_red: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_red_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_yellow_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_green_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_left_arrow_red_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_left_arrow_yellow_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_left_arrow_green_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_right_arrow_red_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_right_arrow_yellow_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_right_arrow_green_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_upper_left_arrow_red_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_upper_left_arrow_yellow_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_upper_left_arrow_green_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_upper_right_arrow_red_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_upper_right_arrow_yellow_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_upper_right_arrow_green_face: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_red_u_turn: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_yellow_u_turn: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., signal_green_u_turn: _Optional[_Union[TrafficControlElement.TrafficLightFaceState, _Mapping]] = ..., speed_bump: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., speed_hump: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., pedestrian_crosswalk: _Optional[_Union[TrafficControlElement.PedestrianCrosswalk, _Mapping]] = ..., keep_clear_zone: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., pedestrian_crossing_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., signal_ahead_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_left_turn_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_right_turn_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., left_turn_yield_on_green_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_parking_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., one_way_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., school_zone_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., parking_zone: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., speed_bump_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., construction_zone: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., stop_here_for_pedestrians_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., state_law_stop_for_pedestrian_in_crosswalk_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., yield_here_for_pedestrians_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., turning_vehicles_yield_to_pedestrians_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., state_law_yield_for_pedestrian_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., railroad_crossing_regulatory_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., railroad_crossing_warning_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., railroad_crossing_other_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., roundabout_circulation_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., roundabout_directional_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., roundabout_other_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_left_turn_on_red_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_right_turn_on_red_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_turn_on_red_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., do_not_enter_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_u_turn_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_left_and_u_turn_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., no_straight_through_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., left_turn_only_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., right_turn_only_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., u_turn_only_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., straight_through_only_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., other_turn_restriction_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., construction_zone_sign: _Optional[_Union[_empty_pb2.Empty, _Mapping]] = ..., geo_frame: _Optional[_Union[GeoFrame, _Mapping]] = ..., points_x_deltas_cm: _Optional[_Iterable[int]] = ..., points_y_deltas_cm: _Optional[_Iterable[int]] = ..., points_z_deltas_cm: _Optional[_Iterable[int]] = ..., geometry_type: _Optional[_Union[TrafficControlElement.GeometryType, str]] = ..., controlled_paths: _Optional[_Iterable[_Union[LaneSequence, _Mapping]]] = ...) -> None: ...

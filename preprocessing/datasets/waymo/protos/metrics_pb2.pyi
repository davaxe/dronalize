import dataset_pb2 as _dataset_pb2
import label_pb2 as _label_pb2
from protos import breakdown_pb2 as _breakdown_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Config(_message.Message):
    __slots__ = ["box_type", "breakdown_generator_ids", "desired_recall_delta", "difficulties", "include_details_in_measurements", "iou_thresholds", "let_metric_config", "matcher_type", "min_heading_accuracy", "min_precision", "num_desired_score_cutoffs", "score_cutoffs"]
    class LongitudinalErrorTolerantConfig(_message.Message):
        __slots__ = ["align_type", "enabled", "longitudinal_tolerance_percentage", "min_longitudinal_tolerance_meter", "sensor_location"]
        class AlignType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        class Location3D(_message.Message):
            __slots__ = ["x", "y", "z"]
            X_FIELD_NUMBER: _ClassVar[int]
            Y_FIELD_NUMBER: _ClassVar[int]
            Z_FIELD_NUMBER: _ClassVar[int]
            x: float
            y: float
            z: float
            def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...
        ALIGN_TYPE_FIELD_NUMBER: _ClassVar[int]
        ENABLED_FIELD_NUMBER: _ClassVar[int]
        LONGITUDINAL_TOLERANCE_PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
        MIN_LONGITUDINAL_TOLERANCE_METER_FIELD_NUMBER: _ClassVar[int]
        SENSOR_LOCATION_FIELD_NUMBER: _ClassVar[int]
        TYPE_ANY_CLOSER_ONLY_RANGE_ALIGNED: Config.LongitudinalErrorTolerantConfig.AlignType
        TYPE_BETWEEN_ORIGIN_AND_GT_ONLY_RANGE_ALIGNED: Config.LongitudinalErrorTolerantConfig.AlignType
        TYPE_CENTER_ALIGNED: Config.LongitudinalErrorTolerantConfig.AlignType
        TYPE_FURTHER_ONLY_RANGE_ALIGNED: Config.LongitudinalErrorTolerantConfig.AlignType
        TYPE_NOT_ALIGNED: Config.LongitudinalErrorTolerantConfig.AlignType
        TYPE_RANGE_ALIGNED: Config.LongitudinalErrorTolerantConfig.AlignType
        TYPE_UNKNOWN: Config.LongitudinalErrorTolerantConfig.AlignType
        align_type: Config.LongitudinalErrorTolerantConfig.AlignType
        enabled: bool
        longitudinal_tolerance_percentage: float
        min_longitudinal_tolerance_meter: float
        sensor_location: Config.LongitudinalErrorTolerantConfig.Location3D
        def __init__(self, enabled: bool = ..., sensor_location: _Optional[_Union[Config.LongitudinalErrorTolerantConfig.Location3D, _Mapping]] = ..., longitudinal_tolerance_percentage: _Optional[float] = ..., min_longitudinal_tolerance_meter: _Optional[float] = ..., align_type: _Optional[_Union[Config.LongitudinalErrorTolerantConfig.AlignType, str]] = ...) -> None: ...
    BOX_TYPE_FIELD_NUMBER: _ClassVar[int]
    BREAKDOWN_GENERATOR_IDS_FIELD_NUMBER: _ClassVar[int]
    DESIRED_RECALL_DELTA_FIELD_NUMBER: _ClassVar[int]
    DIFFICULTIES_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_DETAILS_IN_MEASUREMENTS_FIELD_NUMBER: _ClassVar[int]
    IOU_THRESHOLDS_FIELD_NUMBER: _ClassVar[int]
    LET_METRIC_CONFIG_FIELD_NUMBER: _ClassVar[int]
    MATCHER_TYPE_FIELD_NUMBER: _ClassVar[int]
    MIN_HEADING_ACCURACY_FIELD_NUMBER: _ClassVar[int]
    MIN_PRECISION_FIELD_NUMBER: _ClassVar[int]
    NUM_DESIRED_SCORE_CUTOFFS_FIELD_NUMBER: _ClassVar[int]
    SCORE_CUTOFFS_FIELD_NUMBER: _ClassVar[int]
    box_type: _label_pb2.Label.Box.Type
    breakdown_generator_ids: _containers.RepeatedScalarFieldContainer[_breakdown_pb2.Breakdown.GeneratorId]
    desired_recall_delta: float
    difficulties: _containers.RepeatedCompositeFieldContainer[Difficulty]
    include_details_in_measurements: bool
    iou_thresholds: _containers.RepeatedScalarFieldContainer[float]
    let_metric_config: Config.LongitudinalErrorTolerantConfig
    matcher_type: MatcherProto.Type
    min_heading_accuracy: float
    min_precision: float
    num_desired_score_cutoffs: int
    score_cutoffs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, score_cutoffs: _Optional[_Iterable[float]] = ..., num_desired_score_cutoffs: _Optional[int] = ..., breakdown_generator_ids: _Optional[_Iterable[_Union[_breakdown_pb2.Breakdown.GeneratorId, str]]] = ..., difficulties: _Optional[_Iterable[_Union[Difficulty, _Mapping]]] = ..., matcher_type: _Optional[_Union[MatcherProto.Type, str]] = ..., iou_thresholds: _Optional[_Iterable[float]] = ..., box_type: _Optional[_Union[_label_pb2.Label.Box.Type, str]] = ..., desired_recall_delta: _Optional[float] = ..., let_metric_config: _Optional[_Union[Config.LongitudinalErrorTolerantConfig, _Mapping]] = ..., min_precision: _Optional[float] = ..., min_heading_accuracy: _Optional[float] = ..., include_details_in_measurements: bool = ...) -> None: ...

class DetectionMeasurement(_message.Message):
    __slots__ = ["details", "num_fns", "num_fps", "num_tps", "score_cutoff", "sum_ha", "sum_longitudinal_affinity"]
    class Details(_message.Message):
        __slots__ = ["fn_ids", "fp_ids", "tp_gt_ids", "tp_heading_accuracies", "tp_ious", "tp_longitudinal_affinities", "tp_pr_ids"]
        FN_IDS_FIELD_NUMBER: _ClassVar[int]
        FP_IDS_FIELD_NUMBER: _ClassVar[int]
        TP_GT_IDS_FIELD_NUMBER: _ClassVar[int]
        TP_HEADING_ACCURACIES_FIELD_NUMBER: _ClassVar[int]
        TP_IOUS_FIELD_NUMBER: _ClassVar[int]
        TP_LONGITUDINAL_AFFINITIES_FIELD_NUMBER: _ClassVar[int]
        TP_PR_IDS_FIELD_NUMBER: _ClassVar[int]
        fn_ids: _containers.RepeatedScalarFieldContainer[str]
        fp_ids: _containers.RepeatedScalarFieldContainer[str]
        tp_gt_ids: _containers.RepeatedScalarFieldContainer[str]
        tp_heading_accuracies: _containers.RepeatedScalarFieldContainer[float]
        tp_ious: _containers.RepeatedScalarFieldContainer[float]
        tp_longitudinal_affinities: _containers.RepeatedScalarFieldContainer[float]
        tp_pr_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, fp_ids: _Optional[_Iterable[str]] = ..., fn_ids: _Optional[_Iterable[str]] = ..., tp_gt_ids: _Optional[_Iterable[str]] = ..., tp_pr_ids: _Optional[_Iterable[str]] = ..., tp_ious: _Optional[_Iterable[float]] = ..., tp_heading_accuracies: _Optional[_Iterable[float]] = ..., tp_longitudinal_affinities: _Optional[_Iterable[float]] = ...) -> None: ...
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    NUM_FNS_FIELD_NUMBER: _ClassVar[int]
    NUM_FPS_FIELD_NUMBER: _ClassVar[int]
    NUM_TPS_FIELD_NUMBER: _ClassVar[int]
    SCORE_CUTOFF_FIELD_NUMBER: _ClassVar[int]
    SUM_HA_FIELD_NUMBER: _ClassVar[int]
    SUM_LONGITUDINAL_AFFINITY_FIELD_NUMBER: _ClassVar[int]
    details: _containers.RepeatedCompositeFieldContainer[DetectionMeasurement.Details]
    num_fns: int
    num_fps: int
    num_tps: int
    score_cutoff: float
    sum_ha: float
    sum_longitudinal_affinity: float
    def __init__(self, num_fps: _Optional[int] = ..., num_tps: _Optional[int] = ..., num_fns: _Optional[int] = ..., details: _Optional[_Iterable[_Union[DetectionMeasurement.Details, _Mapping]]] = ..., sum_ha: _Optional[float] = ..., sum_longitudinal_affinity: _Optional[float] = ..., score_cutoff: _Optional[float] = ...) -> None: ...

class DetectionMeasurements(_message.Message):
    __slots__ = ["breakdown", "measurements"]
    BREAKDOWN_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENTS_FIELD_NUMBER: _ClassVar[int]
    breakdown: _breakdown_pb2.Breakdown
    measurements: _containers.RepeatedCompositeFieldContainer[DetectionMeasurement]
    def __init__(self, measurements: _Optional[_Iterable[_Union[DetectionMeasurement, _Mapping]]] = ..., breakdown: _Optional[_Union[_breakdown_pb2.Breakdown, _Mapping]] = ...) -> None: ...

class DetectionMetrics(_message.Message):
    __slots__ = ["breakdown", "mean_average_precision", "mean_average_precision_ha_weighted", "mean_average_precision_longitudinal_affinity_weighted", "measurements", "precisions", "precisions_ha_weighted", "precisions_longitudinal_affinity_weighted", "recalls", "recalls_ha_weighted", "recalls_longitudinal_affinity_weighted", "score_cutoffs"]
    BREAKDOWN_FIELD_NUMBER: _ClassVar[int]
    MEAN_AVERAGE_PRECISION_FIELD_NUMBER: _ClassVar[int]
    MEAN_AVERAGE_PRECISION_HA_WEIGHTED_FIELD_NUMBER: _ClassVar[int]
    MEAN_AVERAGE_PRECISION_LONGITUDINAL_AFFINITY_WEIGHTED_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENTS_FIELD_NUMBER: _ClassVar[int]
    PRECISIONS_FIELD_NUMBER: _ClassVar[int]
    PRECISIONS_HA_WEIGHTED_FIELD_NUMBER: _ClassVar[int]
    PRECISIONS_LONGITUDINAL_AFFINITY_WEIGHTED_FIELD_NUMBER: _ClassVar[int]
    RECALLS_FIELD_NUMBER: _ClassVar[int]
    RECALLS_HA_WEIGHTED_FIELD_NUMBER: _ClassVar[int]
    RECALLS_LONGITUDINAL_AFFINITY_WEIGHTED_FIELD_NUMBER: _ClassVar[int]
    SCORE_CUTOFFS_FIELD_NUMBER: _ClassVar[int]
    breakdown: _breakdown_pb2.Breakdown
    mean_average_precision: float
    mean_average_precision_ha_weighted: float
    mean_average_precision_longitudinal_affinity_weighted: float
    measurements: DetectionMeasurements
    precisions: _containers.RepeatedScalarFieldContainer[float]
    precisions_ha_weighted: _containers.RepeatedScalarFieldContainer[float]
    precisions_longitudinal_affinity_weighted: _containers.RepeatedScalarFieldContainer[float]
    recalls: _containers.RepeatedScalarFieldContainer[float]
    recalls_ha_weighted: _containers.RepeatedScalarFieldContainer[float]
    recalls_longitudinal_affinity_weighted: _containers.RepeatedScalarFieldContainer[float]
    score_cutoffs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, mean_average_precision: _Optional[float] = ..., mean_average_precision_ha_weighted: _Optional[float] = ..., mean_average_precision_longitudinal_affinity_weighted: _Optional[float] = ..., precisions: _Optional[_Iterable[float]] = ..., recalls: _Optional[_Iterable[float]] = ..., precisions_ha_weighted: _Optional[_Iterable[float]] = ..., recalls_ha_weighted: _Optional[_Iterable[float]] = ..., precisions_longitudinal_affinity_weighted: _Optional[_Iterable[float]] = ..., recalls_longitudinal_affinity_weighted: _Optional[_Iterable[float]] = ..., score_cutoffs: _Optional[_Iterable[float]] = ..., breakdown: _Optional[_Union[_breakdown_pb2.Breakdown, _Mapping]] = ..., measurements: _Optional[_Union[DetectionMeasurements, _Mapping]] = ...) -> None: ...

class Difficulty(_message.Message):
    __slots__ = ["levels"]
    LEVELS_FIELD_NUMBER: _ClassVar[int]
    levels: _containers.RepeatedScalarFieldContainer[_label_pb2.Label.DifficultyLevel]
    def __init__(self, levels: _Optional[_Iterable[_Union[_label_pb2.Label.DifficultyLevel, str]]] = ...) -> None: ...

class MatcherProto(_message.Message):
    __slots__ = []
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    TYPE_HUNGARIAN: MatcherProto.Type
    TYPE_HUNGARIAN_TEST_ONLY: MatcherProto.Type
    TYPE_SCORE_FIRST: MatcherProto.Type
    TYPE_UNKNOWN: MatcherProto.Type
    def __init__(self) -> None: ...

class NoLabelZoneObject(_message.Message):
    __slots__ = ["context_name", "frame_timestamp_micros", "zone"]
    CONTEXT_NAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_TIMESTAMP_MICROS_FIELD_NUMBER: _ClassVar[int]
    ZONE_FIELD_NUMBER: _ClassVar[int]
    context_name: str
    frame_timestamp_micros: int
    zone: _label_pb2.Polygon2dProto
    def __init__(self, zone: _Optional[_Union[_label_pb2.Polygon2dProto, _Mapping]] = ..., context_name: _Optional[str] = ..., frame_timestamp_micros: _Optional[int] = ...) -> None: ...

class Object(_message.Message):
    __slots__ = ["camera_name", "context_name", "frame_timestamp_micros", "object", "overlap_with_nlz", "score"]
    CAMERA_NAME_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_NAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_TIMESTAMP_MICROS_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    OVERLAP_WITH_NLZ_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    camera_name: _dataset_pb2.CameraName.Name
    context_name: str
    frame_timestamp_micros: int
    object: _label_pb2.Label
    overlap_with_nlz: bool
    score: float
    def __init__(self, object: _Optional[_Union[_label_pb2.Label, _Mapping]] = ..., score: _Optional[float] = ..., overlap_with_nlz: bool = ..., context_name: _Optional[str] = ..., frame_timestamp_micros: _Optional[int] = ..., camera_name: _Optional[_Union[_dataset_pb2.CameraName.Name, str]] = ...) -> None: ...

class Objects(_message.Message):
    __slots__ = ["no_label_zone_objects", "objects"]
    NO_LABEL_ZONE_OBJECTS_FIELD_NUMBER: _ClassVar[int]
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    no_label_zone_objects: _containers.RepeatedCompositeFieldContainer[NoLabelZoneObject]
    objects: _containers.RepeatedCompositeFieldContainer[Object]
    def __init__(self, objects: _Optional[_Iterable[_Union[Object, _Mapping]]] = ..., no_label_zone_objects: _Optional[_Iterable[_Union[NoLabelZoneObject, _Mapping]]] = ...) -> None: ...

class TrackingMeasurement(_message.Message):
    __slots__ = ["details", "matching_cost", "num_fps", "num_matches", "num_mismatches", "num_misses", "num_objects_gt", "score_cutoff"]
    class Details(_message.Message):
        __slots__ = ["fn_gt_ids", "fp_pred_ids", "tp_gt_ids", "tp_pred_ids"]
        FN_GT_IDS_FIELD_NUMBER: _ClassVar[int]
        FP_PRED_IDS_FIELD_NUMBER: _ClassVar[int]
        TP_GT_IDS_FIELD_NUMBER: _ClassVar[int]
        TP_PRED_IDS_FIELD_NUMBER: _ClassVar[int]
        fn_gt_ids: _containers.RepeatedScalarFieldContainer[str]
        fp_pred_ids: _containers.RepeatedScalarFieldContainer[str]
        tp_gt_ids: _containers.RepeatedScalarFieldContainer[str]
        tp_pred_ids: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, fp_pred_ids: _Optional[_Iterable[str]] = ..., fn_gt_ids: _Optional[_Iterable[str]] = ..., tp_gt_ids: _Optional[_Iterable[str]] = ..., tp_pred_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    MATCHING_COST_FIELD_NUMBER: _ClassVar[int]
    NUM_FPS_FIELD_NUMBER: _ClassVar[int]
    NUM_MATCHES_FIELD_NUMBER: _ClassVar[int]
    NUM_MISMATCHES_FIELD_NUMBER: _ClassVar[int]
    NUM_MISSES_FIELD_NUMBER: _ClassVar[int]
    NUM_OBJECTS_GT_FIELD_NUMBER: _ClassVar[int]
    SCORE_CUTOFF_FIELD_NUMBER: _ClassVar[int]
    details: _containers.RepeatedCompositeFieldContainer[TrackingMeasurement.Details]
    matching_cost: float
    num_fps: int
    num_matches: int
    num_mismatches: int
    num_misses: int
    num_objects_gt: int
    score_cutoff: float
    def __init__(self, num_misses: _Optional[int] = ..., num_fps: _Optional[int] = ..., num_mismatches: _Optional[int] = ..., matching_cost: _Optional[float] = ..., num_matches: _Optional[int] = ..., num_objects_gt: _Optional[int] = ..., score_cutoff: _Optional[float] = ..., details: _Optional[_Iterable[_Union[TrackingMeasurement.Details, _Mapping]]] = ...) -> None: ...

class TrackingMeasurements(_message.Message):
    __slots__ = ["breakdown", "measurements"]
    BREAKDOWN_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENTS_FIELD_NUMBER: _ClassVar[int]
    breakdown: _breakdown_pb2.Breakdown
    measurements: _containers.RepeatedCompositeFieldContainer[TrackingMeasurement]
    def __init__(self, measurements: _Optional[_Iterable[_Union[TrackingMeasurement, _Mapping]]] = ..., breakdown: _Optional[_Union[_breakdown_pb2.Breakdown, _Mapping]] = ...) -> None: ...

class TrackingMetrics(_message.Message):
    __slots__ = ["breakdown", "fp", "measurements", "mismatch", "miss", "mota", "motp", "num_objects_gt", "score_cutoff"]
    BREAKDOWN_FIELD_NUMBER: _ClassVar[int]
    FP_FIELD_NUMBER: _ClassVar[int]
    MEASUREMENTS_FIELD_NUMBER: _ClassVar[int]
    MISMATCH_FIELD_NUMBER: _ClassVar[int]
    MISS_FIELD_NUMBER: _ClassVar[int]
    MOTA_FIELD_NUMBER: _ClassVar[int]
    MOTP_FIELD_NUMBER: _ClassVar[int]
    NUM_OBJECTS_GT_FIELD_NUMBER: _ClassVar[int]
    SCORE_CUTOFF_FIELD_NUMBER: _ClassVar[int]
    breakdown: _breakdown_pb2.Breakdown
    fp: float
    measurements: TrackingMeasurements
    mismatch: float
    miss: float
    mota: float
    motp: float
    num_objects_gt: int
    score_cutoff: float
    def __init__(self, mota: _Optional[float] = ..., motp: _Optional[float] = ..., miss: _Optional[float] = ..., mismatch: _Optional[float] = ..., fp: _Optional[float] = ..., num_objects_gt: _Optional[int] = ..., score_cutoff: _Optional[float] = ..., breakdown: _Optional[_Union[_breakdown_pb2.Breakdown, _Mapping]] = ..., measurements: _Optional[_Union[TrackingMeasurements, _Mapping]] = ...) -> None: ...

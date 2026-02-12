import dataset_pb2 as _dataset_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CompressedFrameLaserData(_message.Message):
    __slots__ = ["laser_calibrations", "lasers", "pose"]
    LASERS_FIELD_NUMBER: _ClassVar[int]
    LASER_CALIBRATIONS_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    laser_calibrations: _containers.RepeatedCompositeFieldContainer[_dataset_pb2.LaserCalibration]
    lasers: _containers.RepeatedCompositeFieldContainer[CompressedLaser]
    pose: _dataset_pb2.Transform
    def __init__(self, lasers: _Optional[_Iterable[_Union[CompressedLaser, _Mapping]]] = ..., laser_calibrations: _Optional[_Iterable[_Union[_dataset_pb2.LaserCalibration, _Mapping]]] = ..., pose: _Optional[_Union[_dataset_pb2.Transform, _Mapping]] = ...) -> None: ...

class CompressedLaser(_message.Message):
    __slots__ = ["name", "ri_return1", "ri_return2"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    RI_RETURN1_FIELD_NUMBER: _ClassVar[int]
    RI_RETURN2_FIELD_NUMBER: _ClassVar[int]
    name: _dataset_pb2.LaserName.Name
    ri_return1: CompressedRangeImage
    ri_return2: CompressedRangeImage
    def __init__(self, name: _Optional[_Union[_dataset_pb2.LaserName.Name, str]] = ..., ri_return1: _Optional[_Union[CompressedRangeImage, _Mapping]] = ..., ri_return2: _Optional[_Union[CompressedRangeImage, _Mapping]] = ...) -> None: ...

class CompressedRangeImage(_message.Message):
    __slots__ = ["range_image_delta_compressed", "range_image_pose_delta_compressed"]
    RANGE_IMAGE_DELTA_COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    RANGE_IMAGE_POSE_DELTA_COMPRESSED_FIELD_NUMBER: _ClassVar[int]
    range_image_delta_compressed: bytes
    range_image_pose_delta_compressed: bytes
    def __init__(self, range_image_delta_compressed: _Optional[bytes] = ..., range_image_pose_delta_compressed: _Optional[bytes] = ...) -> None: ...

class DeltaEncodedData(_message.Message):
    __slots__ = ["mask", "metadata", "residual"]
    MASK_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    RESIDUAL_FIELD_NUMBER: _ClassVar[int]
    mask: _containers.RepeatedScalarFieldContainer[int]
    metadata: Metadata
    residual: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, residual: _Optional[_Iterable[int]] = ..., mask: _Optional[_Iterable[int]] = ..., metadata: _Optional[_Union[Metadata, _Mapping]] = ...) -> None: ...

class Metadata(_message.Message):
    __slots__ = ["quant_precision", "shape"]
    QUANT_PRECISION_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    quant_precision: _containers.RepeatedScalarFieldContainer[float]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., quant_precision: _Optional[_Iterable[float]] = ...) -> None: ...

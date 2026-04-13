from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dronalize.config.models.output import OutputConfig
from dronalize.io import DatasetManifest, adapters, read_manifest
from dronalize.io.backends.pickle import PickleWriter
from dronalize.io.encoding import encode_scene_record, encode_unsplit_scene_record
from dronalize.io.encoding.mds import decode_mds_sample, encode_mds_sample
from dronalize.io.manifest import write_manifest
from dronalize.io.readers import PickleReader
from dronalize.io.records import join_raw_scene_record, split_unsplit_raw_scene_record
from dronalize.runtime.types import OutputPlan
from tests.support import assert_scene_record_equal, make_scene

if TYPE_CHECKING:
    from pathlib import Path


def _output_plan() -> OutputPlan:
    return OutputPlan(
        inner=OutputConfig(
            trajectory_schema="canonical", precision="float32", recenter_positions=True
        )
    )


def test_unsplit_split_helpers_roundtrip_record_contents() -> None:
    scene = make_scene()

    split = encode_scene_record(scene, dtype=np.float64)
    unsplit = encode_unsplit_scene_record(scene, dtype=np.float64)

    rebuilt = split_unsplit_raw_scene_record(unsplit, observation_length=scene.history_frames)
    rejoined = join_raw_scene_record(split)

    np.testing.assert_allclose(unsplit.features, rejoined.features)
    np.testing.assert_array_equal(unsplit.mask, rejoined.mask)
    np.testing.assert_allclose(split.input_features, rebuilt.input_features)
    np.testing.assert_allclose(split.output_features, rebuilt.output_features)


def test_encode_scene_record_uses_passed_agent_ids() -> None:
    scene = replace(make_scene(), passed_agent_ids=frozenset({10}))
    record = encode_scene_record(scene, dtype=np.float32)

    np.testing.assert_array_equal(record.passed_agent_mask, np.array([True, False]))


def test_pickle_writer_roundtrip_preserves_scene_record_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "pickle"
    writer = PickleWriter(output_dir=output_dir, config=_output_plan(), splits=None)

    scene = make_scene()
    expected = encode_scene_record(scene, dtype=np.float32)

    assert writer.write(scene)
    writer.finish_local()
    writer.finish_final()

    reader = PickleReader(output_dir)
    assert len(reader) == 1
    assert_scene_record_equal(reader[0], expected)


def test_mds_encoder_decoder_roundtrip_preserves_data_without_streaming_dependency() -> None:
    scene = make_scene()
    unsplit = encode_unsplit_scene_record(scene, dtype=np.float32)

    sample = encode_mds_sample(unsplit, observation_length=scene.history_frames)
    decoded = decode_mds_sample(sample)
    expected = encode_scene_record(scene, dtype=np.float32)

    assert_scene_record_equal(decoded, expected)


def test_manifest_write_and_read_roundtrip(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        source_trajectory_schema="positions_only",
        trajectory_schema="canonical",
        derived_features=("vx", "vy", "yaw"),
        feature_columns=("x", "y", "vx", "vy", "ax", "ay", "yaw"),
        history_frames=4,
        future_frames=6,
        precision="float32",
        recenter_positions=True,
        has_map=True,
        sample_time=0.1,
        original_sample_time=0.1,
    )

    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)

    assert loaded == manifest


def test_adapter_module_raises_for_unknown_export() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = adapters.__getattr__("DefinitelyMissing")

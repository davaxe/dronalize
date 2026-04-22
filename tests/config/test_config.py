# pyright: standard
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dronalize.config import ProcessingConfig, RuntimeOverride, parse_config
from dronalize.config.models import (
    AgentRangeSpec,
    DatasetConfig,
    ExcludeCategoriesSpec,
    MinSamplesSpec,
    RequireSceneFramesSpec,
    RequireSceneWindowSpec,
)
from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import ConfigurationError
from tests.support import inherited_optional_blocks_descriptor

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, body: str) -> Path:
    config_path = path / "config.toml"
    _ = config_path.write_text(body.strip() + "\n", encoding="utf-8")
    return config_path


def _dataset_config(*, screening: dict[str, object] | None = None) -> DatasetConfig:
    payload: dict[str, object] = {
        "scenes": {"history_frames": 1, "future_frames": 1, "sample_time": 0.1}
    }
    if screening is not None:
        payload["screening"] = screening
    return DatasetConfig.model_validate(payload)


def test_parse_config_parses_profiles_and_dataset_entries(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [profiles.fast.runtime]
            jobs = 2

            [datasets.demo]
            uses = ["fast"]
            """,
        )
    )

    assert isinstance(cfg, ProcessingConfig)
    assert "fast" in cfg.profiles
    assert "demo" in cfg.datasets


def test_resolve_raises_for_missing_profile(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo]
            uses = ["missing"]
            """,
        )
    )

    with pytest.raises(ConfigurationError, match="Profile 'missing' not found"):
        _ = cfg.resolve(
            "demo",
            DatasetConfig.model_validate({
                "scenes": {"history_frames": 1, "future_frames": 1, "sample_time": 0.1}
            }),
        )


def test_parse_config_surfaces_invalid_toml(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        [datasets.demo
        uses = ["fast"]
        """,
    )

    with pytest.raises(ConfigurationError, match="Invalid TOML in config file"):
        _ = parse_config(path)


def test_runtime_override_from_inputs_only_sets_provided_sections() -> None:
    override = RuntimeOverride.from_inputs(
        assign_strategy="scene",
        read_split=None,
        jobs=3,
        trajectory_schema="canonical",
        ratio=(0.7, 0.2, 0.1),
        gap=None,
        segments=None,
    )

    assert override.runtime is not None
    assert override.runtime.jobs == 3
    assert override.output is not None
    assert override.output.trajectory_schema == "canonical"
    assert override.assign is not None

    empty = RuntimeOverride.from_inputs(
        read_strategy=None,
        read_split=None,
        assign_strategy=None,
        jobs=None,
        trajectory_schema=None,
        ratio=None,
        gap=None,
        segments=None,
    )
    assert empty.runtime is None
    assert empty.output is None
    assert empty.read is None
    assert empty.assign is None


def test_runtime_override_from_inputs_rejects_read_split_without_native_read() -> None:
    with pytest.raises(ConfigurationError, match="read_split"):
        _ = RuntimeOverride.from_inputs(read_split=[DatasetSplit.TRAIN])


def test_runtime_override_from_inputs_rejects_assignment_options_without_strategy() -> None:
    with pytest.raises(ConfigurationError, match="Assignment options require"):
        _ = RuntimeOverride.from_inputs(ratio=(0.7, 0.2, 0.1))


def test_runtime_override_from_inputs_rejects_ratio_for_invalid_assign_strategy() -> None:
    with pytest.raises(
        ConfigurationError, match="only valid for scene, source, time, and shuffled-time"
    ):
        _ = RuntimeOverride.from_inputs(assign_strategy="none", ratio=(0.7, 0.2, 0.1))


def test_runtime_override_from_inputs_rejects_gap_for_invalid_assign_strategy() -> None:
    with pytest.raises(ConfigurationError, match="gap"):
        _ = RuntimeOverride.from_inputs(assign_strategy="scene", gap=2, ratio=(0.7, 0.2, 0.1))


def test_runtime_override_from_inputs_requires_segments_for_shuffled_time() -> None:
    with pytest.raises(ConfigurationError, match="segments"):
        _ = RuntimeOverride.from_inputs(assign_strategy="shuffled-time", ratio=(0.7, 0.2, 0.1))


def test_resolve_can_disable_inherited_optional_blocks(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo]
            screening = false

            [datasets.demo.scenes]
            window = false
            resample = false
            lane_change = false
            """,
        )
    )

    resolved = cfg.resolve("demo", inherited_optional_blocks_descriptor())

    assert resolved.screening is None
    assert resolved.scenes.window is None
    assert resolved.scenes.resample is None
    assert resolved.scenes.lane_change is None


def test_screening_extend_is_default_and_keeps_rules_without_inheritance(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening.agent.sample_floor]
            rule = "min_samples"
            minimum = 8
            """,
        )
    )

    resolved = cfg.resolve("demo", _dataset_config())

    assert resolved.screening is not None
    assert resolved.screening.cleanup == {}
    assert resolved.screening.scene == {}
    assert set(resolved.screening.agent) == {"sample_floor"}
    assert isinstance(resolved.screening.agent["sample_floor"], MinSamplesSpec)
    assert resolved.screening.agent["sample_floor"].minimum == 8


def test_screening_extend_merges_namespaces_and_overrides_same_rule_name(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening]
            mode = "extend"

            [datasets.demo.screening.scene.context_window]
            rule = "scene_window"
            start_frame = 0
            end_frame = 3

            [datasets.demo.screening.agent.sample_floor]
            rule = "min_samples"
            minimum = 8
            """,
        )
    )

    resolved = cfg.resolve(
        "demo",
        _dataset_config(
            screening={
                "cleanup": {"trim_static": {"rule": "exclude", "categories": ["STATIC_OBJECT"]}},
                "scene": {"min_context": {"rule": "agent_range", "minimum": 2}},
                "agent": {"sample_floor": {"rule": "min_samples", "minimum": 4}},
            }
        ),
    )

    assert resolved.screening is not None
    assert set(resolved.screening.cleanup) == {"trim_static"}
    assert set(resolved.screening.scene) == {"min_context", "context_window"}
    assert set(resolved.screening.agent) == {"sample_floor"}
    assert isinstance(resolved.screening.cleanup["trim_static"], ExcludeCategoriesSpec)
    assert isinstance(resolved.screening.scene["min_context"], AgentRangeSpec)
    assert isinstance(resolved.screening.scene["context_window"], RequireSceneWindowSpec)
    assert isinstance(resolved.screening.agent["sample_floor"], MinSamplesSpec)
    assert resolved.screening.agent["sample_floor"].minimum == 8


def test_screening_replace_discards_inherited_rules_not_redeclared(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening]
            mode = "replace"

            [datasets.demo.screening.scene.context_window]
            rule = "scene_window"
            start_frame = 0
            end_frame = 3
            """,
        )
    )

    resolved = cfg.resolve(
        "demo",
        _dataset_config(
            screening={
                "cleanup": {"trim_static": {"rule": "exclude", "categories": ["STATIC_OBJECT"]}},
                "scene": {"min_context": {"rule": "agent_range", "minimum": 2}},
                "agent": {"sample_floor": {"rule": "min_samples", "minimum": 4}},
            }
        ),
    )

    assert resolved.screening is not None
    assert resolved.screening.cleanup == {}
    assert set(resolved.screening.scene) == {"context_window"}
    assert resolved.screening.agent == {}
    assert isinstance(resolved.screening.scene["context_window"], RequireSceneWindowSpec)


def test_screening_remove_drops_matching_names_across_all_namespaces(tmp_path: Path) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening]
            mode = "extend"
            remove = ["shared"]
            """,
        )
    )

    resolved = cfg.resolve(
        "demo",
        _dataset_config(
            screening={
                "cleanup": {
                    "shared": {"rule": "exclude", "categories": ["STATIC_OBJECT"]},
                    "keep_cleanup": {"rule": "exclude", "categories": ["ANIMAL"]},
                },
                "scene": {
                    "shared": {"rule": "agent_range", "minimum": 2},
                    "keep_scene": {"rule": "scene_frames", "frames": [0]},
                },
                "agent": {
                    "shared": {"rule": "min_samples", "minimum": 4},
                    "keep_agent": {"rule": "min_samples", "minimum": 2},
                },
            }
        ),
    )

    assert resolved.screening is not None
    assert set(resolved.screening.cleanup) == {"keep_cleanup"}
    assert set(resolved.screening.scene) == {"keep_scene"}
    assert set(resolved.screening.agent) == {"keep_agent"}
    assert isinstance(resolved.screening.cleanup["keep_cleanup"], ExcludeCategoriesSpec)
    assert isinstance(resolved.screening.scene["keep_scene"], RequireSceneFramesSpec)
    assert isinstance(resolved.screening.agent["keep_agent"], MinSamplesSpec)


def test_screening_remove_is_applied_after_replace_and_can_drop_current_rules(
    tmp_path: Path,
) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [datasets.demo.screening]
            mode = "replace"
            remove = ["drop_me"]

            [datasets.demo.screening.agent.drop_me]
            rule = "min_samples"
            minimum = 8

            [datasets.demo.screening.agent.keep_me]
            rule = "min_samples"
            minimum = 3
            """,
        )
    )

    resolved = cfg.resolve("demo", _dataset_config())

    assert resolved.screening is not None
    assert resolved.screening.cleanup == {}
    assert resolved.screening.scene == {}
    assert set(resolved.screening.agent) == {"keep_me"}
    assert isinstance(resolved.screening.agent["keep_me"], MinSamplesSpec)


def test_screening_multiple_profiles_resolve_in_uses_order_then_dataset_block(
    tmp_path: Path,
) -> None:
    cfg = parse_config(
        _write(
            tmp_path,
            """
            [profiles.base.screening.agent.min_obs]
            rule = "min_samples"
            minimum = 4

            [profiles.base.screening.scene.min_context]
            rule = "agent_range"
            minimum = 2

            [profiles.strict.screening]
            mode = "extend"

            [profiles.strict.screening.agent.min_obs]
            rule = "min_samples"
            minimum = 8

            [profiles.strict.screening.agent.anchor_present]
            rule = "frames"
            frames = [19]

            [profiles.curated.screening]
            mode = "replace"

            [profiles.curated.screening.cleanup.trim_static]
            rule = "exclude"
            categories = ["STATIC_OBJECT", "UNIMPORTANT"]

            [profiles.curated.screening.scene.category_mix]
            rule = "category_range"
            ranges = { CAR = { minimum = 1 }, PEDESTRIAN = { minimum = 1 } }

            [datasets.demo]
            uses = ["base", "strict", "curated"]

            [datasets.demo.screening]
            mode = "extend"
            remove = ["category_mix"]

            [datasets.demo.screening.scene.final_context]
            rule = "agent_range"
            minimum = 3
            """,
        )
    )

    resolved = cfg.resolve("demo", _dataset_config())

    assert resolved.screening is not None
    assert set(resolved.screening.cleanup) == {"trim_static"}
    assert set(resolved.screening.scene) == {"final_context"}
    assert isinstance(resolved.screening.cleanup["trim_static"], ExcludeCategoriesSpec)
    assert isinstance(resolved.screening.scene["final_context"], AgentRangeSpec)
    assert resolved.screening.scene["final_context"].minimum == 3
    assert resolved.screening.agent == {}

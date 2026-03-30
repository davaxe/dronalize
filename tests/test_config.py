# pyright: standard

from pathlib import Path

import pytest
from pydantic import ValidationError

from dronalize.core.categories import AgentCategory
from dronalize.core.models import Range
from dronalize.core.scene import (
    CANONICAL_V1,
    POSITIONS_ONLY_V1,
    POSITIONS_VELOCITY_ACCELERATION_V1,
    POSITIONS_VELOCITY_YAW_V1,
)
from dronalize.io import WriterConfig
from dronalize.processing.filters import (
    AgentSelector,
    Filter,
    FilterSpec,
    agent,
    base,
    cleanup,
    scene,
    tol,
)
from dronalize.processing.ingest import BySceneSplit, LoaderConfig, SplitConfig, SplitWeights
from dronalize.processing.maps import (
    BoundingBoxExtraction,
    CircularExtraction,
    FullMapExtraction,
    MapConfig,
    RelevantAreaExtraction,
)
from dronalize.processing.pipeline.functional.resample import ResampleSpec
from dronalize.runtime import Config, ConfigOverrides, load_config_overrides, resolve_runtime_config


def test_map_config_defaults() -> None:
    """Generate a default MapConfig and verify default values."""
    config = MapConfig.default()

    assert config.min_distance == pytest.approx(1.75)
    assert config.interp_distance == pytest.approx(3.0)
    assert config.include_map is True
    assert isinstance(config.extraction, FullMapExtraction)
    assert config.extraction.mode == "full"


def test_map_config_parses_circle() -> None:
    """Parse a dictionary with circle mode to ensure proper union routing."""
    config_dict = {"extraction": {"mode": "circle", "radius": 15.0}}

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, CircularExtraction)
    assert config.extraction.radius == pytest.approx(15.0)


def test_map_config_parses_bounding_box() -> None:
    """Parse a dictionary with bounding_box mode to ensure proper union routing."""
    config_dict = {"extraction": {"mode": "bounding_box", "width": 20.0, "height": 10.0}}

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, BoundingBoxExtraction)
    assert config.extraction.width == pytest.approx(20.0)
    assert config.extraction.height == pytest.approx(10.0)


def test_map_config_rejects_missing_circle() -> None:
    """Raise ValidationError when required parameters for a specific mode are missing."""
    config_dict = {
        "extraction": {
            "mode": "circle"
            # missing "radius"
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        MapConfig.model_validate(config_dict)

    assert "Field required" in str(exc_info.value)
    assert "radius" in str(exc_info.value)


def test_map_config_invalid_mode() -> None:
    """Raise ValidationError when an unsupported extraction mode is provided."""
    config_dict = {
        "extraction": {
            "mode": "triangular",  # invalid mode
            "radius": 5.0,
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        MapConfig.model_validate(config_dict)

    assert (
        "Input tag 'triangular' found using 'mode' does not match any of the expected tags"
        in str(exc_info.value)
    )


def test_map_config_extra_fields() -> None:
    """Raise ValidationError when an unsupported extraction mode is provided."""
    config_dict = {"extraction": {"mode": "circle", "radius": 5.0, "width": 10.0}}
    config = MapConfig.model_validate(config_dict)
    assert isinstance(config.extraction, CircularExtraction)
    assert config.extraction.radius == pytest.approx(5.0)
    # The extra "width" parameter should be ignored and not cause a validation error


def test_map_config_valid_distances() -> None:
    """Initialize MapConfig with valid distance parameters."""
    config = MapConfig(min_distance=2.0, interp_distance=2.5)

    assert config.min_distance == pytest.approx(2.0)
    assert config.interp_distance == pytest.approx(2.5)


def test_map_config_rejects_invalid_distances() -> None:
    """Raise ValidationError when interp_distance is smaller than min_distance."""
    with pytest.raises(ValidationError) as exc_info:
        MapConfig(min_distance=5.0, interp_distance=2.0)

    error_msg = str(exc_info.value)
    assert "Value error" in error_msg
    assert "interp_distance (2.0) must be greater than or equal to min_distance (5.0)" in error_msg


def test_filter_category_from_string() -> None:
    """Parse a single string into a frozenset of AgentCategory."""
    rule = cleanup.ExcludeCategories.define(categories="CAR")

    assert rule.categories == frozenset([AgentCategory.CAR])


def test_filter_category_list() -> None:
    """Parse a list of strings into a frozenset of AgentCategory."""
    rule = cleanup.ExcludeCategories.define(categories=["CAR", "PEDESTRIAN"])

    assert rule.categories == frozenset([AgentCategory.CAR, AgentCategory.PEDESTRIAN])


def test_filter_category_enum() -> None:
    """Accept an existing Enum member directly and coerce it into a frozenset."""
    rule = cleanup.ExcludeCategories.define(categories=AgentCategory.VAN)

    assert rule.categories == frozenset([AgentCategory.VAN])


def test_filter_category_invalid() -> None:
    """Raise ValueError when an invalid agent category string is provided."""
    with pytest.raises(ValueError, match="Unknown agent category"):
        cleanup.ExcludeCategories.define(categories=["CAR", "SPACESHIP"])


def test_frame_rule_from_int() -> None:
    """Parse a single integer into a frozenset for required-agent frames."""
    rule = agent.RequireFrames.define(frames=(19,))

    assert rule.frames == frozenset([19])


def test_frame_rule_from_list() -> None:
    """Parse a list of integers into a frozenset for required-scene frames."""
    rule = scene.RequireFrames.define(frames=[19, 20, 21])

    assert rule.frames == frozenset([19, 20, 21])


def test_filter_defaults_empty() -> None:
    """Empty filters should simply contain no cleanup or check rules."""
    scene_filter = Filter()

    assert scene_filter.cleanup_rules == ()
    assert scene_filter.scene_rules == ()
    assert scene_filter.agent_rules == ()


def test_filter_rejects_duplicate_types() -> None:
    """The same rule type should not appear twice in one filter."""
    with pytest.raises(ValueError, match="Duplicate rule name"):
        Filter.define(scene_rules=[scene.MinimumAgents(minimum=1), scene.MinimumAgents(minimum=2)])


def test_filter_allows_distinct_ids() -> None:
    """Rules of the same type may coexist when they use different effective keys."""
    scene_filter = Filter.define(
        scene_rules=[
            scene.MinimumAgents(
                minimum=1, rule_id="min_cars", selector=AgentSelector.include(["CAR"])
            ),
            scene.MinimumAgents(
                minimum=1, rule_id="min_pedestrians", selector=AgentSelector.include(["PEDESTRIAN"])
            ),
        ]
    )

    assert tuple(rule.name() for rule in scene_filter.scene_rules) == (
        "min_cars",
        "min_pedestrians",
    )


def test_filter_rejects_duplicate_ids() -> None:
    """Duplicate effective keys should still be rejected when explicit ids are used."""
    with pytest.raises(ValueError, match="Duplicate rule name"):
        Filter.define(
            scene_rules=[
                scene.MinimumAgents(minimum=1, rule_id="min_dynamic"),
                scene.RequireWindow(start_frame=0, end_frame=1, rule_id="min_dynamic"),
            ]
        )


def test_filter_rejects_invalid_rule_id() -> None:
    """Rule ids must stay slug-safe for stable diagnostics."""
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        scene.MinimumAgents(minimum=1, rule_id="Bad-ID")


def test_filter_exports_categories() -> None:
    """The package root should expose both cleanup rule variants."""
    rule = cleanup.IncludeCategories.define(categories=["CAR"])

    assert rule.categories == frozenset([AgentCategory.CAR])


def test_filter_define_collects_rules() -> None:
    """Direct rule instances should be stored as immutable tuples."""
    scene_filter = Filter.define(
        cleanup_rules=[
            cleanup.IncludeCategories.define(categories="STATIC_OBJECT", rule_id="only_static")
        ],
        scene_rules=[
            scene.MinimumAgents(
                minimum=3, rule_id="min_static", selector=AgentSelector.include("STATIC_OBJECT")
            )
        ],
        agent_rules=[
            agent.MaxMissingFrames(
                maximum=0,
                selector=AgentSelector.exclude("STATIC_OBJECT"),
                tolerance=tol(absolute=1, relative=0.25),
                rule_id="complete_dynamic",
            ),
            agent.RequireFrames.define(frames=[10, 15], rule_id="anchor_frames"),
            agent.MinSamples(minimum=5, rule_id="sample_floor"),
        ],
    )

    assert isinstance(scene_filter.cleanup_rules[0], cleanup.IncludeCategories)
    assert scene_filter.cleanup_rules[0].categories == frozenset([AgentCategory.STATIC_OBJECT])
    assert scene_filter.cleanup_rules[0].rule_id == "only_static"
    assert isinstance(scene_filter.scene_rules[0], scene.MinimumAgents)
    assert scene_filter.scene_rules[0].minimum == 3
    assert scene_filter.scene_rules[0].rule_id == "min_static"
    assert isinstance(scene_filter.agent_rules[0], agent.MaxMissingFrames)
    assert scene_filter.agent_rules[0].tolerance == tol(absolute=1, relative=0.25)
    assert scene_filter.agent_rules[0].selector == AgentSelector.exclude(["STATIC_OBJECT"])
    assert isinstance(scene_filter.agent_rules[1], agent.RequireFrames)
    assert scene_filter.agent_rules[1].frames == frozenset([10, 15])
    assert scene_filter.agent_rules[1].rule_id == "anchor_frames"
    assert isinstance(scene_filter.agent_rules[2], agent.MinSamples)
    assert scene_filter.agent_rules[2].minimum == 5
    assert scene_filter.agent_rules[2].rule_id == "sample_floor"


def test_filter_spec_parses_basic_rules() -> None:
    """Config-facing filter specs should parse discriminated cleanup and check rules."""
    spec = FilterSpec.model_validate({
        "mode": "replace",
        "cleanup": [{"type": "exclude", "categories": ["CAR", "PEDESTRIAN"]}],
        "scene": [{"type": "min_agents", "minimum": 2}],
        "agent": [
            {
                "type": "frames",
                "frames": [0, 4],
                "tolerance": {"kind": "combined", "absolute": 1, "relative": 0.25},
            }
        ],
    })

    resolved = spec.resolve()

    assert resolved.cleanup_rules == (
        cleanup.ExcludeCategories.define(categories=["CAR", "PEDESTRIAN"]),
    )
    assert resolved.scene_rules == (scene.MinimumAgents(minimum=2),)
    assert resolved.agent_rules == (
        agent.RequireFrames.define(frames=[0, 4], tolerance=tol(absolute=1, relative=0.25)),
    )


def test_filter_spec_parses_ids_and_selectors() -> None:
    """Config-facing filter specs should resolve the extended rule capabilities."""
    spec = FilterSpec.model_validate({
        "mode": "replace",
        "cleanup": [{"type": "include", "categories": ["CAR"], "rule_id": "cars_only"}],
        "scene": [
            {
                "type": "min_agents",
                "minimum": 2,
                "rule_id": "min_cars",
                "selector": {"mode": "include", "categories": ["CAR"]},
            },
            {
                "type": "window",
                "start_frame": 0,
                "end_frame": 4,
                "min_fraction": 0.6,
                "rule_id": "scene_window_obs",
            },
            {"type": "max_missing_frames", "max_missing_frames": 1, "rule_id": "gaps_ok"},
        ],
        "agent": [
            {
                "type": "window",
                "start_frame": 0,
                "end_frame": 4,
                "min_fraction": 0.6,
                "selector": {"mode": "exclude", "categories": ["PEDESTRIAN"]},
                "tolerance": {"kind": "combined", "absolute": 1, "relative": 0.5},
                "rule_id": "dynamic_window",
            }
        ],
    })

    resolved = spec.resolve()

    assert resolved.cleanup_rules == (
        cleanup.IncludeCategories.define(categories=["CAR"], rule_id="cars_only"),
    )
    assert resolved.scene_rules == (
        scene.MinimumAgents(minimum=2, rule_id="min_cars", selector=AgentSelector.include(["CAR"])),
        scene.RequireWindow(
            start_frame=0, end_frame=4, min_fraction=0.6, rule_id="scene_window_obs"
        ),
        scene.MaxMissingFrames(max_missing_frames=1, rule_id="gaps_ok"),
    )
    assert resolved.agent_rules == (
        agent.RequireWindow(
            start_frame=0,
            end_frame=4,
            min_fraction=0.6,
            selector=AgentSelector.exclude(["PEDESTRIAN"]),
            tolerance=tol(absolute=1, relative=0.5),
            rule_id="dynamic_window",
        ),
    )


def test_filter_spec_parses_new_rules() -> None:
    """Config-facing filter specs should parse the newer rule variants."""
    spec = FilterSpec.model_validate({
        "mode": "replace",
        "scene": [
            {"type": "agent_range", "minimum": 2, "maximum": 4},
            {
                "type": "category_range",
                "ranges": {"CAR": {"minimum": 1, "maximum": 2}, "PEDESTRIAN": {"minimum": 1}},
            },
        ],
        "agent": [
            {"type": "starts_by_frame", "frame": 1},
            {"type": "ends_after_frame", "frame": 6},
            {"type": "min_span", "minimum": 5},
        ],
    })

    resolved = spec.resolve()

    assert resolved.scene_rules == (
        scene.AgentRange(minimum=2, maximum=4),
        scene.CategoryRange(
            ranges={"CAR": Range(minimum=1, maximum=2), "PEDESTRIAN": Range(minimum=1)}
        ),
    )
    assert resolved.agent_rules == (
        agent.StartsByFrame(frame=1),
        agent.EndsAfterFrame(frame=6),
        agent.MinSpan(minimum=5),
    )


def test_filter_rules_expose_protocols() -> None:
    """Rule instances should advertise the expected cleanup and check protocols."""
    rules = [
        cleanup.ExcludeCategories.define(categories=["CAR"]),
        scene.MinimumAgents(minimum=1),
        scene.AgentRange(minimum=1, maximum=3),
        scene.CategoryRange(ranges={"CAR": Range(minimum=1)}),
        scene.RequireFrames.define(frames=[0]),
        scene.RequireWindow(start_frame=0, end_frame=1),
        scene.MaxMissingFrames(),
        agent.MaxMissingFrames(maximum=0),
        agent.RequireFrames.define(frames=[0]),
        agent.RequireWindow(start_frame=0, end_frame=1),
        agent.MinSamples(minimum=2),
        agent.StartsByFrame(frame=1),
        agent.EndsAfterFrame(frame=1),
        agent.MinSpan(minimum=2),
    ]

    assert isinstance(rules[0], base.CleanupRule)
    assert all(isinstance(rule, base.SceneCheckRule) for rule in rules[1:7])
    assert all(isinstance(rule, base.AgentCheckRule) for rule in rules[7:])


def test_runtime_config_keeps_filter_objects() -> None:
    """Runtime config merging should preserve direct filter objects from the default config."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filter(
            Filter.define(
                cleanup_rules=[cleanup.ExcludeCategories.define(categories=["CAR"])],
                scene_rules=[scene.MinimumAgents(minimum=2)],
                agent_rules=[agent.RequireFrames.define(frames=[0])],
            )
        ),
        map=MapConfig.default(),
    )

    resolved = resolve_runtime_config(default=default, overrides={"execution": {"workers": 8}})

    assert resolved.loader.filter is not None
    assert resolved.loader.filter == default.loader.filter
    assert resolved.execution.workers == 8


def test_runtime_config_merges_filters() -> None:
    """Filter specs in merge mode should replace same-name rules and keep other defaults."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filter(
            Filter.define(
                cleanup_rules=[cleanup.ExcludeCategories.define(categories=["CAR"])],
                scene_rules=[scene.MinimumAgents(minimum=2)],
                agent_rules=[
                    agent.RequireFrames.define(frames=[0], rule_id="obs_anchor"),
                    agent.RequireFrames.define(frames=[4], rule_id="pred_anchor"),
                ],
            )
        ),
        map=MapConfig.default(),
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides={
            "loader": {
                "filter": {
                    "mode": "merge",
                    "agent": [
                        {"type": "frames", "frames": [1], "rule_id": "obs_anchor"},
                        {"type": "min_samples", "minimum": 3},
                    ],
                }
            }
        },
    )
    assert resolved.loader.filter == Filter.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=["CAR"])],
        scene_rules=[scene.MinimumAgents(minimum=2)],
        agent_rules=[
            agent.RequireFrames.define(frames=[1], rule_id="obs_anchor"),
            agent.RequireFrames.define(frames=[4], rule_id="pred_anchor"),
            agent.MinSamples(minimum=3),
        ],
    )


def test_runtime_config_deep_merge() -> None:
    """Deep merges should preserve unspecified nested fields from the default config."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_resampling(
            ResampleSpec(up=2, down=1)
        ),
        map=MapConfig.default(),
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides={
            "execution": {"workers": 8},
            "loader": {"window": {"window_size": 5, "step_size": 1}},
        },
    )

    assert resolved.execution.workers == 8
    assert resolved.execution.parallel is False
    assert resolved.loader.window is not None
    assert resolved.loader.window.window_size == 5
    assert resolved.loader.resampling == default.loader.resampling


def test_writer_schema_drives_layout() -> None:
    """Writer config should derive schema and feature layout from the selected schema."""
    default_config = WriterConfig()
    positions_only = WriterConfig.create(scene_schema="positions_only")
    positions_velocity_acceleration = WriterConfig.create(
        scene_schema="positions_velocity_acceleration"
    )
    positions_velocity_yaw = WriterConfig.create(scene_schema="positions_velocity_yaw")

    assert default_config.scene_schema == CANONICAL_V1
    assert default_config.feature_dim == 7
    assert default_config.feature_columns == ("x", "y", "vx", "vy", "ax", "ay", "yaw")

    assert positions_only.scene_schema == POSITIONS_ONLY_V1
    assert positions_only.feature_dim == 2
    assert positions_only.feature_columns == ("x", "y")

    assert positions_velocity_acceleration.scene_schema == POSITIONS_VELOCITY_ACCELERATION_V1
    assert positions_velocity_acceleration.feature_dim == 6
    assert positions_velocity_acceleration.feature_columns == ("x", "y", "vx", "vy", "ax", "ay")

    assert positions_velocity_yaw.scene_schema == POSITIONS_VELOCITY_YAW_V1
    assert positions_velocity_yaw.feature_dim == 5
    assert positions_velocity_yaw.feature_columns == ("x", "y", "vx", "vy", "yaw")


def test_runtime_config_merges_writer() -> None:
    """Writer overrides should merge into the default writer config."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1), map=MapConfig.default()
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides={"writer": {"scene_schema": "positions_only", "precision": "float64"}},
    )

    assert resolved.writer.scene_schema == POSITIONS_ONLY_V1
    assert resolved.writer.precision == "float64"
    assert resolved.writer.offset_positions is True


def test_runtime_config_merges_split() -> None:
    """Split overrides should validate and merge through the top-level config model."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1), map=MapConfig.default()
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides={
            "split": {
                "strategy": {"type": "by_scene"},
                "weights": {"train": 0.7, "val": 0.2, "test": 0.1},
            }
        },
    )

    assert resolved.split == SplitConfig(
        strategy=BySceneSplit(), weights=SplitWeights(train=0.7, val=0.2, test=0.1)
    )


def test_load_overrides_global_section(tmp_path: Path) -> None:
    """Global config blocks should be merged into every dataset-specific override block."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[global.execution]
workers = 4

[a43.execution]
parallel = true

[waymo.execution]
chunksize = 8
""",
        encoding="utf-8",
    )

    overrides = load_config_overrides(config_path)

    assert overrides == ConfigOverrides(
        datasets={
            "a43": {"execution": {"workers": 4, "parallel": True}},
            "waymo": {"execution": {"workers": 4, "chunksize": 8}},
        }
    )
    assert overrides.for_dataset("a43") == {"execution": {"workers": 4, "parallel": True}}
    assert overrides.for_dataset("waymo") == {"execution": {"workers": 4, "chunksize": 8}}
    assert overrides.for_dataset("nuscenes") == {}


def test_load_overrides_global_split(tmp_path: Path) -> None:
    """Global split settings should merge into each dataset-specific override block."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[global.split.weights]
train = 0.7
val = 0.2
test = 0.1

[waymo.split.strategy]
type = "by_scene"
""",
        encoding="utf-8",
    )

    overrides = load_config_overrides(config_path)

    assert overrides.for_dataset("waymo") == {
        "split": {
            "weights": {"train": 0.7, "val": 0.2, "test": 0.1},
            "strategy": {"type": "by_scene"},
        }
    }


def test_load_overrides_filter_specs(tmp_path: Path) -> None:
    """TOML loader filter tables should resolve into runtime filter objects."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[a43.loader.filter]
mode = "replace"

[[a43.loader.filter.cleanup]]
type = "exclude"
categories = ["CAR"]

[[a43.loader.filter.scene]]
type = "min_agents"
minimum = 3

[[a43.loader.filter.agent]]
type = "frames"
frames = [0, 4]
tolerance = { kind = "combined", absolute = 1, relative = 0.2 }
""",
        encoding="utf-8",
    )

    overrides = load_config_overrides(config_path)
    resolved = resolve_runtime_config(
        default=Config(
            loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1), map=MapConfig.default()
        ),
        overrides=overrides.for_dataset("a43"),
    )
    assert resolved.loader.filter == Filter.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=["CAR"])],
        scene_rules=[scene.MinimumAgents(minimum=3)],
        agent_rules=[
            agent.RequireFrames.define(frames=[0, 4], tolerance=tol(absolute=1, relative=0.2))
        ],
    )


def test_load_overrides_extended_filters(tmp_path: Path) -> None:
    """TOML loader filter tables should resolve ids, selectors, and window rules."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[a43.loader.filter]
mode = "replace"

[[a43.loader.filter.cleanup]]
type = "include"
categories = ["CAR"]
rule_id = "cars_only"

[[a43.loader.filter.scene]]
type = "min_agents"
minimum = 2
rule_id = "min_cars"

[a43.loader.filter.scene.selector]
mode = "include"
categories = ["CAR"]

[[a43.loader.filter.scene]]
type = "window"
start_frame = 0
end_frame = 4
min_fraction = 0.6
rule_id = "obs_window"

[[a43.loader.filter.agent]]
type = "window"
start_frame = 0
end_frame = 4
min_fraction = 0.8
tolerance = { kind = "combined", absolute = 1, "relative" = 0.5 }
rule_id = "dynamic_window"

[a43.loader.filter.agent.selector]
mode = "exclude"
categories = ["PEDESTRIAN"]
""",
        encoding="utf-8",
    )

    overrides = load_config_overrides(config_path)
    resolved = resolve_runtime_config(
        default=Config(
            loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1), map=MapConfig.default()
        ),
        overrides=overrides.for_dataset("a43"),
    )
    assert resolved.loader.filter == Filter.define(
        cleanup_rules=[cleanup.IncludeCategories.define(categories=["CAR"], rule_id="cars_only")],
        scene_rules=[
            scene.MinimumAgents(
                minimum=2, rule_id="min_cars", selector=AgentSelector.include(["CAR"])
            ),
            scene.RequireWindow(start_frame=0, end_frame=4, min_fraction=0.6, rule_id="obs_window"),
        ],
        agent_rules=[
            agent.RequireWindow(
                start_frame=0,
                end_frame=4,
                min_fraction=0.8,
                tolerance=tol(absolute=1, relative=0.5),
                selector=AgentSelector.exclude(["PEDESTRIAN"]),
                rule_id="dynamic_window",
            )
        ],
    )


def test_map_config_parses_relevant_area() -> None:
    """Parse a dictionary with relevant mode to ensure proper union routing."""
    config_dict = {"extraction": {"mode": "relevant", "padding_factor": 1.3}}

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, RelevantAreaExtraction)
    assert config.extraction.mode == "relevant"
    assert config.extraction.padding_factor == pytest.approx(1.3)


def test_map_config_parses_full_map() -> None:
    """Parse a dictionary with full mode to ensure proper union routing."""
    config_dict = {"extraction": {"mode": "full"}}

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, FullMapExtraction)
    assert config.extraction.mode == "full"


def test_writer_schema_positions_only() -> None:
    """Verify initialization with the 'positions_only' shorthand string."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({**base, "scene_schema": "positions_only"})

    assert config.scene_schema == POSITIONS_ONLY_V1
    assert config.feature_columns == ("x", "y")
    assert config.feature_dim == 2


def test_writer_schema_predefined() -> None:
    """Ensure the config accepts a predefined schema object."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({
        **base,
        "scene_schema": POSITIONS_VELOCITY_ACCELERATION_V1,
    })

    assert config.scene_schema == POSITIONS_VELOCITY_ACCELERATION_V1
    assert config.feature_columns == ("x", "y", "vx", "vy", "ax", "ay")
    assert config.feature_dim == 6


def test_writer_schema_single_custom() -> None:
    """Check that a single custom field is correctly appended to base fields."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({**base, "scene_schema": "vx"})

    assert config.feature_columns == ("x", "y", "vx")
    assert config.feature_dim == 3
    assert config.scene_schema.name == "custom: vx"


def test_writer_schema_multiple_custom() -> None:
    """Validate that multiple colon-separated fields generate a custom schema."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({**base, "scene_schema": "vx:vy:yaw"})

    assert config.feature_columns == ("x", "y", "vx", "vy", "yaw")
    assert config.feature_dim == 5
    assert config.scene_schema.name == "custom: vx:vy:yaw"


def test_writer_schema_custom_order() -> None:
    """Confirm that the internal representation maintains consistent field ordering."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({**base, "scene_schema": "yaw:vx:vy"})

    assert config.feature_columns == ("x", "y", "vx", "vy", "yaw")
    assert config.feature_dim == 5
    assert config.scene_schema.name == "custom: yaw:vx:vy"


def test_writer_schema_rejects_invalid() -> None:
    """Raise ValidationError when required base fields are missing from a custom schema."""
    base = {"precision": "float64", "offset_positions": False}
    with pytest.raises(ValidationError, match=r"must include the base fields"):
        WriterConfig.model_validate({
            **base,
            "scene_schema": {"name": "custom_schema", "fields": ["x", "y", "ax"]},
        })

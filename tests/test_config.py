# pyright: standard

from pathlib import Path

import pytest
from pydantic import ValidationError

from dronalize.core.categories import AgentCategory
from dronalize.core.scene import (
    CANONICAL_V1,
    POSITIONS_ONLY_V1,
    POSITIONS_VELOCITY_ACCELERATION_V1,
    POSITIONS_VELOCITY_YAW_V1,
)
from dronalize.io import WriterConfig
from dronalize.processing.filters import (
    AgentFilterRule,
    CleanupRule,
    DropAgentCategories,
    Filter,
    FilterSpec,
    MinimumAgents,
    MinimumAgentSamples,
    RequireAgentFrames,
    RequireContiguousSceneFrames,
    RequireFullAgentWindow,
    RequireSceneFrames,
    SceneFilterRule,
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
from dronalize.runtime import (
    Config,
    ConfigOverrides,
    load_config_overrides,
    resolve_runtime_config,
)


def test_circular_extraction_valid() -> None:
    """Initialize CircularExtraction with a valid radius."""
    extraction = CircularExtraction(radius=10.5)

    assert extraction.mode == "circle"
    assert extraction.radius == pytest.approx(10.5)


def test_bounding_box_extraction_valid() -> None:
    """Initialize BoundingBoxExtraction with valid dimensions."""
    extraction = BoundingBoxExtraction(width=5.0, height=8.0)

    assert extraction.mode == "bounding_box"
    assert extraction.width == pytest.approx(5.0)
    assert extraction.height == pytest.approx(8.0)


def test_map_config_default_initialization() -> None:
    """Generate a default MapConfig and verify default values."""
    config = MapConfig.default()

    assert config.min_distance == pytest.approx(1.75)
    assert config.interp_distance == pytest.approx(3.0)
    assert config.include_map is True
    assert isinstance(config.extraction, FullMapExtraction)
    assert config.extraction.mode == "full"


def test_map_config_discriminator_circular() -> None:
    """Parse a dictionary with circle mode to ensure proper union routing."""
    config_dict = {
        "extraction": {
            "mode": "circle",
            "radius": 15.0,
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, CircularExtraction)
    assert config.extraction.radius == pytest.approx(15.0)


def test_map_config_discriminator_bounding_box() -> None:
    """Parse a dictionary with bounding_box mode to ensure proper union routing."""
    config_dict = {
        "extraction": {
            "mode": "bounding_box",
            "width": 20.0,
            "height": 10.0,
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, BoundingBoxExtraction)
    assert config.extraction.width == pytest.approx(20.0)
    assert config.extraction.height == pytest.approx(10.0)


def test_map_config_missing_discriminator_params() -> None:
    """Raise ValidationError when required parameters for a specific mode are missing."""
    config_dict = {
        "extraction": {
            "mode": "circle",
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


def test_map_config_additional_params() -> None:
    """Raise ValidationError when an unsupported extraction mode is provided."""
    config_dict = {
        "extraction": {
            "mode": "circle",
            "radius": 5.0,
            "width": 10.0,
        }
    }
    config = MapConfig.model_validate(config_dict)
    assert isinstance(config.extraction, CircularExtraction)
    assert config.extraction.radius == pytest.approx(5.0)
    # The extra "width" parameter should be ignored and not cause a validation error


def test_map_config_distance_validation_success() -> None:
    """Initialize MapConfig with valid distance parameters."""
    config = MapConfig(min_distance=2.0, interp_distance=2.5)

    assert config.min_distance == pytest.approx(2.0)
    assert config.interp_distance == pytest.approx(2.5)


def test_map_config_distance_validation_failure() -> None:
    """Raise ValidationError when interp_distance is smaller than min_distance."""
    with pytest.raises(ValidationError) as exc_info:
        MapConfig(min_distance=5.0, interp_distance=2.0)

    error_msg = str(exc_info.value)
    assert "Value error" in error_msg
    assert "interp_distance (2.0) must be greater than or equal to min_distance (5.0)" in error_msg


def test_filters_parse_single_agent_category_string() -> None:
    """Parse a single string into a frozenset of AgentCategory."""
    rule = DropAgentCategories.define(categories="CAR")

    assert rule.categories == frozenset([AgentCategory.CAR])


def test_filters_parse_list_agent_category_strings() -> None:
    """Parse a list of strings into a frozenset of AgentCategory."""
    rule = DropAgentCategories.define(categories=["CAR", "PEDESTRIAN"])

    assert rule.categories == frozenset([AgentCategory.CAR, AgentCategory.PEDESTRIAN])


def test_filters_accept_existing_enum_member() -> None:
    """Accept an existing Enum member directly and coerce it into a frozenset."""
    rule = DropAgentCategories.define(categories=AgentCategory.VAN)

    assert rule.categories == frozenset([AgentCategory.VAN])


def test_filters_reject_invalid_agent_category() -> None:
    """Raise ValueError when an invalid agent category string is provided."""
    with pytest.raises(ValueError, match="Unknown agent category"):
        DropAgentCategories.define(categories=["CAR", "SPACESHIP"])


def test_filter_rules_parse_single_frame_int() -> None:
    """Parse a single integer into a frozenset for required-agent frames."""
    rule = RequireAgentFrames.define(frames=(19,))

    assert rule.frames == frozenset([19])


def test_filter_rules_parse_list_frames() -> None:
    """Parse a list of integers into a frozenset for required-scene frames."""
    rule = RequireSceneFrames.define(frames=[19, 20, 21])

    assert rule.frames == frozenset([19, 20, 21])


def test_filters_default_to_empty_rule_lists() -> None:
    """Empty filters should simply contain no cleanup or validation rules."""
    filters = Filter()

    assert filters.cleanup_rules == ()
    assert filters.filter_rules == ()


def test_filters_reject_duplicate_rule_types() -> None:
    """The same rule type should not appear twice in one filter."""
    with pytest.raises(ValueError, match="Duplicate rule name"):
        Filter.define(filter_rules=[MinimumAgents(minimum=1), MinimumAgents(minimum=2)])


def test_filter_collects_direct_rule_objects() -> None:
    """Direct rule instances should be stored as immutable tuples."""
    filters = Filter.define(
        cleanup_rules=[DropAgentCategories.define(categories="STATIC_OBJECT")],
        filter_rules=[
            MinimumAgents(minimum=3),
            RequireFullAgentWindow(max_invalid_agents=1, max_invalid_fraction=0.25),
            RequireAgentFrames.define(frames=[10, 15]),
            MinimumAgentSamples(minimum=5),
        ],
    )

    assert isinstance(filters.cleanup_rules[0], DropAgentCategories)
    assert filters.cleanup_rules[0].categories == frozenset([AgentCategory.STATIC_OBJECT])
    assert isinstance(filters.filter_rules[0], MinimumAgents)
    assert filters.filter_rules[0].minimum == 3
    assert isinstance(filters.filter_rules[1], RequireFullAgentWindow)
    assert filters.filter_rules[1].max_invalid_agents == 1
    assert filters.filter_rules[1].max_invalid_fraction == pytest.approx(0.25)
    assert isinstance(filters.filter_rules[2], RequireAgentFrames)
    assert filters.filter_rules[2].frames == frozenset([10, 15])
    assert isinstance(filters.filter_rules[3], MinimumAgentSamples)
    assert filters.filter_rules[3].minimum == 5


def test_filter_spec_parses_discriminated_rules() -> None:
    """Config-facing filter specs should parse discriminated cleanup and validation rules."""
    spec = FilterSpec.model_validate({
        "mode": "replace",
        "cleanup": [{"type": "drop_agent_categories", "categories": ["CAR", "PEDESTRIAN"]}],
        "validate": [
            {"type": "minimum_agents", "minimum": 2},
            {
                "type": "require_agent_frames",
                "frames": [0, 4],
                "max_invalid_agents": 1,
                "max_invalid_fraction": 0.25,
            },
        ],
    })

    resolved = spec.resolve()

    assert resolved.cleanup_rules == (
        DropAgentCategories.define(categories=["CAR", "PEDESTRIAN"]),
    )
    assert resolved.filter_rules == (
        MinimumAgents(minimum=2),
        RequireAgentFrames.define(frames=[0, 4], max_invalid_agents=1, max_invalid_fraction=0.25),
    )


def test_loader_config_with_filters_rejects_out_of_range_negative_frame() -> None:
    """Reject negative frame references that fall outside the configured sequence length."""
    with pytest.raises(ValueError, match=r"Frame index -6 is out of range"):
        LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filters(
            Filter.define(filter_rules=[RequireAgentFrames.define(frames=[-6])]),
        )


def test_loader_config_normalizes_negative_required_agent_frames() -> None:
    """Negative frame references should resolve from the end of the configured sequence."""
    config = LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filters(
        Filter.define(filter_rules=[RequireAgentFrames.define(frames=[-1, 0])]),
    )

    assert config.filters is not None
    assert config.filters.filter_rules == (
        RequireAgentFrames.define(frames=[4, 0]),
    )


def test_loader_config_normalizes_negative_required_scene_frames() -> None:
    """Scene-level frame requirements should support negative indices as well."""
    config = LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filters(
        Filter.define(filter_rules=[RequireSceneFrames.define(frames=[-1, -2, 0])]),
    )

    assert config.filters is not None
    assert config.filters.filter_rules == (
        RequireSceneFrames.define(frames=[4, 3, 0]),
    )


def test_filter_rules_expose_expected_protocols() -> None:
    """Rule instances should advertise the expected cleanup and validation protocols."""
    rules = [
        DropAgentCategories.define(categories=["CAR"]),
        MinimumAgents(minimum=1),
        RequireSceneFrames.define(frames=[0]),
        RequireContiguousSceneFrames(),
        RequireFullAgentWindow(),
        RequireAgentFrames.define(frames=[0]),
        MinimumAgentSamples(minimum=2),
    ]

    assert isinstance(rules[0], CleanupRule)
    assert all(isinstance(rule, SceneFilterRule) for rule in rules[1:4])
    assert all(isinstance(rule, AgentFilterRule) for rule in rules[4:])


def test_resolve_runtime_config_preserves_filter_objects() -> None:
    """Runtime config merging should preserve direct filter objects from the default config."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filters(
            Filter.define(
                cleanup_rules=[DropAgentCategories.define(categories=["CAR"])],
                filter_rules=[
                    MinimumAgents(minimum=2),
                    RequireAgentFrames.define(frames=[0]),
                ],
            )
        ),
        map=MapConfig.default(),
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides={
            "execution": {"workers": 8},
        },
    )

    assert resolved.loader.filters is not None
    assert resolved.loader.filters == default.loader.filters
    assert resolved.execution.workers == 8


def test_resolve_runtime_config_merges_filter_specs_on_top_of_default_filters() -> None:
    """Filter specs in merge mode should replace same-name rules and keep other defaults."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filters(
            Filter.define(
                cleanup_rules=[DropAgentCategories.define(categories=["CAR"])],
                filter_rules=[
                    RequireAgentFrames.define(frames=[0]),
                    MinimumAgents(minimum=2),
                ],
            )
        ),
        map=MapConfig.default(),
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides={
            "loader": {
                "filters": {
                    "mode": "merge",
                    "validate": [
                        {"type": "require_agent_frames", "frames": [1]},
                        {"type": "minimum_agent_samples", "minimum": 3},
                    ],
                }
            }
        },
    )

    assert resolved.loader.filters == Filter.define(
        cleanup_rules=[DropAgentCategories.define(categories=["CAR"])],
        filter_rules=[
            RequireAgentFrames.define(frames=[1]),
            MinimumAgents(minimum=2),
            MinimumAgentSamples(minimum=3),
        ],
    )


def test_resolve_runtime_config_deep_merges_partial_overrides() -> None:
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


def test_writer_config_scene_schema_drives_schema_and_feature_layout() -> None:
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


def test_resolve_runtime_config_merges_writer_overrides() -> None:
    """Writer overrides should merge into the default writer config."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1),
        map=MapConfig.default(),
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides={
            "writer": {
                "scene_schema": "positions_only",
                "precision": "float64",
            }
        },
    )

    assert resolved.writer.scene_schema == POSITIONS_ONLY_V1
    assert resolved.writer.precision == "float64"
    assert resolved.writer.offset_positions is True


def test_resolve_runtime_config_merges_split_overrides() -> None:
    """Split overrides should validate and merge through the top-level config model."""
    default = Config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1),
        map=MapConfig.default(),
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
        strategy=BySceneSplit(),
        weights=SplitWeights(train=0.7, val=0.2, test=0.1),
    )


def test_load_config_overrides_merges_global_section_into_each_dataset(tmp_path: Path) -> None:
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
        },
    )
    assert overrides.for_dataset("a43") == {"execution": {"workers": 4, "parallel": True}}
    assert overrides.for_dataset("waymo") == {"execution": {"workers": 4, "chunksize": 8}}
    assert overrides.for_dataset("nuscenes") == {}


def test_load_config_overrides_merges_global_split_section(tmp_path: Path) -> None:
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


def test_load_config_overrides_supports_loader_filter_specs(tmp_path: Path) -> None:
    """TOML loader filter tables should resolve into runtime filter objects."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """[a43.loader.filters]
mode = "replace"

[[a43.loader.filters.cleanup]]
type = "drop_agent_categories"
categories = ["CAR"]

[[a43.loader.filters.validate]]
type = "minimum_agents"
minimum = 3

[[a43.loader.filters.validate]]
type = "require_agent_frames"
frames = [0, -1]
max_invalid_agents = 1
max_invalid_fraction = 0.2
""",
        encoding="utf-8",
    )

    overrides = load_config_overrides(config_path)
    resolved = resolve_runtime_config(
        default=Config(
            loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1),
            map=MapConfig.default(),
        ),
        overrides=overrides.for_dataset("a43"),
    )

    assert resolved.loader.filters == Filter.define(
        cleanup_rules=[DropAgentCategories.define(categories=["CAR"])],
        filter_rules=[
            MinimumAgents(minimum=3),
            RequireAgentFrames.define(
                frames=[0, 4],
                max_invalid_agents=1,
                max_invalid_fraction=0.2,
            ),
        ],
    )


def test_map_config_discriminator_relevant_area() -> None:
    """Parse a dictionary with relevant mode to ensure proper union routing."""
    config_dict = {
        "extraction": {
            "mode": "relevant",
            "padding_factor": 1.3,
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, RelevantAreaExtraction)
    assert config.extraction.mode == "relevant"
    assert config.extraction.padding_factor == pytest.approx(1.3)


def test_map_config_discriminator_full_map() -> None:
    """Parse a dictionary with full mode to ensure proper union routing."""
    config_dict = {
        "extraction": {
            "mode": "full",
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, FullMapExtraction)
    assert config.extraction.mode == "full"


def test_writer_config_positions_only_schema() -> None:
    """Verify initialization with the 'positions_only' shorthand string."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({**base, "scene_schema": "positions_only"})

    assert config.scene_schema == POSITIONS_ONLY_V1
    assert config.feature_columns == ("x", "y")
    assert config.feature_dim == 2


def test_writer_config_predefined_object_schema() -> None:
    """Ensure the config accepts a predefined schema object."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({
        **base,
        "scene_schema": POSITIONS_VELOCITY_ACCELERATION_V1,
    })

    assert config.scene_schema == POSITIONS_VELOCITY_ACCELERATION_V1
    assert config.feature_columns == ("x", "y", "vx", "vy", "ax", "ay")
    assert config.feature_dim == 6


def test_writer_config_single_field_custom_schema() -> None:
    """Check that a single custom field is correctly appended to base fields."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({**base, "scene_schema": "vx"})

    assert config.feature_columns == ("x", "y", "vx")
    assert config.feature_dim == 3
    assert config.scene_schema.name == "custom: vx"


def test_writer_config_multi_field_custom_schema() -> None:
    """Validate that multiple colon-separated fields generate a custom schema."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({**base, "scene_schema": "vx:vy:yaw"})

    assert config.feature_columns == ("x", "y", "vx", "vy", "yaw")
    assert config.feature_dim == 5
    assert config.scene_schema.name == "custom: vx:vy:yaw"


def test_writer_config_custom_schema_ordering() -> None:
    """Confirm that the internal representation maintains consistent field ordering."""
    base = {"precision": "float64", "offset_positions": False}
    config = WriterConfig.model_validate({**base, "scene_schema": "yaw:vx:vy"})

    assert config.feature_columns == ("x", "y", "vx", "vy", "yaw")
    assert config.feature_dim == 5
    assert config.scene_schema.name == "custom: yaw:vx:vy"


def test_writer_config_invalid_schema_validation() -> None:
    """Raise ValidationError when required base fields are missing from a custom schema."""
    base = {"precision": "float64", "offset_positions": False}
    with pytest.raises(ValidationError, match=r"must include the base fields"):
        WriterConfig.model_validate({
            **base,
            "scene_schema": {
                "name": "custom_schema",
                "fields": ["x", "y", "ax"],
            },
        })

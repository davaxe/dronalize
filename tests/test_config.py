# pyright: standard

from pathlib import Path

import pytest
from pydantic import ValidationError

from dronalize.categories import AgentCategory
from dronalize.config import Config, ConfigOverrides, load_config_overrides, resolve_runtime_config
from dronalize.config.filtering import FilteringConfig
from dronalize.config.loader import LoaderConfig
from dronalize.config.map import (
    MapConfig,
    NoneExtraction,
    RadialExtraction,
    RectangularExtraction,
    SquareExtraction,
)
from dronalize.config.writer import WriterConfig
from dronalize.pipeline.functional.resample import ResampleSpec
from dronalize.scene import (
    CANONICAL_V1,
    POSITIONS_ONLY_V1,
    POSITIONS_VELOCITY_ACCELERATION_V1,
    POSITIONS_VELOCITY_YAW_V1,
)


def test_radial_extraction_valid() -> None:
    """Initialize RadialExtraction with a valid radius."""
    extraction = RadialExtraction(radius=10.5)

    assert extraction.mode == "radial"
    assert extraction.radius == pytest.approx(10.5)


def test_rectangular_extraction_valid() -> None:
    """Initialize RectangularExtraction with valid dimensions."""
    extraction = RectangularExtraction(width=5.0, height=8.0)

    assert extraction.mode == "rectangular"
    assert extraction.width == pytest.approx(5.0)
    assert extraction.height == pytest.approx(8.0)


def test_map_config_default_initialization() -> None:
    """Generate a default MapConfig and verify default values."""
    config = MapConfig.default()

    assert config.min_distance == pytest.approx(1.75)
    assert config.interp_distance == pytest.approx(3.0)
    assert config.include_map is True
    assert isinstance(config.extraction, NoneExtraction)
    assert config.extraction.mode == "none"


def test_map_config_discriminator_radial() -> None:
    """Parse a dictionary with radial mode to ensure proper union routing."""
    config_dict = {
        "extraction": {
            "mode": "radial",
            "radius": 15.0,
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, RadialExtraction)
    assert config.extraction.radius == pytest.approx(15.0)


def test_map_config_discriminator_rectangular() -> None:
    """Parse a dictionary with rectangular mode to ensure proper union routing."""
    config_dict = {
        "extraction": {
            "mode": "rectangular",
            "width": 20.0,
            "height": 10.0,
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, RectangularExtraction)
    assert config.extraction.width == pytest.approx(20.0)
    assert config.extraction.height == pytest.approx(10.0)


def test_map_config_missing_discriminator_params() -> None:
    """Raise ValidationError when required parameters for a specific mode are missing."""
    config_dict = {
        "extraction": {
            "mode": "radial",
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
            "mode": "radial",
            "radius": 5.0,
            "width": 10.0,
        }
    }
    config = MapConfig.model_validate(config_dict)
    assert isinstance(config.extraction, RadialExtraction)
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


def test_filtering_config_single_agent_category_string() -> None:
    """Parse a single string into a frozenset of AgentCategory."""
    config = FilteringConfig.create(exclude_agent_categories="CAR")

    assert config.exclude_agent_categories == frozenset([AgentCategory.CAR])


def test_filtering_config_list_agent_category_strings() -> None:
    """Parse a list of strings into a frozenset of AgentCategory."""
    config = FilteringConfig.create(exclude_agent_categories=["CAR", "PEDESTRIAN"])

    assert config.exclude_agent_categories == frozenset([
        AgentCategory.CAR,
        AgentCategory.PEDESTRIAN,
    ])


def test_filtering_config_existing_enum_member() -> None:
    """Accept an existing Enum member directly and coerce it into a frozenset."""
    config = FilteringConfig.create(exclude_agent_categories=AgentCategory.VAN)

    assert config.exclude_agent_categories == frozenset([AgentCategory.VAN])


def test_filtering_config_invalid_agent_category() -> None:
    """Raise ValidationError when an invalid agent category string is provided."""
    with pytest.raises(ValidationError) as exc_info:
        FilteringConfig.create(exclude_agent_categories=["CAR", "SPACESHIP"])

    error_msg = str(exc_info.value)
    assert "SPACESHIP" in error_msg


def test_filtering_config_single_frame_int() -> None:
    """Parse a single integer into a frozenset of integers for require_frames."""
    config = FilteringConfig.create(require_frames=19)

    assert config.require_frames == frozenset([19])


def test_filtering_config_list_frames() -> None:
    """Parse a list of integers into a frozenset of integers for require_frames."""
    config = FilteringConfig.create(require_frames=[19, 20, 21])

    assert config.require_frames == frozenset([19, 20, 21])


def test_filtering_config_none_values() -> None:
    """Retain None values when no explicit frames or categories are provided."""
    config = FilteringConfig()

    assert config.exclude_agent_categories is None
    assert config.require_frames is None


def test_filtering_config_full_dict_validation() -> None:
    """Validate a complete dictionary matching a TOML configuration block."""
    raw_data = {
        "min_agents": 3,
        "require_all_valid": True,
        "require_frames": [10, 15],
        "exclude_agent_categories": "STATIC_OBJECT",
        "filter_slow_agents": 1.5,
        "min_samples_per_agent": 5,
    }

    config = FilteringConfig.model_validate(raw_data)

    assert config.min_agents == 3
    assert config.require_all_valid is True
    assert config.require_frames == frozenset([10, 15])
    assert config.exclude_agent_categories == frozenset([AgentCategory.STATIC_OBJECT])
    assert config.filter_slow_agents == pytest.approx(1.5)
    assert config.min_samples_per_agent == 5


def test_loader_config_with_filtering_normalizes_negative_require_frames() -> None:
    """Convert valid negative frame indices into offsets from the sequence end."""
    config = LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filtering(
        require_frames=[0, -1, -2]
    )

    assert config.filtering is not None
    assert config.filtering.require_frames == frozenset([0, 3, 4])


def test_loader_config_with_filtering_rejects_out_of_range_negative_frame() -> None:
    """Keep the original invalid frame value in the error for easier debugging."""
    with pytest.raises(ValueError, match=r"Invalid frame index: -6"):
        LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filtering(require_frames=[-6])


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


def test_square_extraction_valid() -> None:
    """Initialize SquareExtraction and verify computed dimensions."""
    extraction = SquareExtraction(size=10.0)

    assert extraction.mode == "square"
    assert extraction.size == pytest.approx(10.0)
    assert extraction.width == pytest.approx(10.0)
    assert extraction.height == pytest.approx(10.0)


def test_square_extraction_model_dump() -> None:
    """Serialize SquareExtraction to verify computed fields are included in output."""
    extraction = SquareExtraction(size=25.0)
    dumped = extraction.model_dump()

    assert dumped["mode"] == "square"
    assert dumped["size"] == pytest.approx(25.0)


def test_map_config_discriminator_square() -> None:
    """Parse a dictionary with square mode to ensure proper union routing."""
    config_dict = {
        "extraction": {
            "mode": "square",
            "size": 15.0,
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, SquareExtraction)
    assert config.extraction.size == pytest.approx(15.0)
    assert config.extraction.width == pytest.approx(15.0)
    assert config.extraction.height == pytest.approx(15.0)


@pytest.mark.parametrize("mode_alias", ["circular", "radius", "circle"])
def test_radial_extraction_aliases(mode_alias: str) -> None:
    """Ensure radial extraction accepts all defined string aliases."""
    config_dict = {
        "extraction": {
            "mode": mode_alias,
            "radius": 12.5,
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, RadialExtraction)
    # The mode string is preserved as the alias the user provided
    assert config.extraction.mode == mode_alias
    assert config.extraction.radius == pytest.approx(12.5)


@pytest.mark.parametrize("mode_alias", ["box", "rectangle"])
def test_rectangular_extraction_aliases(mode_alias: str) -> None:
    """Ensure rectangular extraction accepts all defined string aliases."""
    config_dict = {
        "extraction": {
            "mode": mode_alias,
            "width": 20.0,
            "height": 10.0,
        }
    }

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, RectangularExtraction)
    # The mode string is preserved as the alias the user provided
    assert config.extraction.mode == mode_alias
    assert config.extraction.width == pytest.approx(20.0)
    assert config.extraction.height == pytest.approx(10.0)


def test_square_extraction_missing_size() -> None:
    """Raise ValidationError when the size parameter is missing for square extraction."""
    config_dict = {
        "extraction": {
            "mode": "square",
            # missing "size"
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        MapConfig.model_validate(config_dict)

    assert "Field required" in str(exc_info.value)
    assert "size" in str(exc_info.value)


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

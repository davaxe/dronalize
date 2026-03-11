import pytest
from pydantic import ValidationError

from dronalize.config.filtering import FilteringConfig

# Adjust the import path based on your project structure
from dronalize.config.map import (
    MapConfig,
    NoneExtraction,
    RadialExtraction,
    RectangularExtraction,
    SquareExtraction,
)
from dronalize.core.categories import AgentCategory


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
    config = FilteringConfig.create(filter_agent_category="CAR")

    assert config.filter_agent_category == frozenset([AgentCategory.CAR])


def test_filtering_config_list_agent_category_strings() -> None:
    """Parse a list of strings into a frozenset of AgentCategory."""
    config = FilteringConfig.create(filter_agent_category=["CAR", "PEDESTRIAN"])

    assert config.filter_agent_category == frozenset([AgentCategory.CAR, AgentCategory.PEDESTRIAN])


def test_filtering_config_existing_enum_member() -> None:
    """Accept an existing Enum member directly and coerce it into a frozenset."""
    config = FilteringConfig.create(filter_agent_category=AgentCategory.VAN)

    assert config.filter_agent_category == frozenset([AgentCategory.VAN])


def test_filtering_config_invalid_agent_category() -> None:
    """Raise ValidationError when an invalid agent category string is provided."""
    with pytest.raises(ValidationError) as exc_info:
        FilteringConfig.create(filter_agent_category=["CAR", "SPACESHIP"])

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

    assert config.filter_agent_category is None
    assert config.require_frames is None


def test_filtering_config_full_dict_validation() -> None:
    """Validate a complete dictionary matching a TOML configuration block."""
    raw_data = {
        "min_agents": 3,
        "require_all_valid": True,
        "require_frames": [10, 15],
        "filter_agent_category": "STATIC_OBJECT",
        "filter_slow_agents": 1.5,
        "min_samples_per_agent": 5,
    }

    config = FilteringConfig.model_validate(raw_data)

    assert config.min_agents == 3
    assert config.require_all_valid is True
    assert config.require_frames == frozenset([10, 15])
    assert config.filter_agent_category == frozenset([AgentCategory.STATIC_OBJECT])
    assert config.filter_slow_agents == pytest.approx(1.5)
    assert config.min_samples_per_agent == 5


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
    assert dumped["width"] == pytest.approx(25.0)
    assert dumped["height"] == pytest.approx(25.0)


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

# pyright: standard

from pathlib import Path

import pytest
from pydantic import ValidationError

from dronalize.core.categories import AgentCategory
from dronalize.core.models import Range
from dronalize.core.scene import (
    CANONICAL,
    POSITIONS_ONLY,
    POSITIONS_VELOCITY_ACCELERATION,
    POSITIONS_VELOCITY_YAW,
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
from dronalize.runtime import (
    ConfigFile,
    ConfigResolver,
    FileDatasetConfig,
    ResolvedConfig,
    load_project_config,
    resolve_runtime_config,
)

DEFAULT_LOADER_CONFIG = LoaderConfig(input_len=3, output_len=2, sample_time=0.1)


def _runtime_config(*, loader: LoaderConfig | None = None) -> ResolvedConfig:
    return ResolvedConfig(loader=loader or DEFAULT_LOADER_CONFIG, map=MapConfig.default())


def _file_config(data: object) -> FileDatasetConfig:
    return FileDatasetConfig.model_validate(data)


def _write_project_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def _load_dataset_overrides(
    tmp_path: Path,
    content: str,
    *,
    dataset_name: str = "a43",
) -> FileDatasetConfig:
    return load_project_config(_write_project_config(tmp_path, content)).dataset_config(
        dataset_name
    )


def test_map_config_defaults() -> None:
    """Generate a default MapConfig and verify default values."""
    config = MapConfig.default()

    assert config.min_distance == pytest.approx(1.75)
    assert config.interp_distance == pytest.approx(3.0)
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


def test_filter_define_cleanup_wraps_agent_rules() -> None:
    """Agent rules passed to cleanup helpers should be wrapped as pruning rules."""
    scene_filter = Filter.define_cleanup(agent.MinSamples(minimum=5, rule_id="sample_floor"))

    assert scene_filter.cleanup_rules == (
        cleanup.PruneByRule(
            rule=agent.MinSamples(minimum=5, rule_id="sample_floor"),
            rule_id="sample_floor",
        ),
    )


def test_filter_spec_parses_basic_rules() -> None:
    """Config-facing filter specs should parse discriminated cleanup and check rules."""
    spec = FilterSpec.model_validate({
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
    default = _runtime_config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filter(
            Filter.define(
                cleanup_rules=[cleanup.ExcludeCategories.define(categories=["CAR"])],
                scene_rules=[scene.MinimumAgents(minimum=2)],
                agent_rules=[agent.RequireFrames.define(frames=[0])],
            )
        )
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides=_file_config({"execution": {"jobs": 8}}),
    )

    assert resolved.loader.filter is not None
    assert resolved.loader.filter == default.loader.filter
    assert resolved.execution.jobs == 8
    assert resolved.execution.parallel is True


def test_runtime_config_merges_filters() -> None:
    """Filter overrides should extend existing rules and support explicit removals."""
    default = _runtime_config(
        loader=LoaderConfig(input_len=3, output_len=2, sample_time=0.1).with_filter(
            Filter.define(
                cleanup_rules=[cleanup.ExcludeCategories.define(categories=["CAR"])],
                scene_rules=[scene.MinimumAgents(minimum=2)],
                agent_rules=[
                    agent.RequireFrames.define(frames=[0], rule_id="obs_anchor"),
                    agent.RequireFrames.define(frames=[4], rule_id="pred_anchor"),
                ],
            )
        )
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides=_file_config({
            "loader": {
                "filter": {
                    "mode": "extend",
                    "agent": [
                        {"type": "frames", "frames": [1], "rule_id": "obs_anchor"},
                        {"type": "min_samples", "minimum": 3},
                    ],
                    "remove": ["pred_anchor"],
                }
            }
        }),
    )
    assert resolved.loader.filter == Filter.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=["CAR"])],
        scene_rules=[scene.MinimumAgents(minimum=2)],
        agent_rules=[
            agent.RequireFrames.define(frames=[1], rule_id="obs_anchor"),
            agent.MinSamples(minimum=3),
        ],
    )


def test_runtime_config_deep_merge() -> None:
    """Deep merges should preserve unspecified nested fields from the default config."""
    default = _runtime_config(
        loader=DEFAULT_LOADER_CONFIG.with_resampling(ResampleSpec(up=2, down=1))
    )

    resolved = resolve_runtime_config(
        default=default,
        overrides=_file_config({
            "execution": {"jobs": 8},
            "loader": {"window": {"size": 5, "step": 1}},
        }),
    )

    assert resolved.execution.jobs == 8
    assert resolved.execution.parallel is True
    assert resolved.loader.window is not None
    assert resolved.loader.window.size == 5
    assert resolved.loader.resampling == default.loader.resampling


def test_writer_schema_drives_layout() -> None:
    """Writer config should derive schema and feature layout from the selected schema."""
    default_config = WriterConfig()
    positions_only = WriterConfig.create(scene_schema="positions_only")
    positions_velocity_acceleration = WriterConfig.create(
        scene_schema="positions_velocity_acceleration"
    )
    positions_velocity_yaw = WriterConfig.create(scene_schema="positions_velocity_yaw")

    assert default_config.scene_schema == CANONICAL
    assert default_config.feature_dim == 7
    assert default_config.feature_columns == ("x", "y", "vx", "vy", "ax", "ay", "yaw")

    assert positions_only.scene_schema == POSITIONS_ONLY
    assert positions_only.feature_dim == 2
    assert positions_only.feature_columns == ("x", "y")

    assert positions_velocity_acceleration.scene_schema == POSITIONS_VELOCITY_ACCELERATION
    assert positions_velocity_acceleration.feature_dim == 6
    assert positions_velocity_acceleration.feature_columns == ("x", "y", "vx", "vy", "ax", "ay")

    assert positions_velocity_yaw.scene_schema == POSITIONS_VELOCITY_YAW
    assert positions_velocity_yaw.feature_dim == 5
    assert positions_velocity_yaw.feature_columns == ("x", "y", "vx", "vy", "yaw")


def test_runtime_config_merges_writer() -> None:
    """Writer overrides should merge into the default writer config."""
    default = _runtime_config()

    resolved = resolve_runtime_config(
        default=default,
        overrides=_file_config({"writer": {"schema": "positions_only", "precision": "float64"}}),
    )

    assert resolved.writer.scene_schema == POSITIONS_ONLY
    assert resolved.writer.precision == "float64"
    assert resolved.writer.offset_positions is True


def test_runtime_config_merges_split() -> None:
    """Split overrides should validate and merge through the top-level config model."""
    default = _runtime_config()

    resolved = resolve_runtime_config(
        default=default,
        overrides=_file_config({
            "split": {
                "mode": "scene",
                "ratio": {"train": 0.7, "val": 0.2, "test": 0.1},
            }
        }),
    )

    assert resolved.split == SplitConfig(
        mode=BySceneSplit(), ratio=SplitWeights(train=0.7, val=0.2, test=0.1)
    )


def test_load_overrides_preserves_global_and_dataset_sections(tmp_path: Path) -> None:
    """Project config loading should keep global and dataset sections distinct and typed."""
    overrides = load_project_config(
        _write_project_config(
            tmp_path,
            """[global.execution]
jobs = 4

[datasets.a43.execution]
jobs = 1

[datasets.waymo.execution]
chunksize = 8
""",
        )
    )

    assert isinstance(overrides, ConfigFile)
    assert overrides.global_ == _file_config({"execution": {"jobs": 4}})
    assert overrides.dataset_config("a43") == _file_config({"execution": {"jobs": 1}})
    assert overrides.dataset_config("waymo") == _file_config({"execution": {"chunksize": 8}})
    assert overrides.dataset_config("nuscenes") == FileDatasetConfig()


def test_resolver_applies_global_then_dataset_overrides(tmp_path: Path) -> None:
    """The central resolver should apply global overrides before dataset-specific ones."""
    config_path = _write_project_config(
        tmp_path,
        """[global.execution]
jobs = 4

[datasets.a43.execution]
jobs = 1
""",
    )

    default = _runtime_config()
    overrides = load_project_config(config_path)
    resolver = ConfigResolver()

    resolved = resolver.resolve_from_defaults(default=default, overrides=overrides.global_)
    resolved = resolver.resolve_from_defaults(
        default=resolved, overrides=overrides.dataset_config("a43")
    )

    assert resolved.execution.jobs == 1
    assert resolved.execution.parallel is False


def test_load_overrides_filter_specs(tmp_path: Path) -> None:
    """TOML loader filter tables should resolve into runtime filter objects."""
    overrides = _load_dataset_overrides(
        tmp_path,
        """[datasets.a43.loader.filter]
mode = "replace"

[[datasets.a43.loader.filter.cleanup]]
type = "exclude"
categories = ["CAR"]

[[datasets.a43.loader.filter.scene]]
type = "min_agents"
minimum = 3

[[datasets.a43.loader.filter.agent]]
type = "frames"
frames = [0, 4]
tolerance = { kind = "combined", absolute = 1, relative = 0.2 }
""",
    )
    resolved = resolve_runtime_config(
        default=_runtime_config(),
        overrides=overrides,
    )
    assert resolved.loader.filter == Filter.define(
        cleanup_rules=[cleanup.ExcludeCategories.define(categories=["CAR"])],
        scene_rules=[scene.MinimumAgents(minimum=3)],
        agent_rules=[
            agent.RequireFrames.define(frames=[0, 4], tolerance=tol(absolute=1, relative=0.2))
        ],
    )


def test_load_overrides_prune_by_rule_cleanup(tmp_path: Path) -> None:
    """TOML cleanup tables should parse prune-by-rule definitions."""
    overrides = _load_dataset_overrides(
        tmp_path,
        """[datasets.a43.loader.filter]
mode = "replace"

[[datasets.a43.loader.filter.cleanup]]
type = "prune_by_rule"
rule = { type = "min_samples", minimum = 8, selector = { mode = "include", categories = ["CAR"] }, rule_id = "car_min_samples" }
""",  # noqa: E501
    )
    resolved = resolve_runtime_config(
        default=_runtime_config(),
        overrides=overrides,
    )
    assert resolved.loader.filter == Filter.define(
        cleanup_rules=[
            cleanup.PruneByRule(
                rule=agent.MinSamples(
                    minimum=8,
                    selector=AgentSelector.include(["CAR"]),
                    rule_id="car_min_samples",
                )
            )
        ]
    )


def test_load_overrides_prune_by_rule_cleanup_verbose_toml(tmp_path: Path) -> None:
    """Expanded TOML tables should also parse prune-by-rule cleanup definitions."""
    overrides = _load_dataset_overrides(
        tmp_path,
        """[datasets.a43.loader.filter]
mode = "replace"

[[datasets.a43.loader.filter.cleanup]]
type = "prune_by_rule"

[datasets.a43.loader.filter.cleanup.rule]
type = "min_samples"
minimum = 8
rule_id = "car_min_samples"

[datasets.a43.loader.filter.cleanup.rule.selector]
mode = "include"
categories = ["CAR"]
""",
    )
    resolved = resolve_runtime_config(
        default=_runtime_config(),
        overrides=overrides,
    )
    assert resolved.loader.filter == Filter.define(
        cleanup_rules=[
            cleanup.PruneByRule(
                rule=agent.MinSamples(
                    minimum=8,
                    selector=AgentSelector.include(["CAR"]),
                    rule_id="car_min_samples",
                )
            )
        ]
    )


def test_load_overrides_extended_filters(tmp_path: Path) -> None:
    """Filter file overrides should extend defaults and remove named rules explicitly."""
    overrides = _load_dataset_overrides(
        tmp_path,
        """[datasets.a43.loader.filter]
mode = "extend"
remove = ["pred_anchor"]

[[datasets.a43.loader.filter.scene]]
type = "window"
start_frame = 0
end_frame = 4
min_fraction = 0.6
rule_id = "obs_window"

[[datasets.a43.loader.filter.agent]]
type = "window"
start_frame = 0
end_frame = 4
min_fraction = 0.8
tolerance = { kind = "combined", absolute = 1, "relative" = 0.5 }
rule_id = "dynamic_window"

[datasets.a43.loader.filter.agent.selector]
mode = "exclude"
categories = ["PEDESTRIAN"]
""",
    )
    resolved = resolve_runtime_config(
        default=_runtime_config(
            loader=DEFAULT_LOADER_CONFIG.with_filter(
                Filter.define(
                    scene_rules=[scene.MinimumAgents(minimum=2, rule_id="min_cars")],
                    agent_rules=[agent.RequireFrames.define(frames=[4], rule_id="pred_anchor")],
                )
            )
        ),
        overrides=overrides,
    )
    assert resolved.loader.filter == Filter.define(
        scene_rules=[
            scene.MinimumAgents(minimum=2, rule_id="min_cars"),
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


def test_load_overrides_rejects_unknown_nested_keys(tmp_path: Path) -> None:
    """Typed config loading should reject stale nested override schemas immediately."""
    config_path = _write_project_config(
        tmp_path,
        """[datasets.a43.loader.filter.extend]

[[datasets.a43.loader.filter.extend.validate_scene]]
type = "min_agents"
minimum = 2
""",
    )

    with pytest.raises(ValidationError):
        load_project_config(config_path)


def test_load_overrides_rejects_top_level_dataset_sections(tmp_path: Path) -> None:
    """Top-level dataset sections should no longer be accepted in TOML."""
    config_path = _write_project_config(
        tmp_path,
        """[a43.execution]
jobs = 4
""",
    )

    with pytest.raises(ValidationError):
        load_project_config(config_path)


def test_load_overrides_translates_flat_split_and_map_sections(tmp_path: Path) -> None:
    """Flat TOML split and map sections should load into the authoring config shape."""
    overrides = _load_dataset_overrides(
        tmp_path,
        """[datasets.a43.map]
enabled = true
min_distance = 1.0
interp_distance = 2.5
extraction = "circle"
radius = 60.0

[datasets.a43.split]
mode = "shuffled-time"
ratio = { train = 0.7, val = 0.2, test = 0.1 }
segments = 8
gap = 2
""",
    )

    assert (
        overrides.map
        == _file_config({
            "map": {
                "enabled": True,
                "min_distance": 1.0,
                "interp_distance": 2.5,
                "extraction": "circle",
                "radius": 60.0,
            }
        }).map
    )
    assert (
        overrides.split
        == _file_config({
            "split": {
                "mode": "shuffled-time",
                "ratio": {"train": 0.7, "val": 0.2, "test": 0.1},
                "segments": 8,
                "gap": 2,
            }
        }).split
    )


def test_load_overrides_translates_resampling_derivative_entries(tmp_path: Path) -> None:
    """Array-of-table derivative entries should load into the authoring resampling model."""
    overrides = _load_dataset_overrides(
        tmp_path,
        """[datasets.a43.loader.resampling]
up = 2
down = 1

[[datasets.a43.loader.resampling.output_derivatives]]
order = 1
columns = ["vx", "vy"]

[[datasets.a43.loader.resampling.output_derivatives]]
order = 2
columns = ["ax", "ay"]
""",
    )

    assert overrides.loader is not None
    assert overrides.loader.resampling is not None
    assert overrides.loader.resampling.output_derivative_map() == {
        1: ["vx", "vy"],
        2: ["ax", "ay"],
    }


def test_map_config_parses_relevant_area() -> None:
    """Parse a dictionary with relevant mode to ensure proper union routing."""
    config_dict = {"extraction": {"mode": "relevant", "padding": 1.3}}

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, RelevantAreaExtraction)
    assert config.extraction.mode == "relevant"
    assert config.extraction.padding == pytest.approx(1.3)


def test_map_config_parses_full_map() -> None:
    """Parse a dictionary with full mode to ensure proper union routing."""
    config_dict = {"extraction": {"mode": "full"}}

    config = MapConfig.model_validate(config_dict)

    assert isinstance(config.extraction, FullMapExtraction)
    assert config.extraction.mode == "full"


def test_writer_schema_positions_only() -> None:
    """Verify initialization with the 'positions_only' shorthand string."""
    config = WriterConfig.create("positions_only", precision="float64", offset_positions=False)

    assert config.scene_schema == POSITIONS_ONLY
    assert config.feature_columns == ("x", "y")
    assert config.feature_dim == 2


def test_writer_schema_predefined() -> None:
    """Ensure the config accepts a predefined schema object."""
    config = WriterConfig(
        scene_schema=POSITIONS_VELOCITY_ACCELERATION,
        precision="float64",
        offset_positions=False,
    )

    assert config.scene_schema == POSITIONS_VELOCITY_ACCELERATION
    assert config.feature_columns == ("x", "y", "vx", "vy", "ax", "ay")
    assert config.feature_dim == 6


def test_writer_schema_single_custom() -> None:
    """Check that a single custom field is accepted via structured definition."""
    config = WriterConfig.create(
        {"name": "custom_vx", "fields": ["frame", "id", "x", "y", "vx", "agent_category"]},
        precision="float64",
        offset_positions=False,
    )

    assert config.feature_columns == ("x", "y", "vx")
    assert config.feature_dim == 3
    assert config.scene_schema.name == "custom_vx"


def test_writer_schema_multiple_custom() -> None:
    """Validate that multiple custom fields generate a structured custom schema."""
    config = WriterConfig.create(
        {
            "name": "custom_velocity_yaw",
            "fields": ["frame", "id", "x", "y", "vx", "vy", "yaw", "agent_category"],
        },
        precision="float64",
        offset_positions=False,
    )

    assert config.feature_columns == ("x", "y", "vx", "vy", "yaw")
    assert config.feature_dim == 5
    assert config.scene_schema.name == "custom_velocity_yaw"


def test_writer_schema_custom_order() -> None:
    """Confirm that structured custom schemas still use canonical field ordering."""
    config = WriterConfig.create(
        {
            "name": "custom_yaw_velocity",
            "fields": ["frame", "id", "x", "y", "yaw", "vx", "vy", "agent_category"],
        },
        precision="float64",
        offset_positions=False,
    )

    assert config.feature_columns == ("x", "y", "vx", "vy", "yaw")
    assert config.feature_dim == 5
    assert config.scene_schema.name == "custom_yaw_velocity"


def test_writer_schema_rejects_string_shorthand() -> None:
    """Raise ValueError when using removed colon-separated schema shorthand."""
    with pytest.raises(ValueError, match=r"Unknown scene schema 'vx:vy:yaw'"):
        WriterConfig.create("vx:vy:yaw", precision="float64", offset_positions=False)


def test_writer_schema_rejects_invalid() -> None:
    """Raise ValueError when required base fields are missing from a custom schema."""
    with pytest.raises(ValueError, match=r"must include the base fields"):
        WriterConfig.create(
            {"name": "custom_schema", "fields": ["x", "y", "ax"]},
            precision="float64",
            offset_positions=False,
        )

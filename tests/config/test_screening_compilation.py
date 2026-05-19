# pyright: standard
from __future__ import annotations

import polars as pl
import pytest

from dronalize.config.models import ScreeningConfig
from dronalize.core import AgentCategory
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.screening import ScreeningRuleSet, agent, cleanup, scene, screen_scene


def test_agent_specs_compile_into_runtime_agent_rules() -> None:
    config = ScreeningConfig.model_validate({
        "agent": {
            "required_frames": {"rule": "frames", "frames": [0, 1, 2]},
            "coverage": {"rule": "window", "start_frame": 0, "end_frame": 2, "min_fraction": 0.5},
            "gap_budget": {
                "rule": "max_gap",
                "maximum": 1,
                "require": {"absolute": 2, "relative": 0.75},
            },
        }
    })

    compiled = ScreeningRuleSet.from_config(config)

    assert len(compiled.agent_rules) == 3
    assert isinstance(compiled.agent_rules[0], agent.AgentRequireFrames)
    assert isinstance(compiled.agent_rules[1], agent.AgentRequireWindow)
    assert isinstance(compiled.agent_rules[2], agent.MaxGap)
    assert compiled.agent_rules[0].rule_id == "required_frames"
    assert compiled.agent_rules[1].rule_id == "coverage"
    assert compiled.agent_rules[2].rule_id == "gap_budget"
    assert compiled.agent_rules[2].require is not None
    assert compiled.agent_rules[2].require.absolute == 2
    assert compiled.agent_rules[2].require.relative == pytest.approx(0.75)


def test_scene_specs_compile_into_runtime_scene_rules() -> None:
    config = ScreeningConfig.model_validate({
        "scene": {"agent_bounds": {"rule": "agent_range", "minimum": 1, "maximum": 5}}
    })

    compiled = ScreeningRuleSet.from_config(config)

    assert len(compiled.scene_rules) == 1
    assert isinstance(compiled.scene_rules[0], scene.AgentRange)
    assert compiled.scene_rules[0].rule_id == "agent_bounds"


def test_cleanup_specs_compile_and_wrap_nested_agent_rules() -> None:
    config = ScreeningConfig.model_validate({
        "cleanup": {
            "keep_only_cars": {"rule": "include", "categories": ["car"]},
            "prune_sparse": {
                "rule": "prune_by",
                "agent_rule": {"rule": "min_samples", "minimum": 3},
            },
        }
    })

    compiled = ScreeningRuleSet.from_config(config)

    assert len(compiled.cleanup_rules) == 2
    assert isinstance(compiled.cleanup_rules[0], cleanup.IncludeCategories)
    assert isinstance(compiled.cleanup_rules[1], cleanup.PruneByRule)
    assert isinstance(compiled.cleanup_rules[1].agent_rule, agent.MinSamples)
    assert compiled.cleanup_rules[0].rule_id == "keep_only_cars"
    assert compiled.cleanup_rules[1].rule_id == "prune_sparse"


def test_cleanup_prune_by_rejects_nested_agent_require() -> None:
    with pytest.raises(ValueError, match="require"):
        ScreeningConfig.model_validate({
            "cleanup": {
                "prune_sparse": {
                    "rule": "prune_by",
                    "agent_rule": {"rule": "min_samples", "minimum": 3, "require": {"absolute": 1}},
                }
            }
        })


def test_agent_require_must_define_a_threshold() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ScreeningConfig.model_validate({
            "agent": {"sample_floor": {"rule": "min_samples", "minimum": 3, "require": {}}}
        })


def test_compiled_screen_rules_are_directly_usable_in_screen_scene() -> None:
    config = ScreeningConfig.model_validate({
        "cleanup": {"drop_unimportant": {"rule": "exclude", "categories": ["unimportant"]}},
        "scene": {"enough_agents": {"rule": "agent_range", "minimum": 1}},
        "agent": {"max_missing": {"rule": "max_missing_frames", "maximum": 1}},
    })
    compiled = ScreeningRuleSet.from_config(config)

    frame = pl.DataFrame({
        "scene": [1, 1, 1, 1, 1, 1, 1],
        "id": [1, 1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 3, 0, 1, 2],
        "agent_category": [AgentCategory.CAR] * 7,
    })

    screened = screen_scene(frame, compiled, scene_group_by="scene", columns=TrajectoryColumns())

    assert len(screened) == len(frame)
    assert screened.columns == frame.columns

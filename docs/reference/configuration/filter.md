
# `[loader.filter]` section

<div class="section-intro" markdown="1">
The filter config is nested under `loader` because it runs during data loading. It is used to remove unwanted rows, reject invalid scenes, and prune invalid agents before samples are written.
</div>

## Filter shape

```toml
[datasets.a43.loader.filter]
mode = "extend"
remove = ["pred_anchor"]

[[datasets.a43.loader.filter.cleanup]]
type = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[[datasets.a43.loader.filter.scene]]
type = "min_agents"
minimum = 2

[[datasets.a43.loader.filter.agent]]
type = "min_samples"
minimum = 8
rule_id = "sample_floor"
```

## Agent categories

The following agent categories are supported for filtering and selection. These categories are used in the `categories` field of selectors and cleanup rules. The string representation or corresponding integer values can be used in the configuration. The string representation is case insensitive.
See [dronalize.core.AgentCategory](../api/core/enums/agent-category.md) for enum details.


| Enum | String representation | Integer value | Description |
|---|---|---:|---|
| `AgentCategory.ANIMAL` | `"ANIMAL"` | `1` | Animal present in the scene. |
| `AgentCategory.BICYCLE` | `"BICYCLE"` | `2` | Non-motorized two-wheeled bicycle. |
| `AgentCategory.BUS` | `"BUS"` | `3` | Bus designed for passenger transport. |
| `AgentCategory.CAR` | `"CAR"` | `4` | Standard passenger car. |
| `AgentCategory.EMERGENCY_VEHICLE` | `"EMERGENCY_VEHICLE"` | `5` | Emergency-response vehicle, such as a police car, ambulance, or fire truck. |
| `AgentCategory.MOTORCYCLE` | `"MOTORCYCLE"` | `6` | Motorized two-wheeled vehicle. |
| `AgentCategory.MOVEABLE_OBJECT` | `"MOVEABLE_OBJECT"` | `7` | Object that is not currently fixed and may be moved, such as debris or a cart. |
| `AgentCategory.PEDESTRIAN` | `"PEDESTRIAN"` | `8` | Person moving on foot. |
| `AgentCategory.STATIC_OBJECT` | `"STATIC_OBJECT"` | `9` | Fixed object that does not move, such as a pole or barrier. |
| `AgentCategory.TRAILER` | `"TRAILER"` | `10` | Trailer unit, typically towed by another vehicle. |
| `AgentCategory.TRAM` | `"TRAM"` | `11` | Tram or streetcar operating on rails. |
| `AgentCategory.TRICYCLE` | `"TRICYCLE"` | `12` | Three-wheeled cycle or vehicle. |
| `AgentCategory.TRUCK` | `"TRUCK"` | `13` | Heavy truck or lorry. |
| `AgentCategory.UNIMPORTANT` | `"UNIMPORTANT"` | `14` | Agent considered irrelevant or not important for the current task. |
| `AgentCategory.UNKNOWN` | `"UNKNOWN"` | `15` | Agent exists but its category is not known or cannot be determined reliably. |
| `AgentCategory.VAN` | `"VAN"` | `16` | Van or light commercial van. |

!!! note "Dataset specific categories"
    Not all datasets share the same set of agent categories. In all cases
    additional or different categories have been mapped to the above selection
    as close as possible. Also some datasets do not support a wide range of
    categories and may therefore only include a small subset of the above.

## `[loader.filter]` table

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `"replace"` or `"extend"` | Whether configured rules replace the existing filter or merge into it by rule name. | `required when rules are present` |
| `remove` | `array[str]` | Rule names to remove after merging. | `none` |
| `cleanup` | `array[table]` | Cleanup rules that remove rows before validation. | `none` |
| `scene` | `array[table]` | Scene-level validation rules. | `none` |
| `agent` | `array[table]` | Agent-level validation rules. | `none` |

## Merge behavior

- `replace` discards the inherited filter and uses only the rules defined here.
- `extend` merges rules into the inherited filter by effective rule name.
- `remove` is applied after merging and removes matching rule names.

If a rule has `rule_id`, that id becomes its effective name. Otherwise `dronalize` uses an implicit name such as `scene_min_agents` or `agent_min_samples`.

!!! info "When to use `rule_id`"
    `rule_id` is optional, but necessary if multiple of the same rule type is
    defined. The reason is that most rules add temporary columns in the internal
    Polars data frames, which are named based on their effective rule name. This
    means that if you define two `min_samples` rules without `rule_id`, they
    will both try to write to the same temporary column and cause a conflict. By
    giving them unique `rule_id` values, you can avoid this issue and have more
    control over rule naming.

## Shared rule fields

### `rule_id`

All cleanup, scene, and agent rules may define:

| Key | Type | Description | Default |
|---|---|---|---|
| `rule_id` | `str` | Stable rule name used for merging, diagnostics, and `remove`. Must match `^[a-z0-9_]+$`. | `none` |

### `[loader.filter.<rule>.selector]`

Selectors are supported only on rules that explicitly mention them below. If a rule support selectors,
it will only apply to agents matching the selector. If no selector is provided, the rule applies to all agents.
All agent rules support selectors, and some scene rules do.

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `"include"` or `"exclude"` | Restrict the rule to matching or non-matching categories. | `required` |
| `categories` | `array[str]` | Agent categories included in the selector. | `required` |

### `[loader.filter.agent.tolerance]`

All agent rules may define a tolerance:

| Key | Type | Description | Default |
|---|---|---|---|
| `kind` | `"absolute"`, `"relative"` or `"combined"` | Tolerance type used when agent failures are aggregated at scene level. | `required` |
| `absolute` | `int` | Maximum number of invalid agents to tolerate. | `required for "absolute" and "combined"` |
| `relative` | `float` | Maximum invalid-agent fraction to tolerate. | `required for "relative" and "combined"` |

If no tolerance is set, the default behavior is effectively zero tolerance.

## Cleanup rules

Cleanup rules remove rows before any scene or agent validation runs. Use with
care, as the removed rows will not be considered in any validation and cannot be
recovered later in the pipeline. Cleanup rules are best for removing truly
irrelevant data, for example static objects or unimportant agents that should
not even be considered in the validation.

### `type = "exclude"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"exclude"` | Remove rows whose category is in `categories`. | `required` |
| `categories` | `array[str]` | Categories to remove. | `required` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "include"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"include"` | Keep only rows whose category is in `categories`. | `required` |
| `categories` | `array[str]` | Categories to keep. | `required` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "prune_by_rule"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"prune_by_rule"` | Remove rows corresponding to agents that fail a specific agent rule. | `required` |
| `rule` | `table` | Table definition of agent rule to cleanup. | `required` |

Example:

```toml
[[datasets.a43.loader.filter.cleanup]]
type = "prune_by_rule"
rule = { type = "min_samples", minimum = 8, selector = { mode = "include", categories = ["CAR"] }, rule_id = "car_min_samples" }
```
or equivalent multi-line form:
```toml
[[datasets.a43.loader.filter.cleanup]]
type = "prune_by_rule"

[datasets.a43.loader.filter.cleanup.rule]
type = "min_samples"
minimum = 8
rule_id = "car_min_samples"

[datasets.a43.loader.filter.cleanup.rule.selector]
mode = "include"
categories = ["CAR"]
```

## Scene rules

Scene rules decide whether a scene window is kept or rejected as a whole.

### `type = "min_agents"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"min_agents"` | Require at least a minimum number of retained agents. | `required` |
| `minimum` | `int` | Minimum retained-agent count. | `1` |
| `selector` | `table` | Optional category selector. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "agent_range"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"agent_range"` | Require the retained-agent count to stay within a range. | `required` |
| `minimum` | `int` | Optional minimum retained-agent count. | `none` |
| `maximum` | `int` | Optional maximum retained-agent count. | `none` |
| `selector` | `table` | Optional category selector. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

At least one of `minimum` or `maximum` should be set. If both are set, `maximum` must be greater than or equal to `minimum`.

### `type = "category_range"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"category_range"` | Require category-specific retained-agent count ranges. | `required` |
| `ranges` | `table` | Mapping from category name to `{ minimum, maximum }`. | `required` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

Example:

```toml
[[datasets.a43.loader.filter.scene]]
type = "category_range"
ranges = { CAR = { minimum = 1, maximum = 2 }, PEDESTRIAN = { minimum = 1 } }
```

### `type = "frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"frames"` | Require specific relative frames to exist in the scene window. | `required` |
| `frames` | `array[int]` | Relative frame indices that must be present. | `required` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "window"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"window"` | Require coverage across a relative frame interval. | `required` |
| `start_frame` | `int` | First relative frame in the interval. | `required` |
| `end_frame` | `int` | Last relative frame in the interval. | `required` |
| `min_fraction` | `float` | Minimum required frame coverage fraction. | `1.0` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

`end_frame` must be greater than or equal to `start_frame`.

### `type = "max_missing_frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"max_missing_frames"` | Limit missing frames across the scene window. | `required` |
| `max_missing_frames` | `int` | Maximum allowed number of missing frames. | `0` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

## Agent rules

Agent rules validate individual agents inside an otherwise valid scene. Every agent rule may also define:

- `selector`
- `tolerance`
- `rule_id`

### `type = "frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"frames"` | Require specific relative frames for each retained agent. | `required` |
| `frames` | `array[int]` | Relative frame indices that must be present per agent. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "window"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"window"` | Require coverage across a relative frame interval per agent. | `required` |
| `start_frame` | `int` | First relative frame in the interval. | `required` |
| `end_frame` | `int` | Last relative frame in the interval. | `required` |
| `min_fraction` | `float` | Minimum required frame coverage fraction. | `1.0` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "min_samples"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"min_samples"` | Require a minimum number of samples per agent. | `required` |
| `minimum` | `int` | Minimum number of rows per agent. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "max_missing_frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"max_missing_frames"` | Limit missing frames per agent. | `required` |
| `maximum` | `int` | Maximum allowed number of missing frames. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "max_gap"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"max_gap"` | Limit the largest internal frame gap per agent. | `required` |
| `maximum` | `int` | Maximum allowed internal gap. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "min_consecutive_frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"min_consecutive_frames"` | Require a minimum longest consecutive run per agent. | `required` |
| `minimum` | `int` | Minimum longest consecutive run. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "starts_by_frame"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"starts_by_frame"` | Require the agent to appear by a relative frame. | `required` |
| `frame` | `int` | Latest allowed first frame. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "ends_after_frame"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"ends_after_frame"` | Require the agent to remain through a relative frame. | `required` |
| `frame` | `int` | Earliest allowed last frame. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

### `type = "min_span"`

| Key | Type | Description | Default |
|---|---|---|---|
| `type` | `"min_span"` | Require a minimum span from first to last frame. | `required` |
| `minimum` | `int` | Minimum span in frames. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |
| `rule_id` | `str` | Optional explicit rule name. | `none` |

*[rows]: explicit rows in the tabular data structure. Each row correspond to a singular agent at a specific frame.
*[scene]: collection of agents over a contiguous window of frames. The sample unit that the dataset is split into.

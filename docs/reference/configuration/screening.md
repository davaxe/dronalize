# `[screening]` section

<div class="section-intro" markdown="1">
The screening section defines the policy layer that decides which rows, agents, and scenes are kept after scene extraction. It is authored as three named rule namespaces: `cleanup`, `scene`, and `agent`.
</div>

## Shape

Screening rules are authored as named entries under `cleanup`, `scene`, and `agent`.

```toml
[profiles.basic_screening.screening.cleanup.remove_animals]
rule = "exclude"
categories = ["ANIMAL"]

[datasets.a43.screening]
mode = "extend"
remove = ["remove_animals"]

[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[datasets.a43.screening.scene.min_context]
rule = "agent_range"
minimum = 2

[datasets.a43.screening.agent.anchor_present]
rule = "frames"
frames = [19]
selector = { mode = "include", categories = ["CAR"] }
tolerance = { absolute = 1, relative = 0.05 }
```

The rule name is the final TOML path segment, for example:

- `trim_static` in `[datasets.a43.screening.cleanup.trim_static]`
- `min_context` in `[datasets.a43.screening.scene.min_context]`
- `anchor_present` in `[datasets.a43.screening.agent.anchor_present]`

That rule name is also the name used for merge behavior and removal.

!!! warning "Rule name uniqueness"
    Rule names must be unique within their screening namespace (`cleanup`, `scene`, or `agent`) to avoid merge conflicts. The same rule name can exist in different namespaces.

## Agent categories

The following agent categories are supported in `categories` fields. String
values are case-insensitive, and the corresponding integer values are also
accepted. See [`AgentCategory`][dronalize.core.AgentCategory] for enum details.

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

!!! note "Dataset-specific categories"
    Not all datasets expose the full category set. Dataset-native categories are
    mapped into this common selection as closely as possible.

## `[screening]` table

The parent `screening` table controls merge behavior for the named rules below it.

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `"replace"` or `"extend"` | Whether this screening block replaces inherited rules or merges by rule name. | `"replace"` |
| `remove` | `array[str]` | Named rules to remove after merging. Names may target cleanup, scene, or agent rules. | `none` |

`mode` and `remove` belong on the parent `[...screening]` table, not inside individual rules.

## Merge behavior

- `replace` discards inherited screening rules and uses only the rules defined in the current block.
- `extend` merges the current rules into the inherited screening rules by rule name.
- `remove` is applied after merging.

Merging happens independently across the three namespaces:

- `cleanup`
- `scene`
- `agent`

This means the same rule name can exist once in each namespace, but rule names should still be chosen carefully because they are also used for diagnostics and compiled internal identifiers.

## Shared nested tables

### `selector`

Selectors are supported on all agent rules and on scene rules that explicitly list a `selector` field.

Example:

```toml
[datasets.a43.screening.agent.rule_name.selector]
mode = "include"
categories = ["CAR"]
```

or inline:

```toml
[datasets.a43.screening.agent.rule_name]
... # rule fields
selector = { mode = "include", categories = ["CAR"] }
```

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `"include"` or `"exclude"` | Whether matching categories are kept in scope or excluded from scope. | `"include"` |
| `categories` | `array[str]` | Agent categories included in the selector. | `required` |

### `tolerance`

Agent rules may define a tolerance table. The current config model uses numeric thresholds directly; there is no `kind` field in authored TOML.

Example:

```toml
[datasets.a43.screening.agent.anchor_present.tolerance]
absolute = 1
relative = 0.05
```

| Key | Type | Description | Default |
|---|---|---|---|
| `absolute` | `float` | Maximum number of invalid agents to tolerate. | `none` |
| `relative` | `float` | Maximum invalid-agent fraction to tolerate. | `none` |

!!! note "Tolerance application"
    At least one of `absolute` or `relative` should be set to define a valid
    tolerance. When applied, if the number of invalid agents exceeds the
    `absolute` threshold or the fraction of invalid agents exceeds the `relative`
    threshold, the entire scene is discarded. Otherwise, the scene is retained but
    the invalid agents are still marked as invalid in the output. When both are set
    the stricter of the two thresholds determines whether the scene is retained or discarded.

## Cleanup rules

Cleanup rules remove rows before scene and agent checks run.

They live under:

- `[...screening.cleanup.<rule_name>]`

### `rule = "exclude"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"exclude"` | Remove rows whose category is in `categories`. | `required` |
| `categories` | `array[str]` | Categories to remove. | `required` |

Example:

```toml
[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]
```

### `rule = "include"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"include"` | Keep only rows whose category is in `categories`. | `required` |
| `categories` | `array[str]` | Categories to keep. | `required` |

### `rule = "prune_by"`

The `"prune_by`" rules allows agent-level rules to be applied in a clean-up fashion. For example, if you want to prune short tracks before applying scene-level rules, you can use a nested agent rule under `prune_by` to specify what "short" means.

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"prune_by"` | Remove rows for agents that fail a nested agent rule. | `required` |
| `agent_rule` | `inline table` or nested table | Agent rule used to decide which agents are pruned. | `required` |

For example to prune short car tracks before screening:

```toml
[datasets.a43.screening.cleanup.prune_short_tracks]
rule = "prune_by"
agent_rule = { rule = "min_samples", minimum = 8, selector = { mode = "include", categories = ["CAR"] } }
```

or with a nested table for the agent rule:

```toml
[datasets.a43.screening.cleanup.prune_short_tracks]
rule = "prune_by"

[datasets.a43.screening.cleanup.prune_short_tracks.agent_rule]
rule = "min_samples"
minimum = 8

[datasets.a43.screening.cleanup.prune_short_tracks.agent_rule.selector]
mode = "include"
categories = ["CAR"]
```

## Scene rules

Scene rules decide whether a scene window is retained as a whole.

They live under:

- `[...screening.scene.<rule_name>]`

### `rule = "agent_range"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"agent_range"` | Require the retained-agent count to stay within a range. | `required` |
| `minimum` | `int` | Optional minimum retained-agent count. | `none` |
| `maximum` | `int` | Optional maximum retained-agent count. | `none` |
| `selector` | `table` | Optional category selector. | `none` |

At least one of `minimum` or `maximum` should be set. If both are set, `maximum` must be greater than or equal to `minimum`.

### `rule = "category_range"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"category_range"` | Require category-specific retained-agent count ranges. | `required` |
| `ranges` | `table` | Mapping from category name to `{ minimum, maximum }`. | `required` |

Example:

```toml
[datasets.a43.screening.scene.category_mix]
rule = "category_range"
ranges = { CAR = { minimum = 1, maximum = 2 }, PEDESTRIAN = { minimum = 1 } }
```

### `rule = "scene_frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"scene_frames"` | Require specific relative frames to exist in the scene window. | `required` |
| `frames` | `array[int]` | Relative frame indices that must be present. | `required` |

### `rule = "scene_window"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"scene_window"` | Require coverage across a relative frame interval. | `required` |
| `start_frame` | `int` | First relative frame in the interval. | `required` |
| `end_frame` | `int` | Last relative frame in the interval. | `required` |
| `min_fraction` | `float` | Minimum required frame coverage fraction. | `1.0` |

`end_frame` must be greater than or equal to `start_frame`.

### `rule = "max_missing_frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"max_missing_frames"` | Limit missing frames across the scene window. | `required` |
| `maximum` | `int` | Maximum allowed number of missing frames. | `required` |
| `selector` | `table` | Required category selector in the current config model. | `required` |

## Agent rules

Agent rules validate agents inside a retained scene.

They live under:

- `[...screening.agent.<rule_name>]`

All agent rules may optionally define:

- `selector` to target specific categories for that rule, and
- `tolerance` to allow some tolerance for invalid agents at scene level without discarding the entire scene. This means that if only a few agents fail the rule, it may still be retained depending on the tolerance thresholds.

### `rule = "frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"frames"` | Require specific relative frames for each retained agent. | `required` |
| `frames` | `array[int]` | Relative frame indices that must be present per agent. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "window"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"window"` | Require coverage across a relative frame interval per agent. | `required` |
| `start_frame` | `int` | First relative frame in the interval. | `required` |
| `end_frame` | `int` | Last relative frame in the interval. | `required` |
| `min_fraction` | `float` | Minimum required frame coverage fraction. | `1.0` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "min_samples"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"min_samples"` | Require a minimum number of samples per agent. | `required` |
| `minimum` | `int` | Minimum number of rows per agent. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "max_missing_frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"max_missing_frames"` | Limit missing frames per agent. | `required` |
| `maximum` | `int` | Maximum allowed number of missing frames. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "max_gap"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"max_gap"` | Limit the largest internal frame gap per agent. | `required` |
| `maximum` | `int` | Maximum allowed internal gap. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "min_consecutive_frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"min_consecutive_frames"` | Require a minimum longest consecutive run per agent. | `required` |
| `minimum` | `int` | Minimum longest consecutive run. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "starts_by_frame"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"starts_by_frame"` | Require the agent to appear by a relative frame. | `required` |
| `frame` | `int` | Relative frame. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "ends_after_frame"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"ends_after_frame"` | Require the agent to remain through a relative frame. | `required` |
| `frame` | `int` | Relative frame. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "min_span"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"min_span"` | Require a minimum span from first to last frame. | `required` |
| `minimum` | `int` | Minimum span in frames. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

### `rule = "min_distance"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"min_distance"` | Require a minimum distance traveled across the scene. | `required` |
| `minimum` | `float` | Minimum distance in meters. |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. |

## Practical example

```toml
[profiles.basic_screening.screening.cleanup.remove_animals]
rule = "exclude"
categories = ["ANIMAL"]

[datasets.a43.screening]
mode = "extend"
remove = ["remove_animals"]

[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[datasets.a43.screening.scene.min_context]
rule = "agent_range"
minimum = 2

[datasets.a43.screening.agent.anchor_present]
rule = "frames"
frames = [19]
tolerance = { absolute = 1, relative = 0.05 }
selector = { mode = "include", categories = ["CAR"] }

[datasets.a43.screening.agent.sample_floor]
rule = "min_samples"
minimum = 8
```

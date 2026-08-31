# `[screening]` section

The screening section defines the policy layer that decides which rows, agents, and scenes are kept after scene extraction. It is authored as three named rule namespaces: `cleanup`, `scenes`, and `agents`.

## Shape

Screening rules are authored as named entries under `cleanup`, `scenes`, and `agents`.

```toml
[profiles.basic_screening.screening.cleanup.remove_animals]
rule = "exclude"
categories = ["ANIMAL"]

[datasets.a43.screening.cleanup]
mode = "extend"
remove = ["remove_animals"]

[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[datasets.a43.screening.scenes]
mode = "extend"

[datasets.a43.screening.scenes.min_context]
rule = "agent_range"
minimum = 2

[datasets.a43.screening.agents.anchor_present]
rule = "frames"
frames = [19]
selector = { mode = "include", categories = ["CAR"] }
tolerance = { absolute = 1, relative = 0.05 }
require = { absolute = 3 }
```

The rule name is the final TOML path segment, for example:

- `trim_static` in `[datasets.a43.screening.cleanup.trim_static]`
- `min_context` in `[datasets.a43.screening.scenes.min_context]`
- `anchor_present` in `[datasets.a43.screening.agents.anchor_present]`

That rule name is also the name used for merge behavior and removal.

!!! warning "Rule name uniqueness"
    Rule names must be unique within their screening namespace (`cleanup`, `scenes`, or `agents`) to avoid merge conflicts. The same rule name can exist in different namespaces.

## Agent categories

The following agent categories are supported in `categories` fields. String
values are case-insensitive, and the corresponding integer values are also
accepted. See [`AgentCategory`](../api/core/index.md#prejectory.core.AgentCategory) for enum details.

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

## Namespace tables

Each screening namespace is a mapping patch with its own merge behavior.

| Table | Purpose |
|---|---|
| `[...screening.cleanup]` | Cleanup rules applied before validation. |
| `[...screening.scenes]` | Scene-level acceptance rules. |
| `[...screening.agents]` | Per-agent quality rules. |

Each namespace table accepts the same patch-control keys:

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `"replace"` or `"extend"` | Whether this namespace replaces inherited rules or merges by rule name. | `"extend"` |
| `remove` | `array[str]` | Rule names to remove from this namespace after merging. | `[]` |

`mode` and `remove` are section-local. They do not belong on the parent `[...screening]` table.

## Merge behavior

Screening inheritance is applied independently for `cleanup`, `scenes`, and `agents`.

For each namespace:

1. Start from the inherited named rule map, or from an empty map if nothing is inherited.
2. Apply `mode`.
3. Apply authored rules in the current block.
4. Apply `remove`.

`mode = "extend"` is the default. A current rule with the same name replaces the inherited rule with that name in the same namespace.

`mode = "replace"` discards inherited rules for that namespace first, then uses only the rules authored in the current block.

`remove` is namespace-local. A removed name only affects that one namespace.

## Shared nested tables

### `selector`

Selectors are supported on all agent rules and on scene rules that explicitly list a `selector` field.

Example:

<!-- no-validate -->
```toml
[datasets.a43.screening.agents.rule_name.selector]
mode = "include"
categories = ["CAR"]
```

or inline:

<!-- no-validate -->
```toml
[datasets.a43.screening.agents.rule_name]
... # rule fields
selector = { mode = "include", categories = ["CAR"] }
```

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `"include"` or `"exclude"` | Whether matching categories are kept in scope or excluded from scope. | `"include"` |
| `categories` | `array[str]` | Agent categories included in the selector. | `required` |

### `tolerance`

Agent rules may define a tolerance table.

Example:

```toml
[datasets.a43.screening.agents.anchor_present]
rule = "frames"
frames = [19]
tolerance = { absolute = 1, relative = 0.05 }
```

| Key | Type | Description | Default |
|---|---|---|---|
| `absolute` | `float` | Maximum number of invalid agents to tolerate. | `none` |
| `relative` | `float` | Maximum invalid-agent fraction to tolerate. | `none` |

At least one of `absolute` or `relative` must be set.

### `require`

Agent rules may define a `require` table to keep a scene only when enough selected
agents pass that rule.

Example:

```toml
[datasets.a43.screening.agents.anchor_present]
rule = "frames"
frames = [19]
require = { absolute = 3, relative = 0.75 }
```

| Key | Type | Description | Default |
|---|---|---|---|
| `absolute` | `int` | Minimum number of selected agents that must pass the rule. | `none` |
| `relative` | `float` | Minimum selected-agent pass fraction required to keep the scene. | `none` |

At least one of `absolute` or `relative` must be set. When both are set, both
thresholds must pass.

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

The `prune_by` rule applies an agent-level rule in cleanup mode. For example, if
you want to prune short tracks before applying scene-level rules, you can use a
nested agent rule under `prune_by`.

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"prune_by"` | Remove rows for agents that fail a nested agent rule. | `required` |
| `agent_rule` | `inline table` or nested table | Agent rule used to decide which agents are pruned. | `required` |

Example:

```toml
[datasets.a43.screening.cleanup.prune_short_tracks]
rule = "prune_by"
agent_rule = { rule = "min_observations", minimum = 8, selector = { mode = "include", categories = ["CAR"] } }
```

## Scene rules

Scene rules decide whether a scene window is retained as a whole.

They live under:

- `[...screening.scenes.<rule_name>]`

### `rule = "agent_range"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"agent_range"` | Require the retained-agent count to stay within a range. | `required` |
| `minimum` | `int` | Optional minimum retained-agent count. | `none` |
| `maximum` | `int` | Optional maximum retained-agent count. | `none` |
| `selector` | `table` | Optional category selector. | `none` |

### `rule = "category_range"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"category_range"` | Require category-specific retained-agent count ranges. | `required` |
| `ranges` | `table` | Mapping from category name to `{ minimum, maximum }`. | `required` |

Example:

```toml
[datasets.a43.screening.scenes.category_mix]
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

### `rule = "max_missing_frames"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"max_missing_frames"` | Limit missing frames across the scene window. | `required` |
| `maximum` | `int` | Maximum allowed number of missing frames. | `required` |
| `selector` | `table` | Required category selector in the current config model. | `required` |

## Agent rules

Agent rules validate agents inside a retained scene.

They live under:

- `[...screening.agents.<rule_name>]`

All agent rules may optionally define:

- `selector` to target specific categories for that rule
- `tolerance` to allow some invalid agents without discarding the whole scene
- `require` to require a minimum number or fraction of selected agents to pass

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

### `rule = "min_observations"`

| Key | Type | Description | Default |
|---|---|---|---|
| `rule` | `"min_observations"` | Require a minimum number of observations per agent. | `required` |
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
| `minimum` | `float` | Minimum distance in meters. | `required` |
| `selector` | `table` | Optional category selector. | `none` |
| `tolerance` | `table` | Optional scene-level tolerance for invalid agents. | `none` |

## Practical example

```toml
[profiles.basic_screening.screening.cleanup.remove_animals]
rule = "exclude"
categories = ["ANIMAL"]

[datasets.a43.screening.cleanup]
mode = "extend"

[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[datasets.a43.screening.scenes]
mode = "extend"

[datasets.a43.screening.scenes.min_context]
rule = "agent_range"
minimum = 2

[datasets.a43.screening.agents.anchor_present]
rule = "frames"
frames = [19]
tolerance = { absolute = 1, relative = 0.05 }
selector = { mode = "include", categories = ["CAR"] }

[datasets.a43.screening.agents.observation_floor]
rule = "min_observations"
minimum = 8
```

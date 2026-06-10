# Screening

<div class="section-intro" markdown="1">
Screening is the quality-control stage of the processing pipeline. It removes irrelevant rows, decides whether a scene is usable, and marks which agents satisfy per-agent quality rules.
</div>

For exact syntax and field tables, see the [screening reference](../reference/configuration/screening.md).

## Mental model

Screening has three rule families:

| Rule family | Main question | Typical effect |
| --- | --- | --- |
| `cleanup` | Should these rows or agents be removed before validation? | Drops data before scene checks run. |
| `scenes` | Is this scene usable? | Keeps or rejects the whole scene. |
| `agents` | Which agents satisfy per-agent quality rules? | Marks agents as passed or failed, with optional scene-level aggregate thresholds. |

They run in that order.

## Current config shape

Rules are named maps, not anonymous lists. That matters because inheritance and removal work by
rule name inside each namespace.

```toml
[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[datasets.a43.screening.scenes.min_context]
rule = "agent_range"
minimum = 2

[datasets.a43.screening.agents.observation_floor]
rule = "min_observations"
minimum = 8
```

The rule names here are `trim_static`, `min_context`, and `observation_floor`.

## When to use each rule family

Use `cleanup` when the data is irrelevant by definition.

Examples:

- remove bookkeeping categories everywhere
- prune obviously invalid tracks before the scene is judged
- keep only a deliberate set of categories

Use `scenes` when the requirement is about the scene as a whole.

Examples:

- require at least two agents after cleanup
- require a specific category mix
- require enough frame coverage across the scene

Use `agents` when some agents may be weak but the scene can still be useful.

Examples:

- require a minimum number of observations per selected agent
- limit gaps for pedestrians only
- keep the scene, but mark which agents passed the quality bar

## Selectors, tolerance, and pass requirements

Agent rules can be narrowed with a selector:

```toml
[datasets.a43.screening.agents.anchor_present]
rule = "frames"
frames = [19]

[datasets.a43.screening.agents.anchor_present.selector]
mode = "include"
categories = ["CAR"]
```

Tolerance makes agent checks less brittle:

```toml
[datasets.a43.screening.agents.anchor_present]
rule = "frames"
frames = [19]
tolerance = { absolute = 2, relative = 0.2 }
```

This means a scene can survive even if a small number of selected agents fail the rule; in this case
2 absolute failures or 20% relative failures would be allowed.

Agent rules can also require a minimum number or fraction of selected agents to pass:

```toml
[datasets.a43.screening.agents.anchor_present]
rule = "frames"
frames = [19]
require = { absolute = 3, relative = 0.75 }
```

This keeps a scene only when at least 3 selected agents pass the rule and at least 75% of selected
agents pass the rule. `require` is evaluated after the agent rule selector.

`tolerance` and `require` may be used together. In that case, both aggregate checks must pass.

!!! tip "Non-passing agents will be marked"
    If `tolerance` or `require` allows a scene to survive with failed agents, those
    agents still exist in the scene. They are marked in the output records so
    downstream code can treat them differently.

## Extending inherited rules

Merge behavior is controlled per namespace, not on the parent `[...screening]` table.

```toml
[datasets.a43.screening.scenes]
mode = "extend"
remove = ["old_rule"]
```

The behavior is:

- `extend` is the default and merges the current namespace by rule name
- `replace` discards inherited rules in that namespace first
- `remove` runs last and drops matching names only from that namespace

This is why stable rule names matter.

## Worked multi-profile example

When multiple profiles are used, screening is resolved in order: start from the dataset defaults,
apply profiles in `uses` order, then apply the dataset's own block last.

Start with these three profiles:

```toml
[profiles.base.screening.agents.min_obs]
rule = "min_observations"
minimum = 4

[profiles.base.screening.scenes.min_context]
rule = "agent_range"
minimum = 2

[profiles.strict.screening.agents]
mode = "extend"

[profiles.strict.screening.agents.min_obs]
rule = "min_observations"
minimum = 8

[profiles.strict.screening.agents.anchor_present]
rule = "frames"
frames = [19]

[profiles.curated.screening.cleanup]
mode = "replace"

[profiles.curated.screening.agents]
mode = "replace"

[profiles.curated.screening.scenes]
mode = "replace"

[profiles.curated.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[profiles.curated.screening.scenes.category_mix]
rule = "category_range"
ranges = { CAR = { minimum = 1 }, PEDESTRIAN = { minimum = 1 } }
```

After `base`, the active rules are:

- `agents.min_obs`
- `scenes.min_context`

Then `strict` extends the `agents` namespace, so it overrides `agents.min_obs` from `4` to `8` and
adds `agents.anchor_present`.

Then `curated` replaces each namespace explicitly:

- `cleanup` becomes only `trim_static`
- `agents` becomes empty
- `scenes` becomes only `category_mix`

Now introduce the dataset itself:

```toml
[datasets.a43]
uses = ["base", "strict", "curated"]

[datasets.a43.screening.scenes]
mode = "extend"
remove = ["category_mix"]

[datasets.a43.screening.scenes.final_context]
rule = "agent_range"
minimum = 3
```

The dataset sees the profile result first, then extends only the `scenes` namespace. So the final
effective screening is:

- `cleanup.trim_static`
- `scenes.final_context`

The important takeaway is that `extend` and `replace` are namespace-local. A `replace` in
`screening.scenes` does not touch `cleanup` or `agents`.

## Common patterns

Remove noise, then require minimally useful scenes:

```toml
[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[datasets.a43.screening.scenes.min_context]
rule = "agent_range"
minimum = 2
```

Require a specific interaction mix:

```toml
[datasets.a43.screening.scenes.category_mix]
rule = "category_range"
ranges = { CAR = { minimum = 1 }, PEDESTRIAN = { minimum = 1 } }
```

Keep broad scenes, but demand stronger pedestrian tracks:

```toml
[datasets.a43.screening.agents.pedestrian_span]
rule = "min_consecutive_frames"
minimum = 12

[datasets.a43.screening.agents.pedestrian_span.selector]
mode = "include"
categories = ["PEDESTRIAN"]
```

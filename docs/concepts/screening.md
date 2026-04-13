# Screening

<div class="section-intro" markdown="1">
Screening is the quality control stage of the processing pipeline. It has three main purposes: 1) to remove irrelevant data before validation, 2) to determine which scenes are usable as samples, and 3) to mark which agents satisfy per-agent quality rules. Screening rules are inherited and can be extended or removed in child profiles.
</div>

For exact syntax and field tables, see the [screening reference](../reference/configuration/screening.md).

## Mental model

Screening has three rule families:

| Rule family | Main question | Typical effect |
| --- | --- | --- |
| `cleanup` | Should these rows or agents be removed before validation? | Drops data before scene checks run. |
| `scene` | Is this scene usable as a sample? | Keeps or rejects the whole scene. |
| `agent` | Which agents satisfy per-agent quality rules? | Marks agents as passed or failed, with optional tolerance. |

They run in that order.

## Current config shape

Rules are named maps, not anonymous lists. That matters because inheritance and removal work by
rule name.

```toml
[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[datasets.a43.screening.scene.min_context]
rule = "agent_range"
minimum = 2

[datasets.a43.screening.agent.sample_floor]
rule = "min_samples"
minimum = 8
```

The rule names here are `trim_static`, `min_context`, and `sample_floor`.

## When to use each rule family

Use `cleanup` when the data is irrelevant by definition.

Examples:

- remove bookkeeping categories everywhere
- prune obviously invalid tracks before the scene is judged
- keep only a deliberate set of categories

Use `scene` when the requirement is about the sample as a whole.

Examples:

- require at least two agents after cleanup
- require a specific category mix
- require enough frame coverage across the scene

Use `agent` when some agents may be weak but the scene can still be useful.

Examples:

- require a minimum number of samples per selected agent
- limit gaps for pedestrians only
- keep the scene, but mark which agents passed the quality bar

## Selectors and tolerance

Agent rules can be narrowed with a selector:

```toml
[datasets.a43.screening.agent.anchor_present]
rule = "frames"
frames = [19]

[datasets.a43.screening.agent.anchor_present.selector]
mode = "include"
categories = ["CAR"]
```

Tolerance makes agent checks less brittle:

```toml
[datasets.a43.screening.agent.anchor_present.tolerance]
absolute = 1
relative = 0.05
```

This means a scene can survive even if a small number of selected agents fail the rule.

## Extending inherited rules

`screening` is the section where merge behavior is explicit:

```toml
[datasets.a43.screening]
mode = "extend"
remove = ["old_rule"]
```

- `replace` discards inherited rules and uses only the current block
- `extend` merges by rule name
- `remove` drops named rules across the cleanup, scene, and agent namespaces

This is why stable rule names matter.

## Common patterns

Remove noise, then require minimally useful scenes:

```toml
[datasets.a43.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[datasets.a43.screening.scene.min_context]
rule = "agent_range"
minimum = 2
```

Require a specific interaction mix:

```toml
[datasets.a43.screening.scene.category_mix]
rule = "category_range"
ranges = { CAR = { minimum = 1 }, PEDESTRIAN = { minimum = 1 } }
```

Keep broad scenes, but demand stronger pedestrian tracks:

```toml
[datasets.a43.screening.agent.pedestrian_span]
rule = "min_consecutive_frames"
minimum = 12

[datasets.a43.screening.agent.pedestrian_span.selector]
mode = "include"
categories = ["PEDESTRIAN"]
```

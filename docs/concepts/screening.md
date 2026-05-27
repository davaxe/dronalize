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
| `agent` | Which agents satisfy per-agent quality rules? | Marks agents as passed or failed, with optional scene-level aggregate thresholds. |

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

## Selectors, tolerance, and pass requirements

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
[datasets.a43.screening.agent.anchor_present]
rule = "frames"
frames = [19]
tolerance = { absolute = 2, relative = 0.2 }
```

This means a scene can survive even if a small number of selected agents fail the rule; in this case 2 absolute failures or 20% relative failures would be allowed. Both absolute and relative tolerances are applied, so if either threshold is breached the agent-based screening rule fails for the scene.

Agent rules can also require a minimum number or fraction of selected agents to pass:

```toml
[datasets.a43.screening.agent.anchor_present]
rule = "frames"
frames = [19]
require = { absolute = 3, relative = 0.75 }
```

This keeps a scene only when at least 3 selected agents pass the rule and at least 75% of selected agents pass the rule. `require` is evaluated after the agent rule selector. If no selected agents exist, a positive requirement fails.

`tolerance` and `require` may be used together. In that case, both aggregate checks must pass: the scene must stay within the invalid-agent tolerance and satisfy the minimum passing-agent requirement.

!!! tip "Non-passing agents will be marked"
    If `tolerance` or `require` allows a scene to survive with failed agents, those
    agents still exist in the scenes. Importantly, they will be marked in the
    output records so downstream code can handle them differently, if needed.

    In practice this means the scene can still be emitted, but the runtime keeps
    track of which agent ids passed screening and which did not. When the scene
    is encoded, that information is exported as a per-agent `screened_agent_mask`.

    This is useful when you want to keep borderline or context agents in the
    scene graph while still training, evaluating, or visualizing with a stricter
    subset. For example, a downstream model can use all agents as context but
    only compute loss on agents whose screening mask is `true`.

## Extending inherited rules

`screening` is the section where merge behavior is explicit:

```toml
[datasets.a43.screening]
mode = "extend"
remove = ["old_rule"]
```

The merge order is:

1. start from inherited screening rules, or empty rule maps if nothing is inherited
2. apply `mode`
3. apply `remove`

The behavior is:

- `extend` is the default and merges the current block into inherited rules by name
- in `extend`, a rule only overrides an inherited rule with the same name in the same namespace
- `replace` discards inherited rules first and keeps only rules authored in the current block
- `remove` runs last and drops matching names across the cleanup, scene, and agent namespaces
- because `remove` runs last, it can remove both inherited rules and rules declared in the current block

This is why stable rule names matter.

## Worked multi-profile example

When multiple profiles are used, screening is resolved in order: start from the dataset defaults, apply profiles in `uses` order, then apply the dataset’s own block last.

Start with these three profiles:

```toml
[profiles.base.screening.agent.min_obs]
rule = "min_samples"
minimum = 4

[profiles.base.screening.scene.min_context]
rule = "agent_range"
minimum = 2

[profiles.strict.screening]
mode = "extend"

[profiles.strict.screening.agent.min_obs]
rule = "min_samples"
minimum = 8

[profiles.strict.screening.agent.anchor_present]
rule = "frames"
frames = [19]

[profiles.curated.screening]
mode = "replace"

[profiles.curated.screening.cleanup.trim_static]
rule = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[profiles.curated.screening.scene.category_mix]
rule = "category_range"
ranges = { CAR = { minimum = 1 }, PEDESTRIAN = { minimum = 1 } }
```

Assuming all these profiles are inherited in the order `base`, then `strict`, then `curated`. The `base` profile adds two rules: `agent.min_obs` and `scene.min_context`.

The `strict` profile uses `mode = "extend"`, so it keeps what it inherits, overrides `agent.min_obs` from `4` to `8`, and adds `agent.anchor_present`.

```toml
[profiles.strict.screening]
mode = "extend"
```

So after `base` and `strict`, the active rules are:

- `agent.min_obs`
- `scene.min_context`
- `agent.anchor_present`

Then `curated` changes the picture completely:

```toml
[profiles.curated.screening]
mode = "replace"
```

Because it uses `replace`, it discards everything inherited from earlier profiles and starts over. After `curated`, only these rules remain:

- `cleanup.trim_static`
- `scene.category_mix`

Now introduce the dataset itself:

```toml
[datasets.a43]
uses = ["base", "strict", "curated"]

[datasets.a43.screening]
mode = "extend"
remove = ["category_mix"]

[datasets.a43.screening.scene.final_context]
rule = "agent_range"
minimum = 3
```

The profile chain is applied in the order shown in `uses`, so the dataset first sees the result of `base`, then `strict`, then `curated`. Since `curated` replaced the inherited screening state, the dataset starts from:

- `cleanup.trim_static`
- `scene.category_mix`

Its own screening block then extends that state, adds `scene.final_context`, and removes `scene.category_mix`.

```toml
[datasets.a43.screening]
mode = "extend"
remove = ["category_mix"]
```

So the final effective screening is:

- `cleanup.trim_static`
- `scene.final_context`

The important takeaway is that `extend` keeps building on the current state, while `replace` resets it at that point in the chain. After that, `remove` is applied to the result of the current block. Because the dataset block always runs last, it always has the final say.

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

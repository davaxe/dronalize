# Filtering

<div class="section-intro" markdown="1">
Filtering decides what data is allowed to survive loading before samples are written. The core idea is to separate three different questions: what should be removed immediately, what makes a scene usable, and what makes agents inside that scene acceptable.
</div>

For exact syntax, supported rule types, and option details, see the [filter reference](../reference/configuration/filter.md).

## Mental model

Filtering has three rule families:

| Rule family | Main question | Typical effect |
| --- | --- | --- |
| `cleanup` | Should these rows be considered at all? | Removes rows before validation. |
| `scene` | Is this scene good enough to keep? | Keeps or rejects the whole scene. |
| `agent` | Do the agents in this scene meet quality requirements? | Evaluates per-agent quality and can cause the scene to fail if too many agents fail. |

They run in that order, so earlier rules shape what later rules can see.
Agent rules are checks, not cleanup rules. If rows should disappear before validation, use `cleanup`.

## When to use each rule family

Use `cleanup` when data is irrelevant by definition.

Examples:

- remove static objects everywhere
- drop bookkeeping categories before any validation
- keep only a narrow set of categories for a focused dataset

Use `scene` when the requirement is about the sample as a whole.

Examples:

- require at least two retained agents
- require a specific category mix such as cars and pedestrians
- require enough frame coverage across the whole window

Use `agent` when the scene may still be useful even if some individual agents are weak.

Examples:

- require enough samples per agent
- reject scenes where too many agents have large gaps
- apply stricter rules only to certain categories

## Choosing rule types

For cleanup:

- use `exclude` to remove known noise
- use `include` to keep only a deliberate category set

For scene quality:

- use `min_agents`, `agent_range`, or `category_range` for count-based requirements
- use `frames`, `window`, or `max_missing_frames` for coverage-based requirements

For agent quality:

- use `min_samples`, `min_span`, or `min_consecutive_frames` when you care about amount or continuity
- use `max_missing_frames` or `max_gap` when you want to limit missingness
- use `starts_by_frame` or `ends_after_frame` when agents must overlap certain parts of the sample

## Selectors and tolerance

Selectors narrow a rule to certain categories. They are useful when different agent types deserve different quality requirements.

```toml
[[datasets.a43.loader.filter.agent]]
type = "min_samples"
minimum = 15

[datasets.a43.loader.filter.agent.selector]
mode = "include"
categories = ["PEDESTRIAN"]
```

Tolerance makes agent rules less brittle by allowing some invalid agents without rejecting the scene immediately.

```toml
[datasets.a43.loader.filter.agent.tolerance]
kind = "relative"
relative = 0.2
```

This expresses a rule like "most selected agents should pass," rather than "every selected agent must pass."

!!! note "Selectors availability for scene rules"
    Selectors are available for some scene rules, but not all. For filters that
    support them, the syntax is the same as for agent rules. Check the filter
    reference for details on which scene rules support selectors.

## Common patterns

Remove obvious noise, then keep only minimally useful scenes:

```toml
[datasets.a43.loader.filter]
mode = "replace"

[[datasets.a43.loader.filter.cleanup]]
type = "exclude"
categories = ["STATIC_OBJECT", "UNIMPORTANT"]

[[datasets.a43.loader.filter.scene]]
type = "min_agents"
minimum = 2
```

Require a specific interaction type:

```toml
[[datasets.a43.loader.filter.scene]]
type = "category_range"
ranges = { CAR = { minimum = 1 }, PEDESTRIAN = { minimum = 1 } }
```

Keep broad scenes, but require stronger pedestrian tracks:

```toml
[[datasets.a43.loader.filter.agent]]
type = "min_consecutive_frames"
minimum = 12

[datasets.a43.loader.filter.agent.selector]
mode = "include"
categories = ["PEDESTRIAN"]
```

## Layering filters

When configs are layered, filters can be:

- `replace` to define the whole filter from scratch
- `extend` to keep inherited rules and add or override a few
- `remove` to drop inherited rules by name

Use `rule_id` when you want stable names for extending or removing rules, especially if you define more than one rule of the same type.

*[rows]: explicit rows in the tabular data structure. Each row correspond to a singular agent at a specific frame.
*[scene]: collection of agents over a contiguous window of frames. The sample unit that the dataset is split into.

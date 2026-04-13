# Lane-change sampling

<div class="section-intro" markdown="1">
Lane-change sampling is a highway-oriented scene selection policy. It does not change the scene
format. It changes which extracted windows are kept so the output is less dominated by straight
driving.
</div>

For exact fields, see the [`scenes.lane_change`](../reference/configuration/scenes.md) reference.

## Why it exists

Highway datasets are usually imbalanced:

- most windows contain steady lane following
- relatively few windows contain meaningful lane changes

Lane-change sampling keeps all positive lane-change windows and deterministically thins the
negative windows.

## Current config path

Lane-change sampling lives under `scenes`:

```toml
[datasets.highd.scenes.lane_change]
persist = 2
margin_before = 1
margin_after = 0
required_lane_changes = 2
negative_keep_every = 3
```

## What counts as a positive window

A window counts as positive when it contains enough valid lane-change events.

A valid lane change is stricter than "the lane id changed once". The pipeline checks that:

- the lane assignment changes
- the new lane persists long enough
- any required context before or after the event is present

Those checks are controlled by `persist`, `margin_before`, and `margin_after`.

## How sampling changes

Once windows are labeled:

- positive windows are always kept
- negative windows are kept only every `N`th time, where `N = negative_keep_every`

If `negative_keep_every = 1`, the output selection is the same as the standard pipeline and the
lane-change extension is skipped.

## When to use it

Use lane-change sampling when all of the following are true:

- the dataset is lane-oriented highway traffic
- the loader exposes a usable `lane_id` signal
- your downstream task cares about lane-change behavior enough that the raw class imbalance matters

Several built-in highway datasets ship with lane-change-aware defaults, including `highd`, `exid`,
`i80`, `us101`, and `ad4che`.

## Practical guidance

- Increase `persist` if lane IDs are noisy.
- Increase `required_lane_changes` if you want denser maneuver windows.
- Decrease `negative_keep_every` if you want to keep more straight-driving windows.
- Set `negative_keep_every = 1` if you want the ordinary window distribution back.

## Relationship to the rest of the pipeline

Lane-change sampling sits inside the normal trajectory pipeline:

1. a source is loaded
2. windows are extracted
3. lane-change events are detected for those windows
4. positive windows are kept and negative windows are thinned
5. the surviving scenes continue through screening, split assignment, schema conversion, and export

So this feature changes selection pressure, not the schema, backend, or runtime interface.

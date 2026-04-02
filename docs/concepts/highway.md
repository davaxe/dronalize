# Highway pipeline

<div class="section-intro" markdown="1">
The highway pipeline is a lane-change-aware sampling mode for highway-style datasets. Its goal is not to change the scene format, but to change which windows survive preprocessing so the output is less dominated by straight-driving traffic.
</div>

For the exact TOML keys and validation rules, see the [`[loader.highway]` configuration reference](../reference/configuration/loader.md).

## Why it exists

Many highway datasets are heavily imbalanced:

- most windows contain steady lane following
- relatively few windows contain meaningful lane changes
- a model trained on the raw distribution can become good at the common easy cases while underperforming on the rarer but more important maneuvering cases

The highway pipeline addresses that imbalance during preprocessing by keeping all positive lane-change windows and deterministically thinning the negative no-change windows.

## What counts as a positive window

At a high level, a window is treated as positive when it contains enough **valid lane-change events**.

A valid lane change is not just any moment where the lane id flips. The pipeline checks that:

- the lane assignment changes between consecutive frames
- the new lane persists long enough to avoid counting noisy lane-id flicker
- there are enough frames before and after the event when those margins are required

These checks are controlled by the highway config and are applied before the final scene sampling step.

## How sampling changes

Once windows have been labeled, the highway pipeline applies a simple policy:

- positive windows are always kept
- negative windows are thinned by keeping only every `N`th one

This means the pipeline does **not** invent synthetic lane-change scenes. It only changes how aggressively non-event windows are dropped.

If `negative_keep_every = 1`, the highway pipeline has no effect on scene selection and the standard trajectory pipeline is used instead.

## When to use it

Use highway sampling when all of the following are true:

- the dataset is lane-oriented highway traffic
- the loader provides a usable `lane_id` signal
- your downstream task cares about lane-change behavior enough that raw class imbalance is a problem

Do not use it as a default for every dataset. It is specific to datasets and loaders that expose highway-style sampling behavior.

If a dataset does not support highway sampling, adding `[loader.highway]` is a configuration error and planning fails early with a clear message.

!!! info "Inspect CLI command"
    The `dronalize inspect <dataset>` will show `highway pipeline` under the dataset capabilities when that dataset supports this behavior. For example:
    when running `dronalize inspect highd`:
    ```
    Dataset inspect: highd
                    ╷
      Dataset       │ highd
      Capabilities  │  map   custom splits   highway pipeline
      Native schema │ positions_velocity_acceleration (6 features)
      Schema fields │ frame, id, x, y, vx, vy, ax, ay, agent_category
      Split support │ source*
      ...
    ```
    Some highway dataset do **not** support this behaviour, because they do
    not expose lanes in their data.

## Configuration model

Highway sampling is configured under `[datasets.<name>.loader.highway]`.

Example:

```toml
[datasets.highd.loader.highway]
persist = 2
margin_before = 1
margin_after = 0
required_lane_changes = 2
negative_keep_every = 3
```

Conceptually, these settings mean:

- only count lane changes that persist for at least `2` frames
- require `1` frame of context before the change
- require at least `2` valid lane changes for a window to count as positive
- keep all positive windows, but only every third negative window

## Practical guidance

- Start from the dataset defaults when they already expose highway sampling.
- Increase `persist` if lane ids are noisy and short-lived flips are being counted as events.
- Increase `required_lane_changes` if you want to focus on denser maneuver windows.
- Decrease `negative_keep_every` if you want to keep more straight-driving.
- Set `negative_keep_every = 1` if you want the regular distribution back without removing the highway block from the config.

## Relationship to the rest of the pipeline

Highway sampling sits on top of the normal trajectory pipeline:

1. the loader ingests trajectories
2. windowing creates candidate scenes
3. lane-change detection labels those windows
4. highway sampling keeps all positives and thins negatives
5. the remaining scenes continue through the normal split and writer flow

So this feature changes **selection pressure**, not the output schema, writer format, or runtime interface.

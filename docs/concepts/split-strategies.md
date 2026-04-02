# Split strategies

<div class="section-intro" markdown="1">
Splitting controls how processed data is routed into train, val, and test outputs. The main choice is not just the ratio, but what unit should stay together when data is assigned to a split.
</div>

For exact split syntax and mode-specific fields, see the [split reference](../reference/configuration/split.md). For dataset-specific support, see the [dataset reference](../reference/datasets/index.md).

## Mental model

When choosing a split strategy, ask two questions:

1. What should stay together: scenes, sources, time blocks, or native dataset partitions?
2. How much isolation do you want between train, val, and test?

That leads to six practical choices:

| Mode | Best for |
| --- | --- |
| `none` | Processing everything without split routing. |
| `native` | Reproducing dataset-provided benchmark partitions. |
| `scene` | Random assignment at scene level when scenes are already the natural independent unit. |
| `source` | Keeping full recordings or source files together. |
| `time` | Preserving chronology with contiguous time blocks. |
| `shuffled-time` | Keeping short temporal blocks intact while mixing them across splits. |

`auto` is a convenience mode that asks `dronalize` to use the dataset's recommended custom split strategy when one is available. If the dataset does not expose a clear recommendation, choosing an explicit mode is required.

!!! note "`split-support` CLI command"
    
    Not all modes are supported for all datasets. To check which split modes a dataset supports, use the `split-support` command:

    ```bash
    dronalize split-support <dataset-name>
    ```

     This lists the modes that are available for that dataset and any relevant details about how they work.

## Choosing the right split

Use `native` when you want comparability with a published benchmark.

Use `source` when full recordings or files should never be split apart.

Use `time` when temporal leakage is the main concern and chronology should be preserved.

Use `shuffled-time` when you still want temporal blocks, but one long train block and one long test block would be too rigid.

Use `scene` only when scene-level randomization matches the dataset structure and leakage between nearby scenes is not a major concern.

## Ratios, gaps, and segments

The remaining split settings describe how the chosen mode behaves:

- `ratio` sets the train, val, and test proportions for custom modes
- `gap` inserts a temporal buffer between neighboring time-based partitions
- `segments` matters only for `shuffled-time` and controls how many temporal chunks are created before shuffling
- `read` matters only for `native` and lets you select which native partitions to process

## Practical workflow

1. Check which split modes the dataset supports.
2. Use `native` if you want benchmark compatibility.
3. Otherwise choose the unit that should stay intact: source, scene, or time block.
4. Add `gap` or `segments` only when using a time-based strategy.

This keeps split decisions tied to data leakage and evaluation goals rather than just to convenience.

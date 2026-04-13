# Split strategies

<div class="section-intro" markdown="1">
Splitting controls how processed scenes are routed into train, val, and test outputs. The main
choice is not just the ratio. It is what unit should stay together when data is assigned to a split.
</div>

For exact syntax, see the [split reference](../reference/configuration/split.md). For dataset
support, see the [dataset reference](../reference/datasets/index.md).

## Available strategies

The current split strategies are:

| Strategy | Best for |
| --- | --- |
| `none` | Processing everything without split routing. |
| `native` | Reproducing dataset-defined benchmark partitions. |
| `scene` | Random assignment at scene level. |
| `source` | Keeping full raw sources together. |
| `time` | Preserving chronology with contiguous time blocks. |
| `shuffled-time` | Keeping shorter temporal blocks intact while mixing them across splits. |

## Choosing the right split

Use `native` when you want comparability with the dataset's benchmark setup.

Use `source` when complete recordings or files must stay together.

Use `time` when temporal leakage is the main concern and chronology should stay intact.

Use `shuffled-time` when you still want temporal blocks, but one long train block and one long test
block would be too rigid.

Use `scene` only when scene-level randomization matches the dataset structure and leakage between
nearby scenes is acceptable.


## Ratios and native reads

The remaining split settings describe how the chosen strategy behaves:

- `ratio` sets train, val, and test weights for custom splits: `scene`, `source`, `time`, and `shuffled-time`
- `gap` inserts a temporal buffer between neighboring time partitions
- `segments` matters only for `shuffled-time`, and specifies how many blocks to divide the timeline into before shuffling.
- `splits` matters only for `native` and selects which native partitions to read

From the CLI, the native `splits` selection is exposed as `--read-split`.

## Inspect support before choosing

Not every dataset supports every mode. Use:

```bash
dronalize split-support <dataset>
```
This is the quickest way to check whether a dataset supports native partitions, time-based splitting, or only scene and source routing.

??? example "`split-support` example"
    Using `split-support` to check the `a43` dataset's supported split strategies:
    ```bash
    dronalize split-support a43
    ```
    gives output like:
    ```text
    Dataset       │ a43
    Native splits │ none
    Scene split   │ yes
    Source split  │ yes
    Time split    │ yes
    ```

## Practical workflow

1. Check `split-support` for the dataset.
2. Use `native` when benchmark compatibility matters.
3. Otherwise choose the unit that should stay intact: source, scene, or time block.
4. Add `gap` and `segments` only when a time-based strategy needs them.

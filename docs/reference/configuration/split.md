# `[split]` section

<div class="section-intro" markdown="1">
Split settings control how processed data is routed into train, val, and test outputs. The most important choice is the split `mode`, with the remaining keys depending on that mode.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `str` | Split mode used when routing data into train, val, and test sets. | `"none"` |
| `ratio` | `table` | Train/val/test weights. Required for custom split modes and invalid for `"native"` and `"none"`. | `required for custom modes` |
| `gap` | `int` | Gap between neighboring temporal segments when `mode = "time"` or `mode = "shuffled-time"`. | `0` |
| `segments` | `int` | Number of temporal segments to create before shuffling when `mode = "shuffled-time"`. | `required for "shuffled-time"` |
| `read` | `array[str]` | Dataset-native split names to read when `mode = "native"`, for example `["train"]`. If omitted, all dataset-native splits are used. | `none` |


The most central part of the split configuration is the `mode`. In total there are seven options:

- `"none"`: the default option, which processes all data without splitting.
- `"native"`: use the predefined splits provided by the dataset, if available.
- `"scene"`: split randomly at the scene level. This is simple but can still cause leakage when neighboring scenes are strongly related. It is only supported for datasets that export at scene level.
- `"source"`: split randomly at source level, often individual files or recordings. This works best when sources are numerous and relatively independent.
- `"time"`: split the data into continuous temporal segments for train, val, and test.
- `"shuffled-time"`: similar to `"time"` but first partitions the recording into `N` segments, then shuffles those segments into splits based on the provided ratios.
- `"auto"`: use the recommended split mode for a dataset.

!!! note "Mode availability"
    Not all datasets support all split modes. For example, `"native"` is only available for datasets that provide predefined splits, and `"scene"` is only available for datasets that export at scene level. The CLI will raise an error if you try to use an unsupported mode.

!!! note "Validation"
    `gap` is only valid for `"time"` and `"shuffled-time"`. `segments` is only valid for `"shuffled-time"`. `read` is only valid for `"native"`.

## Minimal example


```toml
[datasets.a43.split]
mode = "shuffled-time"
ratio = { train = 0.7, val = 0.2, test = 0.1 }
segments = 10 # When "shuffled-time" is used, segments is required.
gap = 2 
```

To select only specific dataset-native splits:

```toml
[datasets.waymo.split]
mode = "native"
read = ["train", "val"]
```

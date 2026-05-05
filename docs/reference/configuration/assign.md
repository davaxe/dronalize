# `[assign]` section

<div class="section-intro" markdown="1">
Assignment settings control how loaded scenes are routed into train, val, and
test outputs.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `strategy` | `str` | Output assignment strategy used when routing scenes into train, val, and test sets. | `"none"` |
| `ratio` | `table` | Train/val/test weights. Required for scene, source, time, and shuffled-time assignment. | `required for weighted assignment` |
| `gap` | `int` | Gap between neighboring temporal segments when `strategy = "time"` or `strategy = "shuffled-time"`. | `0` |
| `segments` | `int` | Number of temporal segments to create before shuffling when `strategy = "shuffled-time"`. | `required for "shuffled-time"` |

Assignment strategies:

- `"none"`: produce unsplit output.
- `"preserve-native"`: preserve dataset-native split labels in output.
- `"scene"`: assign splits randomly at the scene level.
- `"source"`: assign splits randomly at the source level.
- `"time"`: assign splits across continuous temporal segments.
- `"shuffled-time"`: partition time into `N` segments, then shuffle segments into output splits.

!!! note "Strategy availability"
    Not all datasets support all assignment strategies. For example,
    `"preserve-native"` is only available for datasets with native partitions,
    and `"scene"` is only available for datasets that export at scene level.

!!! note "Validation"
    `ratio` is required for `"scene"`, `"source"`, `"time"`, and `"shuffled-time"`.
    `gap` is only valid for `"time"` and `"shuffled-time"`.
    `segments` is only valid for `"shuffled-time"`.

## Examples

Keep dataset-native labels:

```toml
[datasets.nuscenes.assign]
strategy = "preserve-native"
```

Assign by shuffled time blocks:

```toml
[datasets.a43.assign]
strategy = "shuffled-time"
ratio = { train = 0.7, val = 0.2, test = 0.1 }
segments = 10
gap = 2
```

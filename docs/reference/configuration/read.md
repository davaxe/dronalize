# `[read]` section

<div class="section-intro" markdown="1">
Read settings control which raw dataset inputs are loaded before any scene
assignment or output writing happens.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `strategy` | `str` | Read strategy used to select raw dataset inputs. | `"all"` |
| `splits` | `array[str]` | Dataset-native partitions to read when `strategy = "native"`, for example `["train"]`. | `all supported native splits` |

Read strategies:

- `"all"`: process the dataset's full input surface.
- `"native"`: read dataset-native partitions. If `splits` is omitted, all native partitions supported by the dataset are read.

!!! note "Strategy availability"
    Not all datasets support native read selection. The CLI will raise an error
    if you try to use `strategy = "native"` on a dataset without native
    partitions.

!!! note "Validation"
    `splits` is only valid for `strategy = "native"`.

## Examples

Read everything a dataset exposes:

```toml
[datasets.a43.read]
strategy = "all"
```

Read only selected native partitions:

```toml
[datasets.waymo.read]
strategy = "native"
splits = ["train", "val"]
```

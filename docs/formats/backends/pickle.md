# Pickle

<div class="section-intro" markdown="1">
The `pickle` backend is the simplest persisted format in `dronalize`. It requires no optional
dependencies and stores one scene record per file.
</div>

## Why choose Pickle

Pickle is a good choice when you want:

- a dependency-light setup
- straightforward local inspection of per-scene files
- a simple baseline format for experimentation

Trade-offs:

- many small files for larger datasets
- not optimized for large-scale streaming training workflows
- no compression support

## Output directory structure

With a split strategy:

```text
output/
├── manifest.json
├── train/
│   ├── 000000.pkl
│   ├── 000001.pkl
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Without a split strategy:

```text
output/
├── manifest.json
└── unsplit/
    ├── 000000.pkl
    ├── 000001.pkl
    └── ...
```

Each `.pkl` file contains one framework-neutral
[`SceneRecord`](../../reference/api/io/storage-and-manifests.md#dronalize.io.SceneRecord).

## Reading Pickle output

<!-- no-validate -->
```python
from pathlib import Path
from dronalize.io.readers import PickleReader

reader = PickleReader(Path("output"), split="train")
record = reader[0]

print(record.scene_number)
print(record.dataset)
print(record.history_features.shape, record.future_features.shape)
```

Use `split=None` (default) for unsplit exports.

## Configuration

Pickle has no backend-specific config block. It uses the general output settings:

- schema
- precision
- recentering

See [Outputs and schemas](../schema.md) for details.

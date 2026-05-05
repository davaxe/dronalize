# Null backend

<div class="section-intro" markdown="1">
The `null` backend runs the full preprocessing pipeline but intentionally skips scene persistence.
It is useful for validation, planning, and runtime benchmarking without storage I/O.
</div>

## What it does

When the `null` backend is selected, the processing pipeline executes as normal with the following differences:

- source loading, preprocessing, screening, splitting, and scene assembly still run
- progress counters and split counts are still produced
- no scene files are written

The root `manifest.json` is still produced, so run metadata remains available.

## Typical use cases

- validate new configs before committing to an export format
- estimate scene counts and split distributions
- benchmark pipeline throughput without disk-write overhead

## Output layout

```text
output/
└── manifest.json
```

No backend-specific split directories or scene files are created.

## How to run

```bash
dronalize process <dataset> \
  --input <input-dir> \
  --output <output-dir> \
  --storage-backend null
```

Pair with `--plan` when you only want to inspect the resolved run plan.

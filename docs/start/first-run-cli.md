# First run (CLI)

<div class="section-intro" markdown="1">
The current CLI exposes five commands: `available`, `inspect`, `show-config`, `split-support`, and
`process`. A good first run is: inspect the dataset, preview the resolved config, then run
`process --plan` before writing anything.
</div>

## Inspect the registry

```bash
dronalize available
dronalize inspect a43
dronalize split-support a43
```

`available` lists the datasets that are usable in the current environment. `inspect` shows a
dataset's defaults, native schema, split support, map support, and dataset-owned options.
`split-support` shows which split strategies are available for that dataset.

!!! note "`available` and `inspect` reflect the current environment"

    The dataset registry only exposes entries whose optional dependencies are currently available.
    If a dataset is missing, check the installation instructions and make sure you have the right
    extras installed.
    
    It does **not** validate that the dataset's expected input files are actually present. If the dataset is listed but you get an error when you try to process, check that the input path is correct and contains the expected files.

## Preview the resolved config

```bash
dronalize show-config a43 --config config.toml
```

`show-config` resolves the same dataset defaults, profile fragments, dataset entry, and CLI
overrides that `process` uses, but it stops before execution. Use it when you want to confirm the
effective `scenes`, `screening`, `split`, `map`, `output`, and dataset-specific settings.

## Plan a processing run

=== ":material-linux: Bash"

    ```bash
    dronalize process a43 \
        --input data/a43/raw \
        --output data/a43/processed \
        --config config.toml \
        --plan
    ```

=== ":fontawesome-brands-windows: PowerShell"

    ```ps1
    dronalize process a43 `
        --input data/a43/raw `
        --output data/a43/processed `
        --config config.toml `
        --plan
    ```

`--plan` resolves the full run and prints a summary without executing it. The summary includes the
dataset, paths, backend, worker count, schema, map usage, split strategy, and any dataset-owned
options that affect the run.

The `process` command currently defaults to the `pickle` backend. Choose a different backend
explicitly when needed:

- `pickle` for the simplest persisted output with no extra dependency
- `mds` for Mosaic Streaming shards, which requires `dronalize[mds]`
- `null` to execute the pipeline without writing any dataset files

## Useful options

| Option | Effect |
| --- | --- |
| `--storage-backend` | Choose `pickle`, `mds`, or `null`. |
| `--scene-schema` | Override the output trajectory schema for this run. |
| `--jobs` | Override worker count. Values above `1` enable parallel execution. |
| `--limit` | Stop after producing the requested number of scenes. |
| `--split` | Override the split strategy for the run. |
| `--read-split` | Select native dataset partitions when `--split native` is used. |
| `--ratio`, `--gap`, `--segments` | Tune split behavior for compatible split strategies. |
| `--include-map/--no-map` | Force map inclusion on or off for datasets that support maps. |

## Execute the run

When the plan looks right, rerun the same command without `--plan`. If `--force` is omitted, the
CLI prints the summary again and asks for confirmation before processing starts.

After a successful run, check the output directory:

- `manifest.json` at the dataset root describes the produced dataset
- split subdirectories such as `train`, `val`, `test`, or `unsplit` contain the backend-specific
  data files

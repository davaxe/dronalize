# First run (CLI)

<div class="section-intro" markdown="1">
The current Typer application exposes five commands: `available`, `inspect`, `show-config`, `split-support`, and `process`. The shortest reliable workflow is to inspect first, preview the effective plan, and only then execute a job.
</div>

## Inspect the surface

```bash
dronalize available
dronalize inspect a43
dronalize split-support a43
```

These commands are backed directly by the dataset registry and descriptor metadata, so they are the best starting point when you are not sure which splits, map support, or loader capabilities a dataset exposes.

## Preview the effective configuration

```bash
dronalize show-config a43 --config config.toml
```

`show-config` uses the same resolver path as `process`, but it is not a full command-line mirror. It supports config, split, schema, output-format, and worker overrides, but not run-only options such as `--input`, `--output`, `--force`, `--progress`, `--limit`, `--seed`, or `--include-map`.
:
## Plan a processing run


=== ":material-linux: Bash"

    Run the command in Bash using `\` for line continuation:
    
    ```bash
    dronalize process a43 \
        --input data/a43/raw \
        --output data/a43/processed \
        --config config.toml \
        --plan
    ```

=== ":fontawesome-brands-windows: PowerShell"
    
    Run the same command in PowerShell, using backticks  for line continuation:
    
    ```ps1
    dronalize process a43 `
        --input data/a44/raw `
        --output data/a43/processed `
        --config config.toml `
        --plan
    ```

Output from either command should look something like this:

```
This is the processing plan. No changes have been made yet.
Processing Plan
                      ╷
  Dataset             │ a43
  Input directory     │ data\a44\raw
  Output directory    │ data\a43\processed
  Output format       │ mds
  Scene schema        │ positions_velocity_yaw (5 features)
  Source window       │ 20/60 @ 10.0 Hz
  Effective window    │ 39/120 @ 20.0 Hz
  Resampling          │ 2:1 (cubic)
  Filtering           │ cleanup: cleanup_exclude | scene: scene_min_agents | agent: agent_frames, sample_floor
  Map                 │ enabled (circle (radius=60), min_distance=1, interp_distance=2.5)
  Execution           │ parallel (4 workers)
  Split mode          │ shuffled-time
  Time split settings │ segments=8, gap=2 frames
  Output ratio        │ train (70%), val (20%), test (10%)
```


Useful planning-time options in the current CLI:

| Option | Effect |
| --- | --- |
| `--jobs` | Override worker count. Values above `1` enable parallel execution. |
| `--scene-schema` | Change the persisted scene schema. |
| `--output-format` | Choose `mds` or `dummy`. |
| `--include-map/--no-map` | Force map inclusion on or off when supported. |
| `--ratio`, `--gap`, `--segments` | Tune split behavior for compatible split modes. |

The checked-in `config.toml` in the repository root is a real example and can be used with both `show-config` and `process`.

## Execute the plan

When the plan looks right, rerun the same command without `--plan`. If `--force` is omitted, a summary will be printed and a prompt for confirmation will appear before any processing starts.

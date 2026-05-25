# Dataset reference

<div class="section-intro" markdown="1">
This section gives a high-level reference for the datasets supported by `dronalize`. Each page summarizes what a dataset is, the kind of traffic it captures, the expected structure on disk, and where to find the original paper or dataset source.
</div>

## At a glance

The supported datasets span several common trajectory-prediction settings:

- Highway traffic, such as [`a43`](a43.md), [`ad4che`](ad4che.md), [`exid`](exid.md), [`highd`](highd.md), [`i80`](i80.md), and [`us101`](us101.md)
- Urban intersections and mixed traffic, such as [`ind`](ind.md), [`interaction`](interaction.md), [`sind`](sind.md), and [`unid`](unid.md)
- Roundabout interaction, such as [`opendd`](opendd.md) and [`round`](round.md)
- Autonomous-driving urban datasets, such as [`argoverse1`](argoverse1.md), [`argoverse2`](argoverse2.md), [`lyft`](lyft.md), [`nuscenes`](nuscenes.md), [`vod`](vod.md), and [`waymo`](waymo.md)
- Pedestrian forecasting datasets, such as [`eth_ucy`](eth_ucy.md), [`eth`](eth.md), [`hotel`](hotel.md), [`univ`](univ.md), [`zara1`](zara1.md), and [`zara2`](zara2.md)

## How to use this section

- Start with the dataset page if you want a quick sense of whether a benchmark fits your use case.
- Use the default processing profile on each page to see the configured horizon, effective horizon,
  and known source-unit bounds before windowing.
- Check the expected structure before downloading or arranging files locally.
- Use the reference links on each page when you need the original paper, official dataset page, or benchmark source.

## Other information

In addition to this reference, use `dronalize inspect <dataset>` and
`dronalize split-support <dataset>` to get current information from the code.
For example:

```bash
dronalize inspect a43
```
prints the current `a43` defaults, source-sequence bounds, and capabilities:

```text
Dataset inspect: a43
                 ╷                         ╷
  Section        │ Setting                 │ Value
 ════════════════╪═════════════════════════╪══════════════════════════════════════════════
  Dataset        │ Name                    │ a43
                 │ Native schema           │ positions_velocity_acceleration (6 features)
                 │ Schema fields           │ frame, id, x, y, vx, vy, ax, ay, agent_category
                 │ Native splits           │ none
                 │ Read modes              │ all
                 │ Assignment modes        │ none, scene, time, shuffled-time
                 │ Map                     │ yes
                 │ Lane-change sampling    │ no
 ────────────────┼─────────────────────────┼──────────────────────────────────────────────
  Scenes         │ Configured horizon      │ 70 frames (default split after 20) @ 10.0 Hz
                 │ Effective horizon       │ 70 frames (default split after 20) @ 10.0 Hz
                 │ Resampling              │ none
                 │ Sliding windows         │ enabled, strict, step 25
                 │ Screening rules         │ cleanup: min_samples | agent: require_frames
 ────────────────┼─────────────────────────┼──────────────────────────────────────────────
  Temporal       │ Source unit             │ recording
                 │ Source bounds           │ 1353-50814 frames, observed
                 │ Configured horizon fits │ yes (70 <= 50814 frames)
                 │ Windowing default       │ yes
                 │ Supported policies      │ strict (default), anchored, partial
                 │ Window validation       │ error
                 │ Max supported horizon   │ 50814 frames
 ────────────────┼─────────────────────────┼──────────────────────────────────────────────
  Output         │ Schema                  │ canonical
                 │ Precision               │ float32
                 │ Recenter positions      │ yes
 ────────────────┼─────────────────────────┼──────────────────────────────────────────────
  Read           │ Strategy                │ all
                 │ Selection               │ all available inputs
 ────────────────┼─────────────────────────┼──────────────────────────────────────────────
  Assignment     │ Strategy                │ none
                 │ Selection               │ unsplit output
 ────────────────┼─────────────────────────┼──────────────────────────────────────────────
  Map            │ Enabled                 │ yes
                 │ Extraction              │ full map
                 │ Min distance            │ 2
                 │ Interp distance         │ 5
                 │ Edge types              │ all
 ────────────────┼─────────────────────────┼──────────────────────────────────────────────
  Loader options │ rows_per_source         │ 40000
```

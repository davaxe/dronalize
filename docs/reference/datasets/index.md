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
- Check the expected structure before downloading or arranging files locally.
- Use the reference links on each page when you need the original paper, official dataset page, or benchmark source.

## Other information

In addition to this reference, use `dronalize inspect <dataset>` and
`dronalize split-support <dataset>` to get current information from the code.
For example:

```bash
dronalize inspect a43
```
prints the current `a43` defaults and capabilities:

```text
Dataset inspect: a43
  Native schema    │ positions_velocity_acceleration (6 features)
  Native splits    │ none
  Read modes       │ all
  Assignment modes │ none, scene, time, shuffled-time
  Map              │ yes

Default scene settings
  Source window    │ 20/50 @ 10.0 Hz
  Effective window │ 20/50 @ 10.0 Hz
  Windowing        │ 70 frames, step 25
  Screening rules  │ cleanup: min_samples
  Loader options   │ rows_per_source=40000
```

For simple information, prefer the CLI because it is generated from the actual
registry and config models. This reference remains useful for details that the
CLI omits, such as expected dataset structure on disk.

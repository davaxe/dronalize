# INTERACTION

<div class="section-intro" markdown="1">
INTERACTION is a benchmark for difficult multi-agent driving scenarios with strong negotiation behavior. It spans multiple traffic cultures and scene types, making it especially useful when interaction quality matters more than simple lane following.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Interactive</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Benchmark</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                                | Notes                                                                    |
| ------------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| Release year        | 2019                                 | Based on the cited dataset paper and release.                            |
| Domain              | Interactive driving scenarios        | Covers merges, ramps, roundabouts, and intersections.                    |
| Capture platform    | Processed benchmark release          | Released as trajectory files with semantic maps.                         |
| Primary agent types | Cars, cyclists, pedestrians          | Designed around multi-agent interaction rather than one actor class.     |
| Map context         | Semantic maps                        | Map information is a core part of the benchmark.                         |
| Geographic coverage | 18 locations across three continents | Intentionally spans different traffic environments and driving cultures. |
| Data format         | CSV trajectories plus OSM maps       | Structured into training, validation, and test scenario folders.         |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 10 obs / 30 pred @ 10 Hz |
| Effective sequence | 10 obs / 30 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Filtering | Require last observation frame (19) |
| Maps | Full map |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom modes, and any recommended strategy.

```bash
dronalize split-support interact
```

## References

- Dataset paper: [INTERACTION Dataset: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps](https://arxiv.org/abs/1910.03088)

## Expected structure

```text
interact/
├── maps/
│   ├── DR_CHN_Merging_ZS0.osm
│   └── ...
├── train/
├── val/
├── test_multi-agent/
└── test_conditional-multi-agent/
```

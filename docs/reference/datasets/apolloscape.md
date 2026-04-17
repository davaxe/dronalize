# ApolloScape

<div class="section-intro" markdown="1">
ApolloScape is an urban trajectory benchmark built around heterogeneous traffic participants. It is commonly used for interaction-aware forecasting in dense city traffic where vehicles, pedestrians, and riders share space.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                                          | Notes                                                                  |
| ------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- |
| Release year        | 2019                                           | Based on the cited benchmark paper.                                    |
| Domain              | Urban traffic                                  | Built for city-scene trajectory prediction.                            |
| Capture platform    | Processed benchmark release                    | Distributed as trajectory files rather than raw sensor recordings.      |
| Primary agent types | Vehicles, pedestrians, bicycles or motorcycles | Emphasizes heterogeneous traffic.                                      |
| Map context         | None handled by current loader                | The current Dronalize ApolloScape loader does not resolve maps.       |
| Geographic coverage | Urban Chinese road scenes                      | Focused on dense mixed-traffic environments.                           |
| Data format         | Text trajectory files                          | Split into benchmark train and evaluation directories.                 |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 4 obs / 6 pred @ 2 Hz |
| Effective sequence | 16 obs / 30 pred @ 10 Hz |
| Resampling | Cubic 5:1 |
| Windowing | 10-frame window, step 1 |
| Filtering | Prune agents with fewer than 2 samples |
| Maps | Disabled |

## Version

Dronalize does not currently rely on a stable release version marker for ApolloScape. The current loader paths and inspected raw layout do not expose a dataset version beyond the benchmark split directories.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `1` | `CAR` | Numeric category mapping from the ApolloScape trajectory files. |
| `2` | `TRUCK` | Numeric category mapping from the ApolloScape trajectory files. |
| `3` | `PEDESTRIAN` | Numeric category mapping from the ApolloScape trajectory files. |
| `4` | `BICYCLE` | Numeric category mapping from the ApolloScape trajectory files. |
| `5` | `UNKNOWN` | Numeric category mapping from the ApolloScape trajectory files. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| Not applicable | Not applicable | The current Dronalize ApolloScape loader does not resolve maps. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support apolloscape
```

## References

- Dataset paper: [TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents](https://arxiv.org/abs/1811.02146)

## Expected structure

```text
apolloscape/
├── prediction_train/
│   ├── result_9048_1.frame.txt
│   ├── result_9048_3.frame.txt
│   └── ...
├── val_split/
│   └── ...
└── prediction_test/
    └── prediction_test.txt  # present in the raw benchmark, not used by the current loader
```

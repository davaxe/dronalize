# ApolloScape

<div class="section-intro" markdown="1">
ApolloScape is an urban trajectory benchmark built around heterogeneous traffic participants. It is commonly used for interaction-aware forecasting in dense city traffic where vehicles, pedestrians, and riders share space.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2019</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 10 frames (default split after 4) @ 2.0 Hz |
| Effective horizon | 46 frames (default split after 16) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 37-119 frames (observed) |
| Resampling | 5:1 (linear) |
| Sliding windows | Enabled, strict, step 1 |
| Screening | Prune agents with fewer than 2 samples |
| Maps | None |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | ApolloScape trajectory benchmark split layout |
| Loader expectation | The loader follows the benchmark split directories and does not parse a separate upstream version marker. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `1` | `CAR` |
| `2` | `TRUCK` |
| `3` | `PEDESTRIAN` |
| `4` | `BICYCLE` |
| `5` | `UNKNOWN` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| Not applicable | Not applicable |

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
└── prediction_test/
    └── prediction_test.txt
```

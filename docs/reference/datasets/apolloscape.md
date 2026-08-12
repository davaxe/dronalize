# ApolloScape

ApolloScape is an urban trajectory benchmark built around heterogeneous traffic participants. It is commonly used for interaction-aware forecasting in dense city traffic where vehicles, pedestrians, and riders share space.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2019</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default prediction task

The `benchmark` prediction task is selected automatically. In the project config, select it explicitly with `task = "benchmark"` or disable prediction bounds with `task = "none"`.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 10 frames (benchmark origin 4) @ 2.0 Hz |
| Effective horizon | 46 frames (benchmark origin 16) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 37-119 frames (observed) |
| Resampling | 5:1 (linear) |
| Sliding windows | Enabled, strict, step 1 |
| Screening | Prune agents with fewer than 2 observations |
| Maps | None |

## Dataset compatibility

Expected raw data layout for this loader.

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

Use the CLI for current native split and assignment support.

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

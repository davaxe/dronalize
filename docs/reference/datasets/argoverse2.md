# Argoverse 2

<div class="section-intro" markdown="1">
Argoverse 2 expands the original Argoverse benchmark into a broader forecasting and perception dataset family. For motion prediction, it offers larger scale, broader city coverage, and detailed map context across diverse urban driving scenes.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Dataset facts

| Field               | Value                                                           | Notes                                                                  |
| ------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Release year        | 2023                                                            | Based on the cited dataset paper and release.                          |
| Domain              | Autonomous-driving forecasting                                  | Designed for large-scale motion prediction and related tasks.          |
| Capture platform    | Self-driving vehicle fleet                                      | Collected from real urban driving.                                     |
| Primary agent types | Vehicles, pedestrians, cyclists, other traffic participants     | Broader participant coverage than many earlier forecasting benchmarks. |
| Map context         | HD maps                                                         | Rich local road geometry is part of the dataset identity.              |
| Geographic coverage | Austin, Detroit, Miami, Pittsburgh, Palo Alto, Washington, D.C. | Built to span multiple North American cities.                          |
| Data format         | Scenario directories                                            | Organized by train, validation, and test scenario folders.             |

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 50 obs / 60 pred @ 10 Hz |
| Effective sequence | 50 obs / 60 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Filtering | Prune agents with fewer than 2 samples |
| Maps | Full map |

## Version

Dronalize does not currently infer a stable upstream release version for Argoverse 2 from the motion-forecasting directory layout. The loader relies on the standard train/val/test structure rather than on a versioned dataset root.

## Normalization

### Agent categories

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `static` | `STATIC_OBJECT` | Direct category mapping from `object_type`. |
| `riderless_bicycle` | `STATIC_OBJECT` | The current loader groups riderless bicycles with static objects. |
| `construction` | `STATIC_OBJECT` | The current loader groups construction objects with static objects. |
| `vehicle` | `CAR` | Direct category mapping from `object_type`. |
| `motorcyclist` | `MOTORCYCLE` | Direct category mapping from `object_type`. |
| `cyclist` | `BICYCLE` | Direct category mapping from `object_type`. |
| `bus` | `BUS` | Direct category mapping from `object_type`. |
| `pedestrian` | `PEDESTRIAN` | Direct category mapping from `object_type`. |
| `background` | `UNIMPORTANT` | The loader keeps the source distinction but maps it into the shared unimportant category. |
| `unknown` | `UNKNOWN` | Direct category mapping from `object_type`. |

### Map types

| Dataset type | Dronalize type | Notes |
| ------------ | -------------- | ----- |
| `SOLID_WHITE` | `LINE_THIN` | Argoverse 2 lane-boundary mapping in the parser. |
| `SOLID_YELLOW` | `LINE_THIN` | Argoverse 2 lane-boundary mapping in the parser. |
| `DOUBLE_SOLID_WHITE` | `LINE_THIN_DOUBLE` | Argoverse 2 lane-boundary mapping in the parser. |
| `DOUBLE_SOLID_YELLOW` | `LINE_THIN_DOUBLE` | Argoverse 2 lane-boundary mapping in the parser. |
| `DASHED_WHITE` | `LINE_THIN_DASHED` | Argoverse 2 lane-boundary mapping in the parser. |
| `DASHED_YELLOW` | `LINE_THIN_DASHED` | Argoverse 2 lane-boundary mapping in the parser. |
| `DOUBLE_DASH_WHITE` | `LINE_THIN_DOUBLE_DASHED` | Argoverse 2 lane-boundary mapping in the parser. |
| `DOUBLE_DASH_YELLOW` | `LINE_THIN_DOUBLE_DASHED` | Argoverse 2 lane-boundary mapping in the parser. |
| `DASH_SOLID_WHITE` | `LINE_THIN_DASHED` | Mixed dash-solid boundaries are currently normalized to dashed thin lines. |
| `DASH_SOLID_YELLOW` | `LINE_THIN_DASHED` | Mixed dash-solid boundaries are currently normalized to dashed thin lines. |
| `SOLID_DASH_WHITE` | `LINE_THIN_DASHED` | Mixed solid-dash boundaries are currently normalized to dashed thin lines. |
| `SOLID_DASH_YELLOW` | `LINE_THIN_DASHED` | Mixed solid-dash boundaries are currently normalized to dashed thin lines. |
| `SOLID_BLUE` | `LINE_THICK` | Argoverse 2 lane-boundary mapping in the parser. |
| `NONE` | `VIRTUAL` | Parsed as no-marking and then treated as virtual in the builder. |
| `UNKNOWN` | `VIRTUAL` | Parsed as no-marking and then treated as virtual in the builder. |
| Pedestrian crossing edge | `PEDESTRIAN_MARKING` | The builder emits pedestrian crossing edges as pedestrian markings. |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support argoverse2
```

## References

- Dataset paper: [Argoverse 2: Next Generation Datasets for Self-driving Perception and Forecasting](https://arxiv.org/abs/2301.00493)

## Expected structure

```text
argoverse2/
├── train/
│   ├── ...
│   └── ffffe3df-8d26-42c3-9e7a-59de044736a0/
├── val/
│   ├── ...
│   └── fffc6ef5-8fb4-4f20-afea-b9cb63c99182/
└── test/
    ├── ...
    └── fffc1965-9f9e-4822-ade7-750d87c4b7b9/
```

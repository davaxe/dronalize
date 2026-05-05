# Argoverse 2

<div class="section-intro" markdown="1">
Argoverse 2 expands the original Argoverse benchmark into a broader forecasting and perception dataset family. For motion prediction, it offers larger scale, broader city coverage, and detailed map context across diverse urban driving scenes.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Mixed urban</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2023</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>HD</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 50 obs / 60 pred @ 10 Hz |
| Effective sequence | 50 obs / 60 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Screening | Prune agents with fewer than 2 samples |
| Maps | Full map |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | Argoverse 2 motion-forecasting train/val/test layout |
| Loader expectation | The loader uses the standard split directories and does not depend on a versioned dataset root. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `static` | `STATIC_OBJECT` |
| `riderless_bicycle` | `STATIC_OBJECT` |
| `construction` | `STATIC_OBJECT` |
| `vehicle` | `CAR` |
| `motorcyclist` | `MOTORCYCLE` |
| `cyclist` | `BICYCLE` |
| `bus` | `BUS` |
| `pedestrian` | `PEDESTRIAN` |
| `background` | `UNIMPORTANT` |
| `unknown` | `UNKNOWN` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `SOLID_WHITE` | `LINE_THIN` |
| `SOLID_YELLOW` | `LINE_THIN` |
| `DOUBLE_SOLID_WHITE` | `LINE_THIN_DOUBLE` |
| `DOUBLE_SOLID_YELLOW` | `LINE_THIN_DOUBLE` |
| `DASHED_WHITE` | `LINE_THIN_DASHED` |
| `DASHED_YELLOW` | `LINE_THIN_DASHED` |
| `DOUBLE_DASH_WHITE` | `LINE_THIN_DOUBLE_DASHED` |
| `DOUBLE_DASH_YELLOW` | `LINE_THIN_DOUBLE_DASHED` |
| `DASH_SOLID_WHITE` | `LINE_THIN_DASHED` |
| `DASH_SOLID_YELLOW` | `LINE_THIN_DASHED` |
| `SOLID_DASH_WHITE` | `LINE_THIN_DASHED` |
| `SOLID_DASH_YELLOW` | `LINE_THIN_DASHED` |
| `SOLID_BLUE` | `LINE_THICK` |
| `NONE` | `VIRTUAL` |
| `UNKNOWN` | `VIRTUAL` |
| Pedestrian crossing edge | `PEDESTRIAN_MARKING` |

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

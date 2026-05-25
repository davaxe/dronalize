# Zara2

<div class="section-intro" markdown="1">
Zara2 is a pedestrian-only scene from the ETH/UCY benchmark family. It is typically used in socially aware forecasting benchmarks that compare how models handle dense human interaction across different scenes.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2007</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Pedestrians</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 20 frames (default split after 8) @ 2.5 Hz |
| Effective horizon | 77 frames (default split after 29) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 89-1440 frames (observed) |
| Resampling | 4:1 (linear) |
| Sliding windows | Enabled, strict, step 1 |
| Screening | Keep agents with at least 2 samples |
| Maps | None |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | UCY ZARA2 scene layout |
| Loader expectation | The loader uses the ETH/UCY raw text layout and does not parse a dataset-specific version marker. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| Any tracked actor | `PEDESTRIAN` |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| Not applicable | Not applicable |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support zara2
```

## References

- Family reference: [You'll never walk alone: Modeling social behavior for multi-target tracking](https://ieeexplore.ieee.org/document/5459260)

## Expected structure

```text
zara2/
├── train/
├── val/
└── test/
```

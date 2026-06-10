# Hotel

Hotel is a pedestrian-only scene from the ETH/UCY trajectory benchmark family. It is typically used for socially aware human motion forecasting in open public spaces with interacting walkers.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2007</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Pedestrians</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default processing profile

Default processing settings for this dataset.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 20 frames (default split after 8) @ 2.5 Hz |
| Effective horizon | 77 frames (default split after 29) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 89-1807 frames (observed) |
| Resampling | 4:1 (linear) |
| Sliding windows | Enabled, strict, step 1 |
| Screening | Keep agents with at least 2 observations |
| Maps | None |

## Dataset compatibility

Expected raw data layout for this loader.

| Field | Value |
| ----- | ----- |
| Expected release/layout | UCY HOTEL scene layout |
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

Use the CLI for current native split and assignment support.

```bash
dronalize split-support hotel
```

## References

- Family reference: [You'll never walk alone: Modeling social behavior for multi-target tracking](https://ieeexplore.ieee.org/document/5459260)

## Expected structure

```text
hotel/
├── train/
├── val/
└── test/
```

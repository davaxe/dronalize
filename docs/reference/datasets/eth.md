# ETH

ETH is one of the classic pedestrian trajectory benchmarks used in socially aware forecasting. It is usually discussed together with the UCY scenes and is valued for crowd interaction studies rather than map-rich vehicle forecasting.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2009</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Pedestrians</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default prediction task

The `benchmark` prediction task is selected automatically. In the project config, select it explicitly with `task = "benchmark"` or disable prediction bounds with `task = "none"`.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 20 frames (benchmark origin 8) @ 2.5 Hz |
| Effective horizon | 77 frames (benchmark origin 29) @ 10.0 Hz |
| Source unit | Recording |
| Source bounds | 89-1440 frames (observed) |
| Resampling | 4:1 (linear) |
| Sliding windows | Enabled, strict, step 1 |
| Screening | Keep agents with at least 2 observations |
| Maps | None |

## Dataset compatibility

Expected raw data layout for this loader.

| Field | Value |
| ----- | ----- |
| Expected release/layout | ETH BIWI pedestrian scene layout |
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
dronalize split-support eth
```

## References

- Dataset paper: [You'll never walk alone: Modeling social behavior for multi-target tracking](https://ieeexplore.ieee.org/document/5459260)

## Expected structure

```text
eth/
├── train/
├── val/
└── test/
```

## Notes

- The ETH and UCY scenes are often used as a shared benchmark family, even when individual scenes are reported separately in the literature.

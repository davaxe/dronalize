# ETH

<div class="section-intro" markdown="1">
ETH is one of the classic pedestrian trajectory benchmarks used in socially aware forecasting. It is usually discussed together with the UCY scenes and is valued for crowd interaction studies rather than map-rich vehicle forecasting.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2009</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Pedestrians</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Camera</strong></div>
  <div class="summary-item"><span>Map context</span><strong>None</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 8 obs / 12 pred @ 2.5 Hz |
| Effective sequence | 29 obs / 48 pred @ 10 Hz |
| Resampling | Linear 4:1 |
| Windowing | 20-frame window, step 1 |
| Filtering | Keep agents with at least 2 samples |
| Maps | Disabled |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

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

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

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

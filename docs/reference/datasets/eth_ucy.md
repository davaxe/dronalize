# ETH/UCY

<div class="section-intro" markdown="1">
ETH/UCY is the combined pedestrian trajectory benchmark family. This dataset key is useful when
you want to process all ETH, Hotel, Univ, Zara1, and Zara2 subdatasets as one input collection.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Pedestrian</strong></div>
  <div class="summary-item"><span>Release years</span><strong>2007-2009</strong></div>
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
| Screening | Keep agents with at least 2 samples |
| Maps | Disabled |

## Dataset compatibility

Dronalize targets the ETH/UCY raw text layout and recursively discovers native split directories.
If the root contains multiple `train`, `val`, or `test` directories, files from matching split
directories are combined for that native split.

| Field | Value |
| ----- | ----- |
| Expected release/layout | Combined ETH/UCY pedestrian scene layout |
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
dronalize split-support eth_ucy
```

## References

- ETH paper: [You'll never walk alone: Modeling social behavior for multi-target tracking](https://ieeexplore.ieee.org/document/5459260)

## Expected structure

```text
eth_ucy/
├── eth/
│   ├── train/
│   ├── val/
│   └── test/
├── hotel/
│   ├── train/
│   ├── val/
│   └── test/
├── univ/
│   ├── train/
│   ├── val/
│   └── test/
├── zara1/
│   ├── train/
│   ├── val/
│   └── test/
└── zara2/
    ├── train/
    ├── val/
    └── test/
```

## Notes

- The individual dataset keys `eth`, `hotel`, `univ`, `zara1`, and `zara2` remain available when you want to process one scene family member independently.
- If a certain subset of the combined dataset is more relevant to your use case, you can remove the irrelevant files. For example, if you only care about the `eth` and `hotel` scenes, you can remove the `univ`, `zara1`, and `zara2` directories from the root before processing.

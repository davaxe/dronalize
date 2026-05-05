# INTERACTION

<div class="section-intro" markdown="1">
INTERACTION is a benchmark for difficult multi-agent driving scenarios with strong negotiation behavior. It spans multiple traffic cultures and scene types, making it especially useful when interaction quality matters more than simple lane following.
</div>

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Interactive</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2019</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Provided</strong></div>
  <div class="summary-item"><span># Samples</span><strong>Processed samples planned</strong></div>
</div>

## Default processing profile

These are the default Dronalize settings used when processing this dataset.

| Setting | Default |
| ------- | ------- |
| Source sequence | 10 obs / 30 pred @ 10 Hz |
| Effective sequence | 10 obs / 30 pred @ 10 Hz |
| Resampling | None |
| Windowing | None |
| Screening | Prune agents with fewer than 2 samples |
| Maps | Disabled |

## Dataset compatibility

Dronalize targets the release or raw layout below. If you have an older or newer download, expect breakage when split names, file names, schemas, or map assets differ.

| Field | Value |
| ----- | ----- |
| Expected release/layout | INTERACTION DR-multi v1.2 |
| Loader expectation | The loader expects the multi-agent v1.2 archive layout and split naming. |

## Normalization

### Agent categories

| Dataset type | Dronalize type |
| ------------ | -------------- |
| `car` | `CAR` |
| `pedestrian/bicycle` with speed `< 2 m/s` | `PEDESTRIAN` |
| `pedestrian/bicycle` with speed `>= 2 m/s` | `BICYCLE` |
| Any other `agent_type` | Unchanged source value |

### Map types

| Dataset type | Dronalize type |
| ------------ | -------------- |
| Not applicable | Not applicable |

## Split support

Use the command below for the most up-to-date split support information for this dataset, including native splits, supported custom split strategies, and any recommended strategy.

```bash
dronalize split-support interaction
```

## References

- Dataset paper: [INTERACTION Dataset: An INTERnational, Adversarial and Cooperative moTION Dataset in Interactive Driving Scenarios with Semantic Maps](https://arxiv.org/abs/1910.03088)

## Expected structure

```text
interaction/
├── maps/
│   ├── DR_CHN_Merging_ZS0.osm
│   └── ...
├── train/
├── val/
├── test_multi-agent/
└── test_conditional-multi-agent/
```

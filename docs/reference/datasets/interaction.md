# INTERACTION

INTERACTION is a benchmark for difficult multi-agent driving scenarios with strong negotiation behavior. It spans multiple traffic cultures and scene types, making it especially useful when interaction quality matters more than simple lane following.

<div class="summary-grid">
  <div class="summary-item"><span>Domain</span><strong>Interactive</strong></div>
  <div class="summary-item"><span>Release year</span><strong>2019</strong></div>
  <div class="summary-item"><span>Primary agents</span><strong>Mixed</strong></div>
  <div class="summary-item"><span>Capture platform</span><strong>Vehicle</strong></div>
  <div class="summary-item"><span>Map context</span><strong>Provided</strong></div>
  <div class="summary-item"><span># Records</span><strong>processed scene records planned</strong></div>
</div>

## Default processing profile

Default processing settings for this dataset.

| Setting | Default |
| ------- | ------- |
| Configured horizon | 40 frames (default split after 10) @ 10.0 Hz |
| Effective horizon | 40 frames (default split after 10) @ 10.0 Hz |
| Source unit | Case |
| Source bounds | 10-40 frames (observed) |
| Resampling | None |
| Sliding windows | Disabled |
| Screening | Prune agents with fewer than 2 observations |
| Maps | Full map |

## Dataset compatibility

Expected raw data layout for this loader.

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

Use the CLI for current native split and assignment support.

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

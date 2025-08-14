# Modeling

This page provides an overview of the modeling components in the **Dronalize** toolbox, including a simple prototype model and the more advanced [MTP-GO](https://arxiv.org/abs/2302.00735) architecture.


## Prototype Model

The prototype model, located in [`models/prototype/model.py`](models/prototype/model.py), serves as a lightweight baseline for trajectory prediction. It is a simple encoder–decoder architecture that:

- Takes as input the past trajectory of an agent  
- Encodes the temporal information using a gated recurrent unit (GRU)  
- Models agent interactions using a graph neural network (GATv2Conv)  
- Decodes the hidden state over a fixed prediction horizon  

This model provides a clean starting point for development and experimentation. For example, adding map-awareness or richer interaction modeling would be a natural next step.

```python
# model.py
import torch
import torch.nn as nn
import torch_geometric.nn as ptg
from torch_geometric.data import HeteroData

class Net(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_inputs = config["num_inputs"]
        num_outputs = config["num_outputs"]
        num_hidden = config["num_hidden"]
        self.ph = config["pred_hrz"]

        self.embed = nn.Linear(num_inputs, num_hidden)
        self.encoder = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.interaction = ptg.GATv2Conv(num_hidden, num_hidden, concat=False)
        self.decoder = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.output = nn.Linear(num_hidden, num_outputs)

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_index = data['agent']['edge_index']
        x = torch.cat([data['agent']['inp_pos'],
                       data['agent']['inp_vel'],
                       data['agent']['inp_yaw']], dim=-1)

        x = self.embed(x)
        _, h = self.encoder(x)
        x = h[-1]

        x = self.interaction(x, edge_index)
        x = x.unsqueeze(1).repeat(1, self.ph, 1)
        x, _ = self.decoder(x)

        return self.output(x)
```

In [models/prototype/litmodule.py](models/prototype/litmodule.py), there is a PyTorch Lightning module that wraps the model and provides
training and evaluation logic.

<br>

## MTP-GO: *Multi-agent Trajectory Prediction by Graph-enhanced neural ODEs*

<div align="center">
  <img alt="Schematics of MTP-GO" src=https://github.com/westny/mtp-go/assets/60364134/4f30cf04-db78-470c-aae0-c50c468afe04 width="800px" style="max-width: 100%;">
</div>


[`models/mtp_go/model.py`](models/mtp_go/model.py) contains the implementation of **MTP-GO**, an advanced trajectory prediction model introduced in the [IEEE T-IV paper](https://arxiv.org/abs/2302.00735). The model features:

- Temporal scene encoding using graph-gated recurrent units (Graph-GRUs)
- A differentially-constrained motion model implemented using neural ODEs
- Multimodal probabilistic predictions via Mixture Density Networks (MDNs) and EKF-based uncertainty propagation

This implementation has been adapted to work with the Dronalize toolbox. While the original model was designed without lane graphs, it relies on **contextual features** to encode road geometry and topology. These features can be automatically generated for the `rounD`, `inD`, and `highD` datasets using the `--add-supp` flag. See the paper and preprocessing docs for further details.

Training and evaluation are handled via a corresponding PyTorch Lightning module in [`models/mtp_go/litmodule.py`](models/mtp_go/litmodule.py).

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    Enabling resilient autonomous motion planning requires robust predictions of surrounding road users' future behavior.
    In response to this need and the associated challenges, we introduce our model titled MTP-GO.
    The model encodes the scene using temporal graph neural networks to produce the inputs to an underlying motion model.
    The motion model is implemented using neural ordinary differential equations where the state-transition functions are learned with the rest of the model.
    Multimodal probabilistic predictions are obtained by combining the concept of mixture density networks and Kalman filtering.
    The results illustrate the predictive capabilities of the proposed model across various data sets, outperforming several state-of-the-art methods on a number of metrics.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

```bibtex
    @article{westny2023mtp,
      title="{MTP-GO}: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural {ODEs}",
      author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
      journal={IEEE Transactions on Intelligent Vehicles},
      year={2023},
      volume={8},
      number={9},
      pages={4223-4236},
      doi={10.1109/TIV.2023.3282308}
    }
```

</details>

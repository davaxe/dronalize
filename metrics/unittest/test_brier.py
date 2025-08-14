import re

import pytest
import torch

from metrics.min_brier import MinBrier


def test_min_brier_multimodal() -> None:
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    prob = torch.randn(batch_size, num_modes)
    mask = torch.randint(0, 2, (batch_size, seq_len))

    min_brier = MinBrier()
    for msk in [None, mask]:
        min_brier.update(pred, trg, prob, msk)
        min_brier.compute()


def test_min_brier_with_none_probability():
    batch_size, seq_len, num_dims = 32, 25, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_dims)
    prob = None

    min_brier = MinBrier()
    with pytest.raises(ValueError, match="Probabilistic criterion requires"
                                             " the probability of the predictions."):
        min_brier.update(pred, trg, prob)


def test_min_brier_with_unimodal_prediction():
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_dims)  # Assuming this should be multi-modal
    prob = torch.randn(batch_size, num_modes)

    min_brier = MinBrier()
    msg = f"The prediction tensor must be 4-dimensional, got shape {pred.shape}"
    with pytest.raises(ValueError, match=re.escape(msg)):
        min_brier.update(pred, trg, prob)

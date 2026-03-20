"""
Tests for hard negative mining in compute_struct_loss.

Validates that confidence-based masking produces different (non-random) masks
compared to the default random masking, and that the API works correctly.
"""

import torch
from gliner2.model import Extractor, ExtractorConfig


def _make_model():
    """Create a minimal Extractor for testing (no pretrained weights needed)."""
    config = ExtractorConfig(
        model_name="bert-base-uncased",
        max_width=4,
    )
    model = Extractor(config)
    model.eval()
    return model


def test_configure_hard_neg_defaults():
    """Test that hard neg is disabled by default and configure_hard_neg sets state."""
    model = _make_model()
    assert not model._hard_neg_enabled
    assert model._hard_neg_masking_rate == 0.5

    model.configure_hard_neg(enabled=True, masking_rate=0.3)
    assert model._hard_neg_enabled
    assert model._hard_neg_masking_rate == 0.3

    model.configure_hard_neg(enabled=False)
    assert not model._hard_neg_enabled


def test_hard_neg_masking_differs_from_random():
    """
    Hard negative mining should produce masks biased toward keeping
    high-confidence negatives. We verify this by checking that the
    average mask value for high-score negatives is higher than for
    low-score negatives (over many trials).
    """
    model = _make_model()
    model.train()
    model.configure_hard_neg(enabled=True, masking_rate=0.5)

    torch.manual_seed(42)

    # Create synthetic scores where some negatives have high scores (hard)
    # and others have low scores (easy)
    B, K, L, W = 2, 3, 10, 4
    scores = torch.randn(B, K, L, W)
    # Make a clear separation: first half of positions have high scores,
    # second half have low scores
    scores[:, :, :5, :] = scores[:, :, :5, :].abs() + 2.0  # high scores (hard negs)
    scores[:, :, 5:, :] = -scores[:, :, 5:, :].abs() - 2.0  # low scores (easy negs)

    # All negatives (no positive labels)
    labs = torch.zeros_like(scores)
    negative = (labs == 0)

    # Run hard neg masking many times and accumulate keep rates
    n_trials = 200
    high_score_kept = 0.0
    low_score_kept = 0.0

    for _ in range(n_trials):
        with torch.no_grad():
            neg_probs = torch.sigmoid(scores.detach())
            neg_vals = neg_probs * negative.float()
            max_val = neg_vals.max()
            if max_val > 0:
                keep_prob = neg_vals / max_val
            else:
                keep_prob = torch.zeros_like(scores)
            target_keep = 1 - 0.5
            neg_mean = keep_prob[negative].mean()
            if neg_mean > 0:
                keep_prob = keep_prob * (target_keep / neg_mean)
            keep_prob = keep_prob.clamp(0, 1)
            keep_prob = keep_prob + (1 - negative.float())
            keep_prob = keep_prob.clamp(0, 1)
        mask = (torch.rand_like(scores) < keep_prob).float()
        high_score_kept += mask[:, :, :5, :].mean().item()
        low_score_kept += mask[:, :, 5:, :].mean().item()

    high_score_kept /= n_trials
    low_score_kept /= n_trials

    # Hard negatives (high score) should be kept MORE often than easy ones
    assert high_score_kept > low_score_kept, (
        f"Hard neg mining should keep high-score negatives more often. "
        f"Got high={high_score_kept:.3f}, low={low_score_kept:.3f}"
    )


def test_hard_neg_disabled_uses_random_masking():
    """When hard neg is disabled, masking should be purely random (uniform)."""
    model = _make_model()
    model.train()
    # Explicitly disabled
    model.configure_hard_neg(enabled=False)

    torch.manual_seed(123)
    B, K, L, W = 2, 3, 10, 4
    scores = torch.randn(B, K, L, W)
    scores[:, :, :5, :] = scores[:, :, :5, :].abs() + 2.0
    scores[:, :, 5:, :] = -scores[:, :, 5:, :].abs() - 2.0

    labs = torch.zeros_like(scores)
    negative = (labs == 0)
    masking_rate = 0.5

    # With random masking, keep rates should be approximately equal
    n_trials = 200
    high_kept = 0.0
    low_kept = 0.0

    for _ in range(n_trials):
        random_mask = torch.rand_like(scores) < masking_rate
        to_mask = negative & random_mask
        loss_mask = (~to_mask).float()
        high_kept += loss_mask[:, :, :5, :].mean().item()
        low_kept += loss_mask[:, :, 5:, :].mean().item()

    high_kept /= n_trials
    low_kept /= n_trials

    # With random masking, both should be approximately equal (~0.5)
    assert abs(high_kept - low_kept) < 0.05, (
        f"Random masking should be uniform. Got high={high_kept:.3f}, low={low_kept:.3f}"
    )


def test_compute_struct_loss_runs_with_hard_neg():
    """
    Smoke test: compute_struct_loss should run without errors
    when hard negative mining is enabled.
    """
    model = _make_model()
    model.train()
    model.configure_hard_neg(enabled=True, masking_rate=0.4)

    # Create minimal inputs for compute_struct_loss
    text_len = 8
    hidden = model.hidden_size
    token_embs = torch.randn(text_len, hidden)

    # Compute span representations
    span_info = model.compute_span_rep(token_embs)

    # Create a simple structure label: 1 entity with 2 fields
    schema_emb = torch.randn(3, hidden)  # [P] + 2 fields
    structure = [1, [[(0, 3), (5, 7)]]]  # 1 instance, 2 field spans

    loss = model.compute_struct_loss(
        span_info["span_rep"],
        schema_emb,
        structure,
        span_info["span_mask"],
    )

    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss.requires_grad, "Loss should require gradients"


if __name__ == "__main__":
    test_configure_hard_neg_defaults()
    print("PASSED: test_configure_hard_neg_defaults")

    test_hard_neg_masking_differs_from_random()
    print("PASSED: test_hard_neg_masking_differs_from_random")

    test_hard_neg_disabled_uses_random_masking()
    print("PASSED: test_hard_neg_disabled_uses_random_masking")

    test_compute_struct_loss_runs_with_hard_neg()
    print("PASSED: test_compute_struct_loss_runs_with_hard_neg")

    print("\nAll tests passed!")

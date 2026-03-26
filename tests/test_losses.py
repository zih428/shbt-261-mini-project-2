import torch

from vocseg.training.losses import CombinedSegmentationLoss


def test_combined_loss_backward_with_noncontiguous_logits():
    base_logits = torch.randn(2, 21, 16, 16, requires_grad=True)
    logits = base_logits.transpose(2, 3)
    target = torch.randint(0, 21, (2, 16, 16)).transpose(1, 2)
    loss_fn = CombinedSegmentationLoss(ignore_index=255)
    loss = loss_fn(logits, target)
    loss.backward()
    assert base_logits.grad is not None

"""Actual forward/backward coverage of each OCR-free trainable model arm."""
from pathlib import Path

import pytest
import torch

from cgis.context import context_contrastive_loss
from cgis.losses import DetectionCriterion
from cgis.models import ContextHead, build_model


torch.set_num_threads(2)


def _records():
    return [{'boxes': torch.tensor([[.25, .25, .12, .1], [.7, .65, .12, .1]]),
             'labels': torch.tensor([0, 1]), 'weights': torch.tensor([.9, .95]),
             'ignore_boxes': torch.empty(0, 4),
             'ignore_labels': torch.empty(0, dtype=torch.long),
             'valid_box': torch.tensor([0., 0., 128., 128.])}]


@pytest.mark.parametrize('arm', ['detail', 'context', 'combined'])
def test_actual_ocr_free_backward_updates_each_enabled_module(arm):
    model, report = build_model(arm, load_pretrained=False)
    model.train()
    criterion = DetectionCriterion(model)
    output = model(torch.rand(1, 3, 128, 128), _records(), epoch=6)
    assert report['uses_ocr'] is False
    assert 'role_pair_logits' not in output
    losses = criterion(output, _records(), epoch=6)
    assert all(torch.isfinite(loss) and loss.ndim == 0 for loss in losses.values())
    sum(losses.values()).backward()
    if arm in ('context', 'combined'):
        assert model.context_head.class_delta.weight.grad.abs().sum() > 0
        assert model.context_head.embedding[-1].weight.grad.abs().sum() > 0
        assert criterion.last_diagnostics['context']['contrastive_anchors'] == 4
        assert float(losses['loss_context_contrast']) > 0
    else:
        assert 'loss_context_contrast' not in losses
    if arm in ('detail', 'combined'):
        assert model.refiner.class_delta.weight.grad.abs().sum() > 0
        assert model.gain_head[-1].weight.grad.abs().sum() > 0
        assert not output['gain_targets'].requires_grad
        assert output['gain_targets'].shape == (1, 300)
        model.eval()
        with torch.no_grad():
            evaluation = model(torch.rand(1, 3, 128, 128), _records(), epoch=6)
        assert evaluation['selected_indices'].shape == (1, 64)
        assert model.refiner.last_roi_count == 64
    else:
        assert 'loss_gain' not in losses and 'fine_outputs' not in output


def test_context_residual_can_increase_or_decrease_but_is_bounded():
    head = ContextHead().eval()
    feature = torch.rand(1, 256, 16, 16)
    boxes = torch.tensor([[[.5, .5, .2, .2]]])
    rectangle = torch.tensor([[0., 0., 128., 128.]])
    with torch.no_grad():
        initial, _ = head(feature, boxes, rectangle, (128, 128))
        assert torch.equal(initial, torch.zeros_like(initial))
        head.class_delta.bias.copy_(torch.tensor([20., -20., 1., -1.]))
        residual, _ = head(feature, boxes, rectangle, (128, 128))
    assert residual[0, 0, 0] > 0 and residual[0, 0, 1] < 0
    assert residual.abs().max() <= 2


def test_hard_negative_geometry_produces_finite_differentiable_loss():
    embedding = torch.randn(2, 4, 64, requires_grad=True)
    loss, diagnostic = context_contrastive_loss({
        'embeddings': embedding, 'labels': torch.tensor([0, 1, 2, 3]),
        'weights': torch.tensor([1., .9, .8, .75]),
        'boxes': torch.tensor([[.2, .2, .1, .1], [.7, .2, .1, .1],
                               [.2, .7, .11, .12], [.7, .7, .12, .1]])})
    assert torch.isfinite(loss) and loss > 0
    assert diagnostic['hard_negative_pairs'] > 0
    loss.backward()
    assert embedding.grad.abs().sum() > 0


def test_empty_target_context_is_trainable_without_inventing_labels():
    model, _ = build_model('context', load_pretrained=False)
    model.train()
    target = _records()[0]
    target.update(boxes=torch.empty(0, 4), labels=torch.empty(0, dtype=torch.long), weights=torch.empty(0))
    outputs = model(torch.rand(1, 3, 128, 128), [target])
    losses = DetectionCriterion()(outputs, [target])
    assert float(losses['loss_context_contrast']) == 0
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert model.context_head.class_delta.weight.grad is not None

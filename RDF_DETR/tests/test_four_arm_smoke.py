"""Exercise the four published arms without data, pretrained weights or GPU."""
import pytest
import torch

from cgis.losses import DetectionCriterion
from cgis.models import build_model

torch.set_num_threads(2)


@pytest.mark.parametrize('arm', ['baseline', 'detail', 'context', 'combined'])
def test_four_arm_synthetic_training_and_inference(arm):
    torch.manual_seed(20261015)
    model, report = build_model(arm, load_pretrained=False)
    target = {'boxes': torch.tensor([[.25, .25, .12, .1], [.7, .65, .12, .1]]),
              'labels': torch.tensor([0, 1]), 'weights': torch.tensor([.9, .95]),
              'ignore_boxes': torch.empty(0, 4), 'ignore_labels': torch.empty(0, dtype=torch.long),
              'valid_box': torch.tensor([0., 0., 128., 128.])}
    image = torch.rand(1, 3, 128, 128)
    model.train()
    output = model(image, [target], epoch=16, profile='full')
    losses = DetectionCriterion(model)(output, [target], epoch=16, profile='full')
    assert report['test_only_random_initialization']
    assert all(torch.isfinite(value).item() for value in losses.values())
    sum(losses.values()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        prediction = model(image, [target], epoch=16, profile='full')
    assert prediction['pred_logits'].shape == (1, 300, 4)
    assert prediction['pred_boxes'].shape == (1, 300, 4)

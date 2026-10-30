import torch

from cgis.losses import DetectionCriterion
from vendor.rtdetr_runtime.rtdetr_criterion import SetCriterion


def target(boxes=None, labels=None, weights=None, ignored=None, ignored_labels=None):
    return {'boxes': torch.as_tensor(boxes if boxes is not None else [], dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels if labels is not None else [], dtype=torch.long),
            'weights': torch.as_tensor(weights if weights is not None else [1.] * len(labels or []), dtype=torch.float32),
            'ignore_boxes': torch.as_tensor(ignored if ignored is not None else [], dtype=torch.float32).reshape(-1, 4),
            'ignore_labels': torch.as_tensor(ignored_labels if ignored_labels is not None else [], dtype=torch.long)}


def test_no_ignore_unit_weight_loss_matches_upstream_criterion():
    criterion = DetectionCriterion()
    outputs = {'pred_logits': torch.tensor([[[1., -1., -2., -3.], [-2., 1., -1., -3.]]], requires_grad=True),
               'pred_boxes': torch.tensor([[[.3, .3, .19, .2], [.7, .7, .2, .19]]], requires_grad=True)}
    targets = [target([[.3, .3, .2, .2], [.7, .7, .2, .2]], [0, 1])]
    expected = SetCriterion(criterion.matcher, {'loss_vfl': 1., 'loss_bbox': 5., 'loss_giou': 2.},
                            ['vfl', 'boxes'], alpha=.75, gamma=2., num_classes=4)(outputs, targets)
    actual = criterion(outputs, targets)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.allclose(actual[key], expected[key], atol=1e-6), (key, actual[key], expected[key])


def test_unmatched_same_class_ignore_and_positive_precedence_gradients():
    criterion = DetectionCriterion()
    logits = torch.tensor([[[2., 1., 1., 1.], [1., 2., 1., 1.]]], requires_grad=True)
    boxes = torch.tensor([[[.3, .3, .2, .2], [.7, .7, .2, .2]]], requires_grad=True)
    outputs = {'pred_logits': logits, 'pred_boxes': boxes}
    targets = [target([[.3, .3, .2, .2]], [0], ignored=[[.3, .3, .2, .2], [.7, .7, .2, .2]], ignored_labels=[0, 1])]
    losses = criterion(outputs, targets)
    sum(losses.values()).backward()
    assert logits.grad[0, 0, 0].abs() > 0  # Positive wins even inside an ignore.
    assert logits.grad[0, 1, 1] == 0       # Same-category unmatched background is masked.
    assert logits.grad[0, 1, 0].abs() > 0  # Other categories remain supervised.


def test_empty_targets_and_all_ignore_are_finite():
    criterion = DetectionCriterion()
    outputs = {'pred_logits': torch.zeros(1, 1, 4, requires_grad=True),
               'pred_boxes': torch.tensor([[[.5, .5, .2, .2]]], requires_grad=True)}
    record = target(ignored=[[.5, .5, .2, .2]] * 4, ignored_labels=[0, 1, 2, 3])
    losses = criterion(outputs, [record])
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert outputs['pred_logits'].grad.abs().sum() == 0


def test_pseudo_weights_attenuate_positive_and_box_losses():
    criterion = DetectionCriterion()
    outputs = {'pred_logits': torch.tensor([[[0., -10., -10., -10.]]]),
               'pred_boxes': torch.tensor([[[.5, .5, .18, .18]]])}
    strong = criterion(outputs, [target([[.5, .5, .2, .2]], [0], [1.])])
    weak = criterion(outputs, [target([[.5, .5, .2, .2]], [0], [.4])])
    assert torch.allclose(weak['loss_bbox'], strong['loss_bbox'] * .4)
    assert torch.allclose(weak['loss_giou'], strong['loss_giou'] * .4)
    assert weak['loss_vfl'] < strong['loss_vfl']


def test_gain_target_is_signed_detached_and_uses_one_coarse_assignment():
    criterion = DetectionCriterion()
    record = target([[.3, .3, .2, .2]], [0])
    coarse = {'pred_logits': torch.tensor([[[0., -3., -3., -3.], [-3., -3., -3., -3.]]], requires_grad=True),
              'pred_boxes': torch.tensor([[[.35, .3, .2, .2], [.7, .7, .2, .2]]], requires_grad=True)}
    fine = {'pred_logits': torch.tensor([[[3., -3., -3., -3.], [3., -3., -3., -3.]]], requires_grad=True),
            'pred_boxes': torch.tensor([[[.3, .3, .2, .2], [.7, .7, .2, .2]]], requires_grad=True)}
    indices = criterion.match(coarse, [record])
    gain, matched, active = criterion.utility_targets(coarse, fine, [record], indices)
    assert gain[0, 0] > 0 and gain[0, 1] < 0
    assert not gain.requires_grad
    assert matched.tolist() == [[True, False]]
    assert active.all()

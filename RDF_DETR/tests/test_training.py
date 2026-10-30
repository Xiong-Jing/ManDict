import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cgis.common import save_checkpoint, seed_everything
from cgis.training import (StepETA, accumulation_divisor, build_optimizer, build_scheduler,
                           capture_rng, restore_rng, selection_key, should_validate, train_epoch)


class SmallDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
        return torch.tensor([index / 5 + noise / 10], dtype=torch.float32), {'value': torch.tensor([index / 5])}


class SmallDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(1, 4)
        self.dropout = nn.Dropout(.2)
        self.head = nn.Linear(4, 1)

    def forward(self, images, targets=None, **_kwargs):
        return {'prediction': self.head(self.dropout(self.backbone(images)))}


class SmallCriterion(nn.Module):
    def forward(self, outputs, targets, **_kwargs):
        expected = torch.stack([item['value'] for item in targets])
        return {'loss_regression': (outputs['prediction'] - expected).square().mean()}


def config():
    return {'profile': 'pilot', 'epochs': 2, 'validation_interval': 1, 'accumulation_steps': 2,
            'clip_grad_norm': 1., 'lr': 1e-3, 'backbone_lr': 1e-4, 'weight_decay': 1e-4,
            'warmup_epochs': 1, 'role_warmup_epochs': 3, 'utility_warmup_end': 5}


def loader(generator):
    return DataLoader(SmallDataset(), batch_size=2, shuffle=True, generator=generator,
                      collate_fn=lambda items: (torch.stack([x for x, _ in items]), [t for _, t in items]))


def test_actual_training_optimizer_scheduler_rng_recovery(tmp_path):
    """A resumed epoch matches continuous training even with dropout and three RNGs."""
    seed_everything(741)
    model = SmallDetector()
    optimizer = build_optimizer(model, config())
    scheduler = build_scheduler(optimizer, config())
    generator = torch.Generator().manual_seed(224)
    criterion = SmallCriterion()
    train_epoch(model, criterion, loader(generator), optimizer, torch.device('cpu'), config(),
                1, 'baseline', precision='float32')
    scheduler.step()
    checkpoint = {'model': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
                  'scheduler': copy.deepcopy(scheduler.state_dict()), 'rng': capture_rng(generator)}
    save_checkpoint(tmp_path / 'last.pt', checkpoint)
    expected_stats = train_epoch(model, criterion, loader(generator), optimizer, torch.device('cpu'), config(),
                                 2, 'baseline', precision='float32')
    expected = copy.deepcopy(model.state_dict())
    restored = SmallDetector()
    restored_optimizer = build_optimizer(restored, config())
    restored_scheduler = build_scheduler(restored_optimizer, config())
    saved = torch.load(tmp_path / 'last.pt', map_location='cpu')
    restored.load_state_dict(saved['model'])
    restored_optimizer.load_state_dict(saved['optimizer'])
    restored_scheduler.load_state_dict(saved['scheduler'])
    restored_generator = torch.Generator()
    restore_rng(saved['rng'], restored_generator)
    actual_stats = train_epoch(restored, criterion, loader(restored_generator), restored_optimizer,
                               torch.device('cpu'), config(), 2, 'baseline', precision='float32')
    assert expected_stats['loss'] == actual_stats['loss']
    assert expected_stats['optimizer_steps'] == actual_stats['optimizer_steps'] == 2
    for name in expected:
        assert torch.equal(expected[name], restored.state_dict()[name]), name
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_short_accumulation_group_uses_its_own_sample_mean():
    assert [accumulation_divisor(i, 5, 4) for i in range(5)] == [4, 4, 4, 4, 1]
    assert [accumulation_divisor(i, 2, 4) for i in range(2)] == [2, 2]


def test_eta_excludes_warmup_and_retains_recent_speed():
    eta = StepETA(warmup=2, window=2)
    eta.update(100.)
    eta.update(100.)
    assert eta.remaining(10) is None
    eta.update(4.)
    eta.update(6.)
    assert eta.remaining(10) == 50.
    eta.update(2.)
    assert eta.remaining(10) == 40.


def test_full_validation_schedule_and_selection_tie():
    cfg = {**config(), 'epochs': 60, 'validation_interval': 5}
    assert [epoch for epoch in range(1, 61) if should_validate(epoch, cfg)] == [1] + list(range(5, 61, 5))
    empty_ap = {'overall_macro_f1': .5, 'ap50_95': None}
    with_ap = {'overall_macro_f1': .5, 'ap50_95': .1}
    assert selection_key(with_ap, 2) > selection_key(empty_ap, 1)
    assert selection_key(with_ap, 2) > selection_key(with_ap, 3)


def test_nonfinite_loss_does_not_update_parameters():
    model = SmallDetector()
    before = copy.deepcopy(model.state_dict())
    optimizer = build_optimizer(model, config())

    class NonfiniteCriterion(nn.Module):
        def forward(self, outputs, targets, **kwargs):
            return {'loss_bad': outputs['prediction'].sum() * float('nan')}

    with pytest.raises(FloatingPointError):
        train_epoch(model, NonfiniteCriterion(), loader(torch.Generator()), optimizer,
                    torch.device('cpu'), config(), 1, 'baseline', precision='float32')
    for name, value in before.items():
        assert torch.equal(value, model.state_dict()[name])

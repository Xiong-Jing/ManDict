"""Training primitives shared by RT-DETR baseline and RDF-DETR."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .common import format_duration, log


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == 'cuda')
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def autocast_context(device, precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and precision == 'bf16' else nullcontext()


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def capture_rng(generator=None):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state(),
             'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if generator is not None:
        state['loader_generator'] = generator.get_state()
    return state


def restore_rng(state, generator=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])
    if generator is not None and 'loader_generator' in state:
        generator.set_state(state['loader_generator'].cpu())


@dataclass
class StepETA:
    """Exclude cold start and use a bounded recent rolling estimate."""
    warmup: int = 10
    window: int = 100
    seen: int = 0
    samples: list = field(default_factory=list)

    def update(self, seconds):
        self.seen += 1
        if self.seen > self.warmup:
            self.samples.append(float(seconds))
            self.samples = self.samples[-self.window:]

    @property
    def seconds_per_step(self):
        return sum(self.samples) / len(self.samples) if self.samples else None

    def remaining(self, steps):
        return None if self.seconds_per_step is None else max(0, steps) * self.seconds_per_step


def should_validate(epoch, config):
    return epoch == 1 or epoch == int(config['epochs']) or epoch % int(config['validation_interval']) == 0


def selection_key(metrics, epoch):
    # Ties prefer stronger AP, then the earlier epoch; no test information enters this choice.
    ap = metrics.get('ap50_95')
    return (float(metrics['overall_macro_f1']), float(ap) if ap is not None else -1.0, -int(epoch))


def build_optimizer(model, config):
    backbone, remaining = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith('backbone.') else remaining).append(parameter)
    groups = []
    if backbone:
        groups.append({'params': backbone, 'lr': float(config['backbone_lr']), 'group_name': 'backbone'})
    if remaining:
        groups.append({'params': remaining, 'lr': float(config['lr']), 'group_name': 'heads'})
    return torch.optim.AdamW(groups, weight_decay=float(config['weight_decay']))


def build_scheduler(optimizer, config):
    epochs, warmup = int(config['epochs']), int(config.get('warmup_epochs', 1))

    def scale(completed_epochs):
        # First epoch uses 10% LR; the second starts at base LR, followed by cosine decay.
        if completed_epochs < warmup:
            return 0.1 + 0.9 * completed_epochs / max(warmup, 1)
        progress = (completed_epochs - warmup) / max(epochs - warmup, 1)
        return 0.01 + 0.99 * (1.0 + math.cos(math.pi * min(progress, 1.0))) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def phase_name(experiment, epoch, config):
    if experiment == 'baseline':
        return 'baseline'
    if experiment == 'context':
        return 'context_joint'
    if epoch <= int(config['role_warmup_epochs']):
        return 'fine_warmup'
    if epoch <= int(config['utility_warmup_end']):
        return 'utility_warmup'
    return 'mixed_joint'


def accumulation_divisor(batch_index, total_batches, accumulation_steps):
    """The final short accumulation group must retain the correct mean gradient."""
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_batches - group_start)


def train_epoch(model, criterion, loader, optimizer, device, config, epoch, experiment,
                precision='bf16', timing=None, queue_models=0):
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)
    if total_batches == 0:
        raise RuntimeError('训练集为空。')
    accumulation = int(config['accumulation_steps'])
    eta = StepETA()
    totals, optimizer_steps = {}, 0
    module_diagnostics = {} if experiment in ('detail', 'context', 'combined') else None
    epoch_start = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (images, targets) in enumerate(loader):
        batch_start = time.monotonic()
        images = images.to(device, non_blocking=device.type == 'cuda')
        targets = move_to_device(targets, device)
        with autocast_context(device, precision):
            outputs = model(images, targets=targets, epoch=epoch, profile=config['profile'])
            losses = criterion(outputs, targets, epoch=epoch, profile=config['profile'])
            if not losses or any(not key.startswith('loss_') for key in losses):
                raise RuntimeError('Criterion 必须仅返回加权 loss_* 标量，不能重复返回total。')
            loss = sum(losses.values())
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'{experiment} epoch={epoch} batch={batch_index + 1}: loss非有限值。')
        (loss / accumulation_divisor(batch_index, total_batches, accumulation)).backward()
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == total_batches:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['clip_grad_norm']))
            if not torch.isfinite(gradient_norm).item():
                raise FloatingPointError('梯度范数非有限值；未执行本次optimizer.step。')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        batch_seconds = time.monotonic() - batch_start
        eta.update(batch_seconds)
        detached = {key: float(value.detach()) for key, value in losses.items()}
        detached['total'] = float(loss.detach())
        for name, value in detached.items():
            totals[name] = totals.get(name, 0.0) + value
        if module_diagnostics is not None:
            context = criterion.last_diagnostics.get('context')
            if context is not None:
                sums = module_diagnostics.setdefault('context', {'batches': 0,
                            'selected_per_class': [0, 0, 0, 0], 'no_negative_evidence_batches': 0})
                sums['batches'] += 1
                sums['no_negative_evidence_batches'] += int(context.get('no_negative_evidence', False))
                for key in ('candidate_positive_boxes', 'selected_positive_boxes', 'excluded_low_weight',
                            'excluded_invalid', 'excluded_same_class_ignore', 'excluded_conflicting_class',
                            'excluded_budget', 'contrastive_anchors', 'hard_negative_pairs'):
                    sums[key] = sums.get(key, 0) + int(context.get(key, 0))
                for category, count in enumerate(context.get('selected_per_class', [])):
                    sums['selected_per_class'][category] += int(count)
                sums['residual_absolute_mean_sum'] = sums.get('residual_absolute_mean_sum', 0.) + float(
                    context.get('residual_absolute_mean', 0.))
            if 'selected_indices' in outputs:
                sums = module_diagnostics.setdefault('detail', {'batches': 0, 'selected_queries_per_image': 64,
                            'gain_target_batches': 0, 'gain_target_mean_sum': 0.})
                sums['batches'] += 1
                sums['selected_queries_per_image'] = int(outputs['selected_indices'].shape[1])
                if 'gain_target_mean' in criterion.last_diagnostics:
                    sums['gain_target_batches'] += 1
                    sums['gain_target_mean_sum'] += criterion.last_diagnostics['gain_target_mean']
        step = batch_index + 1
        if step == 1 or step % 20 == 0 or step == total_batches:
            remaining_batches = total_batches - step
            remaining_epochs = int(config['epochs']) - epoch
            validation_seconds = (timing or {}).get('validation_seconds')
            validations = sum(should_validate(e, config) for e in range(epoch, int(config['epochs']) + 1))
            model_eta = eta.remaining(remaining_batches + remaining_epochs * total_batches)
            if model_eta is not None and validation_seconds is not None:
                model_eta += validations * validation_seconds
            elif validations:
                model_eta = None
            queue_eta = None
            if queue_models == 0:
                queue_eta = model_eta
            elif (timing or {}).get('next_model_seconds') is not None and model_eta is not None:
                queue_eta = model_eta + queue_models * timing['next_model_seconds']
            log({'stage': 'train', 'experiment': experiment, 'phase': phase_name(experiment, epoch, config),
                 'epoch': epoch, 'epochs': config['epochs'], 'batch': step, 'batches': total_batches,
                 'loss': detached, 'lr': [group['lr'] for group in optimizer.param_groups],
                 'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else None,
                 'eta_epoch': format_duration(eta.remaining(remaining_batches)),
                 'eta_model': format_duration(model_eta), 'eta_pipeline': format_duration(queue_eta),
                 'eta_note': 'estimate; later-stage speed pending measurement' if queue_models else 'estimate'})
        del outputs, losses, loss, images, targets
    result = {'loss': {key: value / total_batches for key, value in totals.items()},
            'seconds': time.monotonic() - epoch_start, 'optimizer_steps': optimizer_steps,
            'seconds_per_step': eta.seconds_per_step,
            'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else None}
    if module_diagnostics is not None:
        if 'context' in module_diagnostics:
            context = module_diagnostics['context']
            context['residual_absolute_mean'] = context.pop('residual_absolute_mean_sum', 0.) / max(1, context['batches'])
        if 'detail' in module_diagnostics:
            detail = module_diagnostics['detail']
            detail['gain_target_mean'] = (detail.pop('gain_target_mean_sum') / detail['gain_target_batches']
                                          if detail['gain_target_batches'] else None)
        result['module_diagnostics'] = module_diagnostics
    return result


@torch.no_grad()
def predict(model, loader, device, config, experiment, epoch, precision='bf16', split='val'):
    from .metrics import postprocess
    model.eval()
    records = []
    eta = StepETA(warmup=2)
    start = time.monotonic()
    for index, (images, targets) in enumerate(loader):
        batch_start = time.monotonic()
        images = images.to(device, non_blocking=device.type == 'cuda')
        moved_targets = move_to_device(targets, device)
        with autocast_context(device, precision):
            outputs = model(images, targets=moved_targets, epoch=epoch, profile=config['profile'])
        records.extend(postprocess(outputs, targets, image_size=int(config['image_size'])))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        eta.update(time.monotonic() - batch_start)
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(loader):
            log({'stage': 'predict', 'experiment': experiment, 'split': split, 'epoch': epoch,
                 'batch': index + 1, 'batches': len(loader), 'images': len(records),
                 'eta_validation': format_duration(eta.remaining(len(loader) - index - 1)), 'eta_note': 'estimate'})
        del images, moved_targets, outputs
    return records, time.monotonic() - start


def flatten_metrics(metrics):
    result = {'macro_pseudo_f1': metrics['overall_macro_f1']}
    for name, values in metrics['per_class'].items():
        for metric in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn'):
            if metric in values:
                result[f'{name}_{metric}'] = values[metric]
    for name in ('ap50_95', 'ap50', 'ap75'):
        result[f'pseudo_{name}'] = metrics.get(name)
    return result


def write_history_csv(path, history):
    rows = []
    for item in history:
        row = {'epoch': item['epoch'], 'threshold': item.get('threshold'),
               'train_seconds': item.get('train', {}).get('seconds'),
               'train_loss': item.get('train', {}).get('loss', {}).get('total')}
        if item.get('metrics'):
            row.update(flatten_metrics(item['metrics']))
        rows.append(row)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

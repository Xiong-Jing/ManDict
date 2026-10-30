"""Official RT-DETR-style weighted VFL/box criterion with pseudo-ignore support.

The unchanged upstream criterion is retained under vendor for attribution and
comparison. This adaptation applies identical weak-label rules to every branch.
Returned values are already weighted scalars; sum them ONCE (no total duplicate).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vendor.rtdetr_runtime.matcher import HungarianMatcher
from vendor.rtdetr_runtime.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from vendor.rtdetr_runtime.rtdetr_criterion import SetCriterion
from .models import stage_for
from .context import context_contrastive_loss


def _float_outputs(outputs):
    return {'pred_logits': outputs['pred_logits'].float(), 'pred_boxes': outputs['pred_boxes'].float()}


def _weights(target):
    return target.get('weights', target['boxes'].new_ones(len(target['labels']))).float()


class DetectionCriterion(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        # No reference to model is registered here: the optimizer owns model once.
        self.matcher = HungarianMatcher({'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                                        use_focal_loss=True, alpha=.25, gamma=2.)
        self.alpha, self.gamma = .75, 2.
        self.last_diagnostics = {}

    @torch.no_grad()
    def match(self, outputs, targets):
        device = outputs['pred_logits'].device
        if sum(len(target['labels']) for target in targets) == 0:
            return [(torch.empty(0, dtype=torch.long, device=device),
                     torch.empty(0, dtype=torch.long, device=device)) for _ in targets]
        indices = self.matcher(_float_outputs(outputs), targets)
        return [(source.to(device), truth.to(device)) for source, truth in indices]

    def classification_components(self, outputs, targets, indices):
        logits, boxes = outputs['pred_logits'].float(), outputs['pred_boxes'].float()
        target = torch.zeros_like(logits)
        quality = torch.zeros_like(logits)
        positive_weight = torch.ones_like(logits)
        active = torch.ones_like(logits)
        matched = torch.zeros(logits.shape[:2], dtype=torch.bool, device=logits.device)
        for batch, (source, truth) in enumerate(indices):
            if len(source):
                labels = targets[batch]['labels'][truth]
                matched[batch, source] = True
                target[batch, source, labels] = 1
                iou, _ = box_iou(box_cxcywh_to_xyxy(boxes[batch, source]),
                                 box_cxcywh_to_xyxy(targets[batch]['boxes'][truth].float()))
                quality[batch, source, labels] = iou.diag().detach()
                positive_weight[batch, source, labels] = _weights(targets[batch])[truth]
            ignores = targets[batch].get('ignore_boxes')
            labels = targets[batch].get('ignore_labels')
            if ignores is None or labels is None or len(ignores) == 0:
                continue
            ious, _ = box_iou(box_cxcywh_to_xyxy(boxes[batch].detach()),
                             box_cxcywh_to_xyxy(ignores.float()))
            unmatched = ~matched[batch]
            for category in range(4):
                subset = labels == category
                if bool(subset.any()):
                    ignored = (ious[:, subset].max(-1).values >= .5) & unmatched
                    active[batch, ignored, category] = 0
        return target, quality, positive_weight, active, matched

    def branch_losses(self, outputs, targets, indices=None, normalizer=None):
        outputs = _float_outputs(outputs)
        if indices is None:
            indices = self.match(outputs, targets)
        if normalizer is None:
            normalizer = max(1, sum(len(target['labels']) for target in targets))
        target, quality, positive_weight, active, _matched = self.classification_components(outputs, targets, indices)
        logits = outputs['pred_logits']
        vfl_weights = self.alpha * logits.sigmoid().detach().pow(self.gamma) * (1 - target) + quality
        vfl = F.binary_cross_entropy_with_logits(logits, quality, reduction='none')
        vfl = (vfl * vfl_weights * positive_weight * active).sum() / normalizer
        l1 = outputs['pred_boxes'].sum() * 0
        giou = l1
        for batch, (source, truth) in enumerate(indices):
            if not len(source):
                continue
            predicted = outputs['pred_boxes'][batch, source]
            truth_boxes = targets[batch]['boxes'][truth].float()
            weights = _weights(targets[batch])[truth]
            l1 = l1 + (F.l1_loss(predicted, truth_boxes, reduction='none').sum(-1) * weights).sum()
            overlaps = generalized_box_iou(box_cxcywh_to_xyxy(predicted), box_cxcywh_to_xyxy(truth_boxes))
            giou = giou + ((1 - overlaps.diag()) * weights).sum()
        return {'loss_vfl': vfl, 'loss_bbox': 5 * l1 / normalizer, 'loss_giou': 2 * giou / normalizer}

    @torch.no_grad()
    def utility_targets(self, coarse, fine, targets, indices):
        """Signed detached improvement with one assignment and no denoising queries."""
        coarse = _float_outputs(coarse)
        fine = _float_outputs(fine)
        target, _, pos_weights, active, matched = self.classification_components(coarse, targets, indices)
        # Fixed 0/1 class proxy permits an interpretable coarse/fine comparison.
        def per_query(outputs):
            logits = outputs['pred_logits']
            probabilities = logits.sigmoid()
            p_t = probabilities * target + (1 - probabilities) * (1 - target)
            alpha_t = .25 * target + .75 * (1 - target)
            classification = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            result = (classification * (1 - p_t).pow(2) * alpha_t * pos_weights * active).sum(-1)
            for batch, (source, truth) in enumerate(indices):
                if not len(source):
                    continue
                predicted = outputs['pred_boxes'][batch, source]
                reference = targets[batch]['boxes'][truth].float()
                weights = _weights(targets[batch])[truth]
                l1 = (predicted - reference).abs().sum(-1)
                giou = 1 - generalized_box_iou(box_cxcywh_to_xyxy(predicted),
                                              box_cxcywh_to_xyxy(reference)).diag()
                result[batch, source] += (5 * l1 + 2 * giou) * weights
            return result
        gain = (per_query(coarse) - per_query(fine)).detach()
        return gain, matched, active.bool().any(-1)

    @staticmethod
    def _append(result, values, suffix='', coefficient=1.):
        result.update({key + suffix: coefficient * value for key, value in values.items()})

    def forward(self, outputs, targets, epoch=1, profile='pilot'):
        coarse = outputs.get('coarse_outputs', outputs)
        indices = self.match(coarse, targets)
        result = {}
        self._append(result, self.branch_losses(coarse, targets, indices))
        for index, auxiliary in enumerate(outputs.get('aux_outputs', [])):
            self._append(result, self.branch_losses(auxiliary, targets), f'_aux_{index}')
        dn_meta = outputs.get('dn_meta')
        if dn_meta is not None:
            dn_indices = SetCriterion.get_cdn_matched_indices(dn_meta, targets)
            normalizer = max(1, sum(len(target['labels']) for target in targets) * dn_meta['dn_num_group'])
            for index, auxiliary in enumerate(outputs.get('dn_aux_outputs', [])):
                self._append(result, self.branch_losses(auxiliary, targets, dn_indices, normalizer), f'_dn_{index}')
        fine = outputs.get('fine_outputs')
        stage = stage_for(epoch, profile)
        self.last_diagnostics = {'stage': stage}
        if fine is not None:
            self._append(result, self.branch_losses(fine, targets, indices), '_fine', .5)
            if stage == 'joint':
                self._append(result, self.branch_losses(outputs, targets, indices), '_mixed')
            if stage != 'fine_warmup':
                expected, positives, active = self.utility_targets(coarse, fine, targets, indices)
                outputs['gain_targets'] = expected
                regression = F.smooth_l1_loss(outputs['gain_pred'].float(), expected, reduction='none')
                groups = [regression[positives & active], regression[(~positives) & active]]
                values = [group.mean() for group in groups if group.numel()]
                result['loss_gain'] = .1 * (torch.stack(values).mean() if values else regression.sum() * 0)
                self.last_diagnostics['gain_target_mean'] = float(expected.mean())
        pairs = outputs.get('role_pair_logits', [])
        if pairs:
            positive = torch.cat([pair['positive'] for pair in pairs])
            negative = torch.cat([pair['negative'] for pair in pairs])
            weights = torch.cat([pair['weights'] for pair in pairs]).float()
            if positive.numel():
                bce = (F.binary_cross_entropy_with_logits(positive.float(), torch.ones_like(positive).float(), reduction='none') +
                       F.binary_cross_entropy_with_logits(negative.float(), torch.zeros_like(negative).float(), reduction='none')) * .5
                result['loss_role'] = .2 * (bce * weights).mean()
                result['loss_role_pair'] = .1 * (F.relu(.5 - positive.float() + negative.float()) * weights).mean()
                self.last_diagnostics['role_pairs'] = int(positive.numel())
        context_aux = outputs.get('context_aux')
        if context_aux is not None:
            contrast, diagnostics = context_contrastive_loss(context_aux)
            result['loss_context_contrast'] = .05 * contrast
            self.last_diagnostics['context'] = {
                **outputs.get('context_diagnostics', {}),
                **context_aux.get('selection', {}), **diagnostics,
                'uses_ocr': False, 'ignore_is_negative': False,
            }
        return result

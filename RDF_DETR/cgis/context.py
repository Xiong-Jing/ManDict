"""Positive-field context supervision; no OCR or inferred sentence labels.

Supervised contrastive learning is an auxiliary, established technique. The
experiment uses it only on trustworthy existing positive pseudo boxes. Ignore
regions remain unknown; they are never assigned a negative class here.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from vendor.rtdetr_runtime.box_ops import box_cxcywh_to_xyxy, box_iou


CONTEXT_POLICY = {
    'reference_type': 'pseudo',
    'positive_source': 'existing_four_class_positive_pseudo_boxes',
    'minimum_weight': .75,
    'maximum_auxiliary_boxes_per_image': 64,
    'same_class_ignore_iou_or_box_coverage': .5,
    'different_class_conflict_iou_or_smaller_box_coverage': .5,
    'negative_source': 'known_different_class_positive_fields_only',
    'hard_negative_log_area_gap': 1.1,
    'hard_negative_log_aspect_gap': .7,
    'hard_negative_multiplier': 2.,
    'temperature': .2,
    'contrastive_coefficient': .05,
    'residual_bound': 2.,
    'placement': 'coarse_decoder_logits_before_detail_gain_and_refinement',
    'uses_ocr': False,
    'uses_ignore_as_negative': False,
}


def _intersections(first, second):
    lower = torch.maximum(first[:, None, :2], second[None, :, :2])
    upper = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    return (upper - lower).clamp_min(0).prod(-1)


@torch.no_grad()
def select_context_targets(targets, max_per_image=64, min_weight=.75,
                           ignore_iou=.5, conflict_iou=.5):
    """Return trustworthy positive indices without changing detector targets.

    Confidence and overlap filtering affects this auxiliary objective only. In
    particular, original positive/ignore precedence in the detector is intact.
    Class round robin prevents a dense category from using the entire ROI budget.
    """
    selected = []
    counts = {'candidate_positive_boxes': 0, 'selected_positive_boxes': 0,
              'excluded_low_weight': 0, 'excluded_invalid': 0,
              'excluded_same_class_ignore': 0, 'excluded_conflicting_class': 0,
              'excluded_budget': 0, 'selected_per_class': [0, 0, 0, 0]}
    for target in targets:
        boxes, labels = target['boxes'].float(), target['labels'].long()
        weights = target.get('weights', boxes.new_ones(len(boxes))).float()
        counts['candidate_positive_boxes'] += len(boxes)
        valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > 0).all(-1)
        valid &= (labels >= 0) & (labels < 4) & torch.isfinite(weights)
        counts['excluded_invalid'] += int((~valid).sum())
        low_weight = valid & (weights < min_weight)
        counts['excluded_low_weight'] += int(low_weight.sum())
        valid &= weights >= min_weight
        xyxy = box_cxcywh_to_xyxy(boxes)
        area = boxes[:, 2:].prod(-1).clamp_min(1e-12)
        ignored = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
        ignore_boxes = target.get('ignore_boxes')
        ignore_labels = target.get('ignore_labels')
        if ignore_boxes is not None and ignore_labels is not None and len(ignore_boxes):
            ignore_xyxy = box_cxcywh_to_xyxy(ignore_boxes.float())
            iou, _ = box_iou(xyxy, ignore_xyxy)
            coverage = _intersections(xyxy, ignore_xyxy) / area[:, None]
            same = labels[:, None] == ignore_labels[None]
            ignored = (((iou >= ignore_iou) | (coverage >= ignore_iou)) & same).any(-1)
        counts['excluded_same_class_ignore'] += int((valid & ignored).sum())
        valid &= ~ignored
        # Mutually conflicting known classes do not become hard negatives. Only
        # surviving reliable boxes participate, so a rejected low-weight box
        # cannot remove an otherwise valid auxiliary sample.
        conflict = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
        if bool(valid.any()):
            iou, _ = box_iou(xyxy, xyxy)
            coverage = _intersections(xyxy, xyxy) / torch.minimum(area[:, None], area[None]).clamp_min(1e-12)
            other_class = labels[:, None] != labels[None]
            collision = ((iou >= conflict_iou) | (coverage >= conflict_iou)) & other_class
            collision &= valid[:, None] & valid[None]
            conflict = collision.any(-1)
        counts['excluded_conflicting_class'] += int((valid & conflict).sum())
        valid &= ~conflict
        queues = []
        for category in range(4):
            subset = (valid & (labels == category)).nonzero(as_tuple=False).flatten()
            order = torch.argsort(weights[subset], descending=True, stable=True)
            queues.append(subset[order].tolist())
        chosen = []
        while any(queues) and len(chosen) < max_per_image:
            for queue in queues:
                if queue and len(chosen) < max_per_image:
                    chosen.append(queue.pop(0))
        index = torch.as_tensor(chosen, dtype=torch.long, device=boxes.device)
        counts['excluded_budget'] += int(valid.sum()) - len(index)
        counts['selected_positive_boxes'] += len(index)
        for category in range(4):
            counts['selected_per_class'][category] += int((labels[index] == category).sum())
        selected.append(index)
    return selected, counts


def context_contrastive_loss(aux, temperature=.2):
    """Weighted two-view supervised contrast with known-class negatives only.

    Similar size/aspect different-class fields receive stronger denominator
    weight. A batch with no known different class has zero contrastive evidence;
    its differentiable zero does not invent background or sentence supervision.
    """
    embeddings = aux['embeddings'].float()
    zero = embeddings.sum() * 0
    count = embeddings.shape[1] if embeddings.ndim == 3 else 0
    diagnostics = {'auxiliary_boxes': count, 'contrastive_anchors': 0,
                   'hard_negative_pairs': 0, 'no_negative_evidence': True}
    if count == 0:
        return zero, diagnostics
    if embeddings.shape[0] != 2:
        raise ValueError('Context contrast requires exactly two feature views')
    labels = aux['labels'].repeat(2)
    weights = aux['weights'].float().clamp(0, 1).repeat(2)
    boxes = aux['boxes'].float().repeat(2, 1)
    vectors = F.normalize(embeddings.reshape(-1, embeddings.shape[-1]), dim=-1)
    similarity = vectors @ vectors.t() / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same_class = labels[:, None] == labels[None]
    positive = same_class & ~eye
    negative = ~same_class
    area = boxes[:, 2:].prod(-1).clamp_min(1e-10).log()
    aspect = (boxes[:, 2].clamp_min(1e-10) / boxes[:, 3].clamp_min(1e-10)).log()
    hard = negative & ((area[:, None] - area[None]).abs() <= 1.1)
    hard &= (aspect[:, None] - aspect[None]).abs() <= .7
    active = positive.any(-1) & negative.any(-1) & (weights > 0)
    diagnostics.update(contrastive_anchors=int(active.sum()),
                       hard_negative_pairs=int(hard.sum()),
                       no_negative_evidence=not bool(negative.any()))
    if not bool(active.any()):
        return zero, diagnostics
    denominator_weight = torch.where(hard, similarity.new_tensor(2.), similarity.new_tensor(1.))
    # A low-confidence known positive also supplies proportionately weaker
    # evidence when used as another anchor's contrast partner.
    denominator_weight = denominator_weight * weights[None].clamp_min(1e-8)
    adjusted = similarity + denominator_weight.log()
    adjusted = adjusted.masked_fill(eye, -float('inf'))
    log_probability = similarity + denominator_weight.log() - torch.logsumexp(adjusted, dim=-1, keepdim=True)
    positive_weights = positive.to(similarity.dtype) * weights[None]
    per_anchor = -(log_probability * positive_weights).sum(-1) / positive_weights.sum(-1).clamp_min(1e-8)
    value = (per_anchor[active] * weights[active]).sum() / weights[active].sum().clamp_min(1e-8)
    return value, diagnostics

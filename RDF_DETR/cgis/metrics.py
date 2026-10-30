"""Four-class pseudo-reference evaluation, independent of the old OCR project."""
from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stdout
import copy
import io

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .common import CATEGORIES


def iou_xywh(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    intersection = max(0., min(ax + aw, bx + bw) - max(ax, bx)) * max(0., min(ay + ah, by + bh) - max(ay, by))
    union = max(aw, 0) * max(ah, 0) + max(bw, 0) * max(bh, 0) - intersection
    return intersection / union if union > 0 else 0.


def hungarian_matches(predictions, references, iou_threshold=.5):
    """Legacy protocol: maximize total IoU, then apply the overlap threshold."""
    if not predictions or not references:
        return []
    matrix = np.array([[iou_xywh(p['bbox'], r['bbox']) for r in references] for p in predictions], dtype=np.float64)
    rows, columns = linear_sum_assignment(-matrix)
    return [(int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, columns)
            if matrix[i, j] + 1e-12 >= iou_threshold]


def _group(records, refs):
    allowed = {int(image['id']) for image in refs['images']}
    predictions = defaultdict(list)
    references = defaultdict(list)
    seen = set()
    for record in records:
        image_id = int(record['image_id'])
        if image_id not in allowed or image_id in seen:
            raise ValueError('Prediction outside evaluation whitelist or duplicate image')
        seen.add(image_id)
        predictions[image_id].extend(record.get('fields', []))
    if seen != allowed:
        raise ValueError('Every evaluation image must have one prediction record, including empty outputs')
    for annotation in refs['annotations']:
        if int(annotation['image_id']) not in allowed:
            raise ValueError('Reference annotation is outside selected evaluation images')
        references[int(annotation['image_id'])].append(annotation)
    return allowed, predictions, references


def _prf(records, refs, threshold):
    allowed, predictions, references = _group(records, refs)
    per_class = {}
    for category_id, category in CATEGORIES.items():
        counts = dict(tp=0, fp=0, fn=0, ignored_predictions=0)
        for image_id in allowed:
            predicted = [p for p in predictions[image_id] if int(p['category_id']) == category_id
                         and float(p['score']) + 1e-12 >= threshold]
            positive = [r for r in references[image_id] if int(r['category_id']) == category_id and not r.get('ignore', False)]
            ignored = [r for r in references[image_id] if int(r['category_id']) == category_id and r.get('ignore', False)]
            matches = hungarian_matches(predicted, positive)
            matched = {p for p, _, _ in matches}
            ignored_count = sum(any(iou_xywh(p['bbox'], r['bbox']) + 1e-12 >= .5 for r in ignored)
                                for index, p in enumerate(predicted) if index not in matched)
            counts['tp'] += len(matches)
            counts['fn'] += len(positive) - len(matches)
            counts['fp'] += len(predicted) - len(matches) - ignored_count
            counts['ignored_predictions'] += ignored_count
        precision = counts['tp'] / max(counts['tp'] + counts['fp'], 1)
        recall = counts['tp'] / max(counts['tp'] + counts['fn'], 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        per_class[category] = dict(precision=precision, recall=recall, f1=f1, **counts)
    return dict(reference_type='pseudo', confidence_threshold=float(threshold), iou_threshold=.5,
                ignore_iou_threshold=.5, per_class=per_class,
                overall_macro_f1=sum(c['f1'] for c in per_class.values()) / 4)


def compute_ap(records, refs):
    """COCO positive matching first, then typed IoU-ignore on unmatched DTs.

    Ignore boxes are never crowd GT: crowd uses IoA, which is a different rule.
    Positive matches, official area ranges/maxDets and COCO score ordering remain.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError('Install the project dependency pycocotools before AP evaluation') from error
    allowed, _, references = _group(records, refs)
    active = []
    for annotation in refs['annotations']:
        if annotation.get('ignore', False):
            continue
        box = [float(v) for v in annotation['bbox']]
        active.append(dict(id=len(active) + 1, image_id=int(annotation['image_id']),
                           category_id=int(annotation['category_id']), bbox=box,
                           area=box[2] * box[3], iscrowd=0))
    detections = [dict(image_id=int(record['image_id']), category_id=int(field['category_id']),
                       bbox=list(map(float, field['bbox'])), score=float(field['score']))
                  for record in records for field in record.get('fields', [])]
    with redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = dict(info={}, images=copy.deepcopy(refs['images']), annotations=active,
                            categories=[dict(id=i, name=name) for i, name in CATEGORIES.items()])
        coco.createIndex()
        if detections:
            predicted = coco.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = dict(images=copy.deepcopy(refs['images']), annotations=[],
                                     categories=coco.dataset['categories'])
            predicted.createIndex()
        evaluator = COCOeval(coco, predicted, 'bbox')
        evaluator.params.imgIds = sorted(allowed)
        evaluator.params.catIds = list(CATEGORIES)
        evaluator.evaluate()
        for item in evaluator.evalImgs:
            if item is None:
                continue
            ignored = [r for r in references[item['image_id']] if r.get('ignore', False)
                       and int(r['category_id']) == item['category_id']]
            if not ignored:
                continue
            for column, detection_id in enumerate(item['dtIds']):
                detection = predicted.anns[detection_id]
                if any(iou_xywh(detection['bbox'], r['bbox']) + 1e-12 >= .5 for r in ignored):
                    item['dtIgnore'][:, column] |= item['dtMatches'][:, column] == 0
        evaluator.accumulate()
        evaluator.summarize()
    values = evaluator.stats.tolist()
    per_class = {}
    precision = evaluator.eval['precision']  # IoU, recall, category, area, maxDet
    for index, category in enumerate(CATEGORIES.values()):
        all_values = precision[:, :, index, 0, -1]
        valid = all_values[all_values > -1]
        at_50 = precision[0, :, index, 0, -1]
        at_50 = at_50[at_50 > -1]
        per_class[category] = dict(ap50_95=float(valid.mean()) if valid.size else None,
                                   ap50=float(at_50.mean()) if at_50.size else None)
    names = ('ap50_95', 'ap50', 'ap75', 'ap_small', 'ap_medium', 'ap_large')
    return dict(protocol='ignore-aware Pseudo-AP: official COCO matching; unmatched typed ignore IoU>=0.5',
                max_detections=100, **{key: value if value >= 0 else None for key, value in zip(names, values)},
                per_class=per_class)


def evaluate(records, refs, threshold=.55, compute_ap=True):
    result = _prf(records, refs, threshold)
    if compute_ap:
        ap = globals()['compute_ap'](records, refs)
        result.update(ap=ap, ap50_95=ap['ap50_95'], ap50=ap['ap50'], ap75=ap['ap75'])
    return result


def select_threshold(records, refs):
    trials = [_prf(records, refs, integer / 100) for integer in range(5, 96, 5)]
    winner = min(trials, key=lambda row: (-row['overall_macro_f1'], row['confidence_threshold']))
    result = dict(winner)
    ap = compute_ap(records, refs)
    result.update(ap=ap, ap50_95=ap['ap50_95'], ap50=ap['ap50'], ap75=ap['ap75'])
    return result['confidence_threshold'], result, trials


@torch.inference_mode()
def postprocess(outputs, targets, image_size=896):
    """Official sigmoid/top300 over query×class, followed by letterbox inversion."""
    from .data import restore_boxes
    logits = outputs.get('role_adjusted_logits', outputs['pred_logits'])
    probabilities = logits.float().sigmoid()
    scores, indices = probabilities.flatten(1).topk(min(300, probabilities.shape[1] * 4), dim=1)
    labels = indices % 4
    query_indices = indices // 4
    boxes = outputs['pred_boxes'].gather(1, query_indices[..., None].expand(-1, -1, 4))
    records = []
    for predicted_boxes, classes, confidence, target in zip(boxes, labels, scores, targets):
        original = restore_boxes(predicted_boxes.detach().cpu(), target, image_size=image_size)
        fields = []
        for box, category, score in zip(original.tolist(), classes.cpu().tolist(), confidence.cpu().tolist()):
            if box[2] <= 0 or box[3] <= 0:
                continue
            category += 1
            fields.append(dict(field_id=len(fields) + 1, category_id=category, category=CATEGORIES[category],
                               bbox=box, score=float(score), entry_id=None, pair_id=None))
        records.append(dict(image_id=int(target['image_id']), file_name=target['file_name'], fields=fields))
    return records

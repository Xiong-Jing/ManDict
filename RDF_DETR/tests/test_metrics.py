from contextlib import redirect_stdout
import io

import pytest

from cgis.metrics import evaluate, compute_ap, select_threshold


def fixture():
    refs = dict(images=[dict(id=1, file_name='1.png', width=100, height=100)], annotations=[
        dict(id=1, image_id=1, category_id=1, bbox=[0, 0, 10, 10]),
        dict(id=2, image_id=1, category_id=1, bbox=[20, 0, 10, 10], ignore=True)])
    fields = [dict(category_id=1, bbox=[0, 0, 10, 10], score=.99),
              dict(category_id=1, bbox=[0, 0, 10, 10], score=.98),
              dict(category_id=1, bbox=[20, 0, 10, 10], score=.97),
              dict(category_id=1, bbox=[20, 0, 10, 10], score=.96),
              dict(category_id=2, bbox=[20, 0, 10, 10], score=.95)]
    return [dict(image_id=1, file_name='1.png', fields=fields)], refs


def test_duplicate_fp_typed_ignore_and_positive_priority():
    records, refs = fixture()
    metrics = evaluate(records, refs, .5)
    assert metrics['per_class']['chinese_word']['tp'] == 1
    assert metrics['per_class']['chinese_word']['fp'] == 1
    assert metrics['per_class']['chinese_word']['ignored_predictions'] == 2
    assert metrics['per_class']['pos']['fp'] == 1
    refs['annotations'].append(dict(id=3, image_id=1, category_id=1, bbox=[0, 0, 10, 10], ignore=True))
    assert evaluate(records, refs)['per_class']['chinese_word']['tp'] == 1


def test_ap_matches_official_coco_without_ignore():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    records, refs = fixture()
    refs['annotations'] = [dict(refs['annotations'][0], area=100, iscrowd=0)]
    actual = compute_ap(records, refs)
    with redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = dict(info={}, images=refs['images'], annotations=refs['annotations'],
                            categories=[dict(id=i, name=str(i)) for i in range(1, 5)])
        coco.createIndex()
        detected = coco.loadRes([dict(image_id=1, **f) for f in records[0]['fields']])
        evaluator = COCOeval(coco, detected, 'bbox')
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    assert actual['ap50_95'] == pytest.approx(evaluator.stats[0])
    assert actual['ap50'] == pytest.approx(evaluator.stats[1])


def test_empty_predictions_and_threshold_selection_no_test_tuning():
    records, refs = fixture()
    threshold, selected, grid = select_threshold(records, refs)
    assert threshold == .05
    assert len(grid) == 19
    assert selected['reference_type'] == 'pseudo'
    records[0]['fields'] = []
    assert evaluate(records, refs)['overall_macro_f1'] == 0
    with pytest.raises(ValueError):
        evaluate([], refs)

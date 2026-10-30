"""Independent, OCR-free four-arm experiments with read-only baseline provenance."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import traceback

import torch

from .common import ROOT, digest_json, log, read_json, seed_everything, sha256, write_json
from .environment import inspect_environment, verify_cuda_compute
from .runner import (data_identity, exclusive_run, experiment_identity, export_results,
                     gpu_memory_report, load_config, make_loader, print_metrics,
                     references_for_loader, run_model, source_identity, summary_reports)
from .training import autocast_context, build_optimizer, move_to_device, predict


ARMS = ('baseline', 'detail', 'context', 'combined')
VERSION = 'ablations_v1'


def output_root(root, profile, test_only=False):
    return root / 'runs' / ('cpu_ablation_diagnostics' if test_only else VERSION) / profile


def _require(condition, reason):
    if not condition:
        raise RuntimeError('Baseline import refused: ' + reason +
                           '. Keep the original results; run --experiment baseline to train '
                           'a new COCO-initialized baseline in the independent ablation directory.')


def verify_legacy_source(root, recorded_source):
    """An audited, hash-bound migration permits only preregistered legacy edits.

    New extension files never existed in the old identity. Existing files not explicitly
    registered must remain byte-identical; no old manifest/fingerprint is rewritten.
    """
    path = root / 'configs' / 'ablation_baseline_compat.json'
    _require(path.is_file(), 'missing audited compatibility report')
    report = read_json(path)
    _require(report.get('schema_version') == 1 and report.get('baseline_invariance_verified') is True,
             'baseline numerical invariance has not been verified')
    legacy = report.get('legacy_source')
    _require(isinstance(legacy, dict) and legacy == recorded_source,
             'old source does not match the preregistered baseline source')
    changed = report.get('changed_paths', {})
    _require(set(changed).issubset(legacy), 'compatibility report contains unregistered old paths')
    current = source_identity(root)
    for name, old_hash in legacy.items():
        _require(name in current, f'legacy source file is missing: {name}')
        if name in changed:
            record = changed[name]
            _require(record.get('old_sha256') == old_hash and
                     record.get('new_sha256') == current[name],
                     f'changed-file hash mismatch: {name}')
        else:
            _require(current[name] == old_hash, f'unapproved source change: {name}')
    return {'compatibility_report_sha256': sha256(path),
            'changed_paths': sorted(changed),
            'baseline_invariance_scope': report.get('baseline_invariance_scope'),
            'legacy_source_sha256': digest_json(legacy)}


def import_legacy_baseline(root, config, metadata, environment, source, output):
    """Verify old completed baseline and write a new provenance record only."""
    from .data import load_references
    from .metrics import evaluate
    source = Path(source).resolve()
    output = Path(output)
    required = ('manifest.json', 'completion.json', 'metrics_val.json',
                'predictions_val.json', 'checkpoint_selection.json', 'best.pt')
    for name in required:
        _require((source / name).is_file(), f'missing {source / name}')
    manifest, done = read_json(source / 'manifest.json'), read_json(source / 'completion.json')
    identity = manifest.get('identity', {})
    _require(manifest.get('complete') is True and manifest.get('trained') is True and
             done.get('complete') is True, 'old run is not complete')
    _require(identity.get('experiment') == 'baseline' and done.get('experiment') == 'baseline',
             'source is not a baseline experiment')
    _require(manifest.get('reference_type') == done.get('reference_type') == 'pseudo',
             'reference type differs')
    _require(manifest.get('fingerprint') == done.get('fingerprint') == digest_json(identity),
             'old manifest/completion identity is inconsistent')
    current = experiment_identity(root, config, metadata, environment, 'baseline')
    for key in ('config', 'data', 'pretrained_sha256', 'runtime', 'role_supervision'):
        _require(identity.get(key) == current.get(key), f'{key} differs')
    _require(int(done.get('epochs', -1)) == int(config['epochs']), 'training budget differs')
    _require(bool(done.get('test_only')) == bool(environment['test_only']), 'diagnostic mode differs')
    if not environment['test_only']:
        _require(manifest.get('gpu_verified') is True and done.get('gpu_verified') is True,
                 'old baseline has no real GPU training verification')
    source_review = verify_legacy_source(root, identity.get('source'))
    _require(sha256(source / 'best.pt') == done.get('best_sha256'), 'best checkpoint SHA-256 differs')
    checkpoint = torch.load(source / 'best.pt', map_location='cpu')
    _require(checkpoint.get('fingerprint') == manifest['fingerprint'] and
             checkpoint.get('experiment') == 'baseline' and
             checkpoint.get('epoch') == done.get('best_epoch') and
             checkpoint.get('threshold') == done.get('threshold') and
             checkpoint.get('config') == config,
             'best checkpoint metadata differs from the completed baseline')
    del checkpoint
    metrics = read_json(source / 'metrics_val.json')
    selection = read_json(source / 'checkpoint_selection.json')
    _require(metrics == done.get('metrics_val') == selection.get('metrics'), 'stored metrics differ')
    _require(selection.get('epoch') == done.get('best_epoch') and
             selection.get('threshold') == done.get('threshold') and
             selection.get('selection_split') == 'val', 'checkpoint selection differs')
    records = read_json(source / 'predictions_val.json')
    references = load_references(root, 'val', config['profile'])
    # Formal import uses the entire fixed validation profile. CPU diagnostics cannot
    # accidentally admit a full validation run into a two-batch diagnostic experiment.
    loader = make_loader(root, config, 'val', num_workers=0)
    references = references_for_loader(references, loader)
    ids = [int(record['image_id']) for record in records]
    _require(len(ids) == len(set(ids)) and set(ids) == {int(r['id']) for r in references['images']},
             'prediction image set does not match this validation profile')
    replay = evaluate(records, references, threshold=done['threshold'])
    _require(_numerically_equal(replay, metrics), 'prediction replay does not reproduce saved metrics')
    evidence = {name: sha256(source / name) for name in required}
    provenance = {'schema_version': 1, 'mode': 'read_only_baseline_import',
                  'source_directory': str(source), 'source_fingerprint': manifest['fingerprint'],
                  'reference_type': 'pseudo', 'input_identity_sha256': digest_json(
                      {k: current[k] for k in ('config', 'data', 'pretrained_sha256', 'runtime')}),
                  'evidence_sha256': evidence, 'source_review': source_review,
                  'prediction_replay_verified': True}
    destination = output / 'baseline' / 'baseline_import.json'
    if destination.exists():
        _require(read_json(destination) == provenance, 'immutable import provenance changed')
    else:
        write_json(destination, provenance)
    # No checkpoint, metrics or manifest in the source directory is changed.
    result = {**done, 'imported': True, 'checkpoint_path': str(source / 'best.pt'),
              'baseline_import': str(destination), 'baseline_source': str(source)}
    write_json(output / 'baseline' / 'imported_completion.json', result)
    log({'stage': 'baseline_import', 'source': str(source), 'best_epoch': done['best_epoch'],
         'macro_pseudo_f1': metrics['overall_macro_f1'], 'source_read_only': True,
         'prediction_replay_verified': True})
    return result


def _numerically_equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_numerically_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_numerically_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)
    return left == right


def smoke(root, config, environment, experiment, output):
    from .models import build_model
    from .losses import DetectionCriterion
    from .metrics import postprocess
    device = torch.device(environment['requested_device'])
    seed_everything(int(config['seed']))
    model, report = build_model(experiment, pretrained_path=root / 'weights/rtdetr_r18vd_coco.pth',
                                load_pretrained=True, role_enabled=[False] * 4)
    model.to(device).train()
    criterion = DetectionCriterion(model=model).to(device)
    loader = make_loader(root, config, 'train', num_workers=0,
                         generator=torch.Generator().manual_seed(int(config['seed'])))
    images, targets = next(iter(loader))
    images, moved_targets = images.to(device), move_to_device(targets, device)
    optimizer = build_optimizer(model, config)
    prefixes = ([] if experiment == 'baseline' else
                (['refiner', 'gain_head'] if experiment == 'detail' else
                 ['context_head'] if experiment == 'context' else ['refiner', 'gain_head', 'context_head']))
    tracked = {name: value.detach().clone() for name, value in model.named_parameters()
               if any(name.startswith(prefix + '.') for prefix in prefixes)}
    epoch = int(config['utility_warmup_end']) + 1 if experiment != 'baseline' else 1
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, environment['precision']):
        outputs = model(images, targets=moved_targets, epoch=epoch, profile=config['profile'])
        losses = criterion(outputs, moved_targets, epoch=epoch, profile=config['profile'])
        loss = sum(losses.values())
    if not torch.isfinite(loss).item():
        raise FloatingPointError('OCR-free smoke produced non-finite loss')
    loss.backward()
    gradients = {prefix: sum(float(parameter.grad.detach().abs().sum())
                            for name, parameter in model.named_parameters()
                            if name.startswith(prefix + '.') and parameter.grad is not None)
                 for prefix in prefixes}
    if any(value <= 0 or not math.isfinite(value) for value in gradients.values()):
        raise RuntimeError(f'New module lacks finite nonzero smoke gradient: {gradients}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['clip_grad_norm']), error_if_nonfinite=True)
    optimizer.step()
    changes = {prefix: sum(float((parameter.detach() - tracked[name]).abs().sum())
                          for name, parameter in model.named_parameters()
                          if name.startswith(prefix + '.') and name in tracked)
               for prefix in prefixes}
    if any(value <= 0 for value in changes.values()):
        raise RuntimeError(f'New module did not update in smoke: {changes}')
    model.eval()
    with torch.no_grad(), autocast_context(device, environment['precision']):
        inferred = model(images, targets=moved_targets, epoch=epoch, profile=config['profile'])
        records = postprocess(inferred, targets, image_size=config['image_size'])
    refined = int(inferred['selected_indices'].shape[1]) if 'selected_indices' in inferred else 0
    if experiment in ('detail', 'combined') and refined != int(config['detail_queries']):
        raise RuntimeError(f'Expected exactly {config["detail_queries"]} refined queries, got {refined}')
    if experiment == 'context' and refined:
        raise RuntimeError('Context-only experiment unexpectedly ran the detail branch')
    value = {'smoke_passed': True, 'experiment': experiment, 'smoke_epoch': epoch,
             'reference_type': 'pseudo', 'ocr_required': False, 'real_optimizer_update': True,
             'new_module_gradient_absolute_sum': gradients, 'new_module_parameter_change': changes,
             'gpu_verified': device.type == 'cuda', 'test_only': environment['test_only'],
             'refined_queries_per_image': refined, 'queries_replaced_not_appended': True,
             'loss': float(loss.detach()), 'loss_components': {k: float(v.detach()) for k, v in losses.items()},
             'module_diagnostics': criterion.last_diagnostics,
             'images': len(records), 'fields': sum(len(item['fields']) for item in records),
             'trained_metrics_available': False, 'memory': gpu_memory_report(device),
             'source_sha256': digest_json(source_identity(root)), 'weight_loading': report,
             'smoke_scope': 'one ephemeral optimizer update; formal training restarts from the same COCO weights'}
    write_json(output / f'smoke_{experiment}.json', value)
    log(value)
    del model, criterion, optimizer, loader, images, moved_targets, outputs, inferred, loss, losses, tracked
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return value


def train_positive_report(root, config):
    from .data import load_references
    refs = load_references(root, 'train', config['profile'])
    counts = {str(category): 0 for category in range(1, 5)}
    images = {str(category): set() for category in range(1, 5)}
    for annotation in refs['annotations']:
        category = str(annotation['category_id'])
        if not annotation.get('ignore') and category in counts:
            counts[category] += 1
            images[category].add(annotation['image_id'])
    return {'source': 'existing frozen training pseudo-positive boxes only', 'ocr_required': False,
            'pseudo_positive_boxes': counts, 'images_per_category': {k: len(v) for k, v in images.items()},
            'ignore_regions_are_not_sentence_negatives': True}


def _collect_completed(root, config, metadata, environment, output, results):
    for arm in ARMS:
        if arm in results:
            continue
        path = output / arm / 'completion.json'
        if path.exists():
            # run_model checks the full immutable identity and best checksum before reuse.
            results[arm] = run_model(root, config, metadata, environment, arm, output_root=output)
    return results


def evaluate_full_test(root, config, environment, output, results):
    from .data import load_references
    from .metrics import evaluate
    from .models import build_model
    if (config['profile'] != 'full' or set(results) != set(ARMS) or
            environment.get('test_only') or environment.get('requested_device') != 'cuda' or
            config.get('diagnostic_batches')):
        return False
    loader = make_loader(root, config, 'test', num_workers=int(config['num_workers']),
                         generator=torch.Generator().manual_seed(int(config['seed']) + 2))
    references = references_for_loader(load_references(root, 'test', 'full'), loader)
    summaries = {}
    for arm in ARMS:
        result = results[arm]
        folder = output / arm
        checkpoint = Path(result.get('checkpoint_path', folder / 'best.pt'))
        if sha256(checkpoint) != result['best_sha256']:
            raise RuntimeError(f'{arm}: locked checkpoint SHA mismatch before test')
        path = folder / 'test_summary.json'
        if path.exists():
            saved = read_json(path)
            if (saved.get('best_sha256') != result['best_sha256'] or
                    saved.get('threshold') != result['threshold'] or
                    saved.get('experiment_fingerprint') != result['fingerprint']):
                raise RuntimeError(f'{arm}: old test result belongs to another checkpoint')
        else:
            model, _ = build_model(arm, load_pretrained=False, role_enabled=[False] * 4)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=True)
            model.to(environment['requested_device'])
            records, seconds = predict(model, loader, torch.device(environment['requested_device']), config,
                                       arm, result['best_epoch'], environment['precision'], split='test')
            metrics = evaluate(records, references, threshold=result['threshold'])
            saved = {'reference_type': 'pseudo', 'split': 'test', 'threshold_selection_split': 'val',
                     'threshold': result['threshold'], 'best_sha256': result['best_sha256'],
                     'experiment_fingerprint': result['fingerprint'],
                     'metrics_test': metrics, 'inference_seconds': seconds}
            write_json(folder / 'predictions_test.json', records)
            write_json(folder / 'metrics_test.json', metrics)
            write_json(path, saved)
            print_metrics(arm, 'test', result['best_epoch'], result['threshold'], metrics)
            del model
            if environment['requested_device'] == 'cuda':
                torch.cuda.empty_cache()
        summaries[arm] = saved
    summary_reports(output, summaries, split='test')
    return True


def status(root):
    launcher_path = root / 'runs/ablation_launcher.json'
    value = read_json(launcher_path) if launcher_path.exists() else {'state': 'not_started'}
    if value.get('pid'):
        try:
            os.kill(int(value['pid']), 0)
            value['process_exists'] = True
        except (ProcessLookupError, PermissionError):
            value['process_exists'] = False
    value['models'] = {}
    for base in ('ablations_v1', 'cpu_ablation_diagnostics'):
        for item in (root / 'runs' / base).glob('*/*/status.json'):
            value['models'][str(item.relative_to(root))] = read_json(item)
    log(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description='OCR-free controlled four-arm RT-DETR experiments')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare', 'smoke', 'run', 'status'), default='run')
    parser.add_argument('--profile', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--experiment', choices=(*ARMS, 'all'), default='all')
    parser.add_argument('--baseline-source', type=Path, default=None,
                        help='read-only completed legacy baseline; default runs/experiments/PROFILE/baseline')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--precision', choices=('bf16', 'float32'), default='bf16')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--diagnostic-batches', type=int)
    parser.add_argument('--diagnostic-epochs', type=int)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == 'status':
        return status(root)
    config = load_config(root, args.profile)
    for key in ('diagnostic_batches', 'diagnostic_epochs'):
        value = getattr(args, key)
        if value is not None:
            if not args.test_only or args.device != 'cpu' or value < 1:
                parser.error('diagnostic overrides require --test-only --device cpu and positive values')
            config['epochs' if key == 'diagnostic_epochs' else key] = value
            config['num_workers'] = 0
    output = output_root(root, args.profile, args.test_only)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        try:
            from .data import prepare_data
            environment = inspect_environment(args.device, args.precision, args.test_only)
            torch.set_num_threads(4)
            metadata = prepare_data(root, args.profile)
            write_json(output / 'environment.json', environment)
            write_json(output / 'positive_supervision.json', train_positive_report(root, config))
            log({'stage': 'ablation_prepare', 'version': VERSION, 'profile': args.profile,
                 'output': str(output), 'ocr_required': False, 'reference_type': 'pseudo'})
            selected = ARMS if args.experiment == 'all' else (args.experiment,)
            if args.stage == 'prepare':
                return metadata
            environment = verify_cuda_compute(environment)
            write_json(output / 'environment.json', environment)
            if args.stage == 'smoke':
                return {arm: smoke(root, config, environment, arm, output) for arm in selected}
            if args.device == 'cpu' and not (args.test_only and config.get('diagnostic_batches') and args.diagnostic_epochs):
                raise RuntimeError('CPU run requires --test-only --diagnostic-batches N --diagnostic-epochs N')
            results = {}
            legacy = args.baseline_source or (root / 'runs/experiments' / args.profile / 'baseline')
            new_baseline = output / 'baseline/completion.json'
            if args.experiment != 'baseline':
                if new_baseline.exists():
                    results['baseline'] = run_model(root, config, metadata, environment, 'baseline', output_root=output)
                elif legacy.exists() or args.baseline_source is not None:
                    results['baseline'] = import_legacy_baseline(root, config, metadata, environment, legacy, output)
                elif (output / 'baseline/baseline_import.json').exists():
                    raise RuntimeError('Imported baseline source disappeared; preserve/reconnect its original directory')
            for index, arm in enumerate(selected):
                if arm == 'baseline' and arm in results:
                    continue
                if arm == 'baseline' and (output / 'baseline/baseline_import.json').exists():
                    raise RuntimeError('This namespace already has an immutable imported baseline; use it or choose a new project copy')
                smoke_path = output / f'smoke_{arm}.json'
                previous = read_json(smoke_path) if smoke_path.exists() else {}
                valid = (previous.get('smoke_passed') and previous.get('source_sha256') == digest_json(source_identity(root))
                         and (previous.get('gpu_verified') or args.test_only))
                if not valid:
                    smoke(root, config, environment, arm, output)
                results[arm] = run_model(root, config, metadata, environment, arm,
                                         queue_models=len(selected) - index - 1, output_root=output)
                _collect_completed(root, config, metadata, environment, output, results)
                summary_reports(output, results)
                export_results(root, output, args.profile)
            _collect_completed(root, config, metadata, environment, output, results)
            summary_reports(output, results)
            tested = evaluate_full_test(root, config, environment, output, results)
            failure_path = output / 'failure.json'
            if failure_path.exists():
                previous_failure = read_json(failure_path)
                previous_failure['resolved_by_successful_resume'] = True
                previous_failure['resolution'] = 'The same immutable experiment identity completed successfully.'
                write_json(output / 'failure_previous.json', previous_failure)
                failure_path.unlink()
            write_json(output / 'pipeline_status.json', {'state': 'complete', 'experiments': list(results),
                       'reference_type': 'pseudo', 'ocr_required': False, 'test_evaluated': tested})
            export_results(root, output, args.profile)
            return results
        except BaseException as error:
            failure = {'state': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                       'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                       'pid': os.getpid(), 'resume': 'rerun exactly the same start_ablation.sh command',
                       'original_experiments_untouched': True}
            write_json(output / 'failure.json', failure)
            write_json(output / 'pipeline_status.json', failure)
            export_results(root, output, args.profile)
            log(failure)
            raise


if __name__ == '__main__':
    main()

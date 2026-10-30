"""One entry point for reproducible baseline/final experiments and recovery."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
import traceback

import torch
from torch.utils.data import DataLoader
import yaml

from .common import ROOT, SEED, digest_json, log, read_json, save_checkpoint, seed_everything, sha256, write_json
from .environment import inspect_environment, verify_cuda_compute
from .training import (autocast_context, build_optimizer, build_scheduler, capture_rng,
                       flatten_metrics, move_to_device, predict, restore_rng, seed_worker,
                       selection_key, should_validate, train_epoch, write_history_csv)


def load_config(root, profile):
    value = yaml.safe_load((root / 'configs' / f'{profile}.yaml').read_text(encoding='utf-8'))
    if value.get('profile') != profile:
        raise RuntimeError('配置profile与请求不一致。')
    return value


def result_root(root, profile, test_only=False):
    return root / 'runs' / ('cpu_diagnostics' if test_only else 'experiments') / profile


def source_identity(root):
    files = []
    for directory in ('cgis', 'configs', 'vendor'):
        base = root / directory
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.json') and '__pycache__' not in path.parts:
                files.append(path)
    for pattern in ('*.sh', 'requirements*.lock', 'scripts/*.sh'):
        files.extend(root.glob(pattern))
    return {str(path.relative_to(root)).replace('\\', '/'): sha256(path) for path in sorted(set(files))}


def data_identity(root, metadata):
    files = []
    for directory in ('annotations', 'splits', 'source', 'manifests', 'profiles'):
        base = root / 'data' / directory
        if base.exists():
            files.extend(path for path in base.rglob('*') if path.is_file() and path.suffix in ('.json', '.txt'))
    files.extend((root / 'data').glob('*manifest*.json'))
    return {
        'files': {str(path.relative_to(root)).replace('\\', '/'): sha256(path) for path in sorted(set(files))},
        'splits': metadata.get('splits', {}),
        'image_manifest': metadata.get('image_manifest_sha256', metadata.get('manifest_sha256')),
    }


def experiment_identity(root, config, metadata, environment, experiment, role_report=None):
    weight = root / 'weights' / 'rtdetr_r18vd_coco.pth'
    if not weight.is_file():
        raise FileNotFoundError(f'缺少官方COCO预训练权重：{weight}')
    role = None
    if role_report is not None:
        role_file = Path(role_report['role_file'])
        role = {'enabled_categories': role_report['enabled_categories'], 'sha256': sha256(role_file)}
    return {
        'experiment': experiment, 'config': config, 'source': source_identity(root),
        'data': data_identity(root, metadata), 'pretrained_sha256': sha256(weight), 'role_supervision': role,
        'runtime': {key: environment[key] for key in ('python', 'torch', 'torchvision', 'numpy',
                                                      'cuda_runtime', 'requested_device', 'precision', 'test_only')},
    }


def ensure_manifest(output, identity):
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = digest_json(identity)
    path = output / 'manifest.json'
    if path.exists():
        previous = read_json(path)
        if previous['fingerprint'] != fingerprint:
            raise RuntimeError(f'输入、代码、配置、角色标签或环境发生变化，不能混用旧训练：{output}。'
                               '请保留原结果并使用新的项目副本/实验目录。')
        return previous
    manifest = {'fingerprint': fingerprint, 'identity': identity, 'reference_type': 'pseudo',
                'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'trained': False, 'gpu_verified': False, 'complete': False}
    write_json(path, manifest)
    return manifest


def make_loader(root, config, split, augment=False, role_file=None, generator=None, num_workers=None):
    from .data import DictionaryDataset, collate_fn
    dataset = DictionaryDataset(root, split=split, profile=config['profile'], augment=augment,
                                role_file=role_file, image_size=int(config['image_size']))
    if len(dataset) == 0:
        raise RuntimeError(f'{split}划分为空。')
    if config.get('diagnostic_batches'):
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), int(config['diagnostic_batches']) * int(config['batch_size']))))
    workers = int(config['num_workers']) if num_workers is None else num_workers
    return DataLoader(dataset, batch_size=int(config['batch_size']), shuffle=augment,
                      num_workers=workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
                      drop_last=False, persistent_workers=False, generator=generator,
                      worker_init_fn=seed_worker)


def print_metrics(experiment, split, epoch, threshold, metrics):
    log({'stage': 'metrics', 'reference_type': 'pseudo', 'experiment': experiment,
         'split': split, 'epoch': epoch, 'threshold': threshold,
         'per_class': metrics['per_class'], 'macro_pseudo_f1': metrics['overall_macro_f1'],
         'ignore_aware_pseudo_ap': metrics.get('ap')})


def references_for_loader(references, loader):
    """Keep diagnostic subsetting out of metric denominators and normal runs."""
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        selected = [dataset.dataset.images[int(index)] for index in dataset.indices]
    else:
        selected = dataset.images
    ids = {int(image['id']) for image in selected}
    return {**references,
            'images': [image for image in references['images'] if int(image['id']) in ids],
            'annotations': [annotation for annotation in references['annotations']
                            if int(annotation['image_id']) in ids]}


def gpu_memory_report(device):
    if device.type != 'cuda':
        return {'allocated_gib': None, 'reserved_gib': None}
    return {'allocated_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3,
            'reserved_gib': torch.cuda.max_memory_reserved(device) / 1024 ** 3}


def run_model(root, config, metadata, environment, experiment, role_report=None, queue_models=0,
              output_root=None):
    from .data import load_references
    from .metrics import select_threshold
    from .models import build_model
    from .losses import DetectionCriterion

    output = (Path(output_root) if output_root is not None else
              result_root(root, config['profile'], environment['test_only'])) / experiment
    identity = experiment_identity(root, config, metadata, environment, experiment, role_report)
    manifest = ensure_manifest(output, identity)
    fingerprint = manifest['fingerprint']
    done_path = output / 'completion.json'
    if done_path.exists():
        done = read_json(done_path)
        if done.get('fingerprint') != fingerprint or sha256(output / 'best.pt') != done.get('best_sha256'):
            raise RuntimeError('已完成结果的指纹或best.pt校验不一致。')
        log({'stage': 'reuse_complete', 'experiment': experiment, 'output': str(output)})
        return done

    device = torch.device(environment['requested_device'])
    role_enabled = role_report['enabled_categories'] if role_report else [False] * 4
    seed_everything(int(config['seed']))
    model, weight_report = build_model(experiment, pretrained_path=root / 'weights' / 'rtdetr_r18vd_coco.pth',
                                       role_enabled=role_enabled, load_pretrained=True)
    model = model.to(device)
    criterion = DetectionCriterion(model=model).to(device)
    write_json(output / 'weight_loading_report.json', weight_report)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    # Extra module initialization must not shift the image order/augmentation seed.
    seed_everything(int(config['seed']))
    generator = torch.Generator().manual_seed(int(config['seed']))
    val_generator = torch.Generator().manual_seed(int(config['seed']) + 1)
    train_loader = make_loader(root, config, 'train', augment=True,
                               role_file=role_report['role_file'] if role_report else None, generator=generator)
    val_loader = make_loader(root, config, 'val', generator=val_generator)
    references = references_for_loader(load_references(root, 'val', config['profile']), val_loader)
    history, timing = [], {}
    completed_epoch, pending_validation, best_key = 0, False, None
    best_threshold, best_epoch, best_metrics = None, None, None
    last_path = output / 'last.pt'
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location='cpu')
        if checkpoint.get('fingerprint') != fingerprint:
            raise RuntimeError('检查点指纹与当前工程不同。')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        completed_epoch = int(checkpoint['completed_epoch'])
        pending_validation = bool(checkpoint['pending_validation'])
        history, timing = checkpoint['history'], checkpoint.get('timing', {})
        best_key = tuple(checkpoint['best_key']) if checkpoint.get('best_key') is not None else None
        best_threshold, best_epoch, best_metrics = (checkpoint.get(key) for key in ('best_threshold', 'best_epoch', 'best_metrics'))
        restore_rng(checkpoint['rng'], generator)
        log({'stage': 'resume', 'experiment': experiment, 'completed_epoch': completed_epoch,
             'next_epoch': completed_epoch + 1, 'pending_validation': pending_validation})

    def save_last():
        save_checkpoint(last_path, {
            'fingerprint': fingerprint, 'experiment': experiment, 'reference_type': 'pseudo',
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'rng': capture_rng(generator), 'completed_epoch': completed_epoch, 'pending_validation': pending_validation,
            'best_key': best_key, 'best_threshold': best_threshold, 'best_epoch': best_epoch,
            'best_metrics': best_metrics, 'history': history, 'timing': timing,
            'role_enabled': role_enabled, 'config': config,
        })

    def validate_current():
        nonlocal pending_validation, best_key, best_threshold, best_epoch, best_metrics
        records, seconds = predict(model, val_loader, device, config, experiment, completed_epoch,
                                   precision=environment['precision'])
        timing['validation_seconds'] = seconds
        threshold, metrics, grid = select_threshold(records, references)
        # Update one entry per epoch, including recovery after interruption during evaluation.
        item = next(row for row in history if row['epoch'] == completed_epoch)
        item.update(threshold=float(threshold), metrics=metrics, validation_seconds=seconds)
        key = selection_key(metrics, completed_epoch)
        if best_key is None or key > best_key:
            best_key, best_threshold, best_epoch, best_metrics = key, float(threshold), completed_epoch, metrics
            save_checkpoint(output / 'best.pt', {
                'fingerprint': fingerprint, 'experiment': experiment, 'model': model.state_dict(),
                'epoch': completed_epoch, 'threshold': best_threshold, 'role_enabled': role_enabled,
                'reference_type': 'pseudo', 'config': config, 'metrics_val': metrics,
            })
            write_json(output / 'predictions_val.json', records)
            write_json(output / 'metrics_val.json', metrics)
            write_json(output / 'threshold_grid.json', grid)
        pending_validation = False
        write_json(output / 'history.json', history)
        write_history_csv(output / 'history.csv', history)
        write_json(output / 'checkpoint_selection.json', {
            'reference_type': 'pseudo', 'selection_split': 'val', 'epoch': best_epoch,
            'threshold': best_threshold, 'metrics': best_metrics, 'tie_break': 'macro_f1, ap50_95, earliest_epoch',
        })
        save_last()
        print_metrics(experiment, 'val', completed_epoch, threshold, metrics)

    if pending_validation:
        validate_current()
    for epoch in range(completed_epoch + 1, int(config['epochs']) + 1):
        write_json(output / 'status.json', {'state': 'training', 'epoch': epoch,
                   'epochs': config['epochs'], 'pid': os.getpid(), 'reference_type': 'pseudo'})
        stats = train_epoch(model, criterion, train_loader, optimizer, device, config, epoch, experiment,
                            precision=environment['precision'], timing=timing, queue_models=queue_models)
        scheduler.step()
        completed_epoch = epoch
        pending_validation = should_validate(epoch, config)
        history.append({'epoch': epoch, 'train': stats, 'lr_next_epoch': scheduler.get_last_lr()})
        timing['last_epoch_seconds'] = stats['seconds']
        save_last()
        if pending_validation:
            write_json(output / 'status.json', {'state': 'validating', 'epoch': epoch, 'pid': os.getpid()})
            validate_current()
        else:
            write_json(output / 'history.json', history)
            write_history_csv(output / 'history.csv', history)
        manifest.update(trained=True, gpu_verified=device.type == 'cuda')
        write_json(output / 'manifest.json', manifest)
    if best_epoch is None:
        raise RuntimeError('训练完成但不存在有效验证权重。')
    done = {'fingerprint': fingerprint, 'experiment': experiment, 'reference_type': 'pseudo',
            'complete': True, 'epochs': config['epochs'], 'best_epoch': best_epoch,
            'threshold': best_threshold, 'metrics_val': best_metrics, 'best_sha256': sha256(output / 'best.pt'),
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'trainable_parameters': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            'role_enabled': role_enabled, 'gpu_verified': device.type == 'cuda', 'memory': gpu_memory_report(device),
            'test_only': environment['test_only'], 'timing': timing}
    validation_images = min(len(references['images']), int(config.get('diagnostic_batches', 10 ** 9)) * int(config['batch_size']))
    done['timing']['validation_pipeline_ms_per_image'] = (
        1000.0 * timing['validation_seconds'] / max(1, validation_images))
    done['timing']['latency_scope'] = 'validation pipeline, includes image loading and output conversion; excludes AP evaluation'
    write_json(done_path, done)
    write_json(output / 'status.json', {'state': 'complete', 'epoch': completed_epoch, 'pid': os.getpid()})
    manifest.update(complete=True, trained=True, gpu_verified=device.type == 'cuda')
    write_json(output / 'manifest.json', manifest)
    del model, criterion, optimizer, train_loader, val_loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return done


def summary_reports(output, results, split='val'):
    import csv
    rows = []
    for experiment, result in results.items():
        metrics = result.get(f'metrics_{split}', result.get('metrics'))
        if metrics is not None:
            rows.append({'experiment': experiment, 'reference_type': 'pseudo',
                         'split': split, 'threshold': result['threshold'], **flatten_metrics(metrics)})
    write_json(output / f'comparison_{split}.json', {'reference_type': 'pseudo', 'results': results})
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f'comparison_{split}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# {len(rows)} 组模型 {split} 伪标签检测对比', '',
             '这些指标衡量与冻结伪标签的一致程度，不代表人工核验准确率。', '',
             '| 模型 | Macro Pseudo-F1 | Pseudo-AP50 | Pseudo-AP50:95 |',
             '|---|---:|---:|---:|']
    for row in rows:
        values = [row.get(key) for key in ('macro_pseudo_f1', 'pseudo_ap50', 'pseudo_ap50_95')]
        rendered = ['N/A' if value is None else f'{value:.4f}' for value in values]
        lines.append('| ' + row['experiment'] + ' | ' + ' | '.join(rendered) + ' |')
    (output / f'comparison_{split}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def export_results(root, output, profile):
    """Downloadable diagnostics; never include optimizer/model weights or raw OCR cache."""
    files = [path for path in output.rglob('*') if path.is_file() and path.suffix in ('.json', '.csv', '.md')]
    for base in (root / 'data' / 'weak_roles' / profile, root / 'reports'):
        if base.exists():
            files.extend(path for path in base.rglob('*') if path.is_file()
                         and path.suffix in ('.json', '.csv', '.md') and 'ocr_cache' not in path.parts)
    archive = output / 'results.tar.gz'
    temporary = output / 'results.tar.gz.tmp'
    with tarfile.open(temporary, 'w:gz') as handle:
        for path in sorted(set(files)):
            handle.add(path, arcname=str(path.relative_to(root)).replace('\\', '/'), recursive=False)
    temporary.replace(archive)
    log({'stage': 'results_export', 'archive': str(archive), 'files': len(set(files)),
         'note': 'no weights/checkpoints/raw OCR cache; pseudo-label metrics only'})
    return archive


def evaluate_locked_test(root, config, environment, results):
    """Only run test when both full models finished; never select on test."""
    from .data import load_references
    from .metrics import evaluate
    from .models import build_model
    if config['profile'] != 'full' or set(results) != {'baseline', 'final'}:
        return
    output = result_root(root, 'full', environment['test_only'])
    references = load_references(root, 'test', 'full')
    loader = make_loader(root, config, 'test', generator=torch.Generator().manual_seed(SEED + 2))
    test_results = {}
    for experiment, result in results.items():
        test_output = output / experiment
        checkpoint_path = test_output / 'best.pt'
        if sha256(checkpoint_path) != result['best_sha256']:
            raise RuntimeError('测试前锁定权重校验失败。')
        existing = test_output / 'test_summary.json'
        if existing.exists():
            saved = read_json(existing)
            if saved.get('best_sha256') != result['best_sha256'] or saved.get('threshold') != result['threshold']:
                raise RuntimeError('既有测试结果不属于当前锁定权重/阈值。')
            test_results[experiment] = saved
            continue
        model, _ = build_model(experiment, load_pretrained=False, role_enabled=result['role_enabled'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(environment['requested_device'])
        records, seconds = predict(model, loader, torch.device(environment['requested_device']), config,
                                   experiment, result['best_epoch'], environment['precision'], split='test')
        metrics = evaluate(records, references, threshold=result['threshold'])
        saved = {'reference_type': 'pseudo', 'split': 'test', 'threshold_selection_split': 'val',
                 'threshold': result['threshold'], 'best_sha256': result['best_sha256'],
                 'metrics_test': metrics, 'inference_seconds': seconds}
        write_json(test_output / 'predictions_test.json', records)
        write_json(test_output / 'metrics_test.json', metrics)
        write_json(existing, saved)
        print_metrics(experiment, 'test', result['best_epoch'], result['threshold'], metrics)
        test_results[experiment] = saved
        del model
        if environment['requested_device'] == 'cuda':
            torch.cuda.empty_cache()
    summary_reports(output, test_results, split='test')


def smoke(root, config, environment, experiment='baseline', role_report=None):
    from .data import DictionaryDataset, collate_fn
    from .metrics import postprocess
    from .models import build_model
    from .losses import DetectionCriterion
    device = torch.device(environment['requested_device'])
    seed_everything(int(config['seed']))
    role_enabled = role_report['enabled_categories'] if role_report else [False] * 4
    model, weight_report = build_model(experiment, pretrained_path=root / 'weights' / 'rtdetr_r18vd_coco.pth',
                                       load_pretrained=True, role_enabled=role_enabled)
    model = model.to(device).train()
    criterion = DetectionCriterion(model=model).to(device)
    smoke_epoch = int(config['utility_warmup_end']) + 1 if experiment == 'final' else 1
    if experiment == 'final':
        if not role_report or not any(role_enabled):
            raise RuntimeError('完整模型冒烟必须使用达到覆盖要求的真实训练弱角色对。')
        dataset = DictionaryDataset(root, split='train', profile=config['profile'], augment=False,
                                    role_file=role_report['role_file'], image_size=int(config['image_size']))
        eligible = [index for index, image in enumerate(dataset.images)
                    if any(role_enabled[pair['label']] for pair in dataset.role_pairs.get(image['file_name'], []))]
        if not eligible:
            raise RuntimeError('完整模型冒烟找不到含有效角色对的训练图。')
        indices = eligible[:int(config['batch_size'])]
        if len(indices) < int(config['batch_size']):
            indices += [index for index in range(len(dataset)) if index not in indices][:int(config['batch_size']) - len(indices)]
        images, targets = collate_fn([dataset[index] for index in indices])
    else:
        loader = make_loader(root, config, 'train', num_workers=0,
                             generator=torch.Generator().manual_seed(int(config['seed'])))
        images, targets = next(iter(loader))
    images, targets_device = images.to(device), move_to_device(targets, device)
    optimizer = build_optimizer(model, config)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, environment['precision']):
        outputs = model(images, targets=targets_device, epoch=smoke_epoch, profile=config['profile'])
        losses = criterion(outputs, targets_device, epoch=smoke_epoch, profile=config['profile'])
        loss = sum(losses.values())
    if not torch.isfinite(loss).item():
        raise FloatingPointError('冒烟训练loss非有限值。')
    loss.backward()
    gradient_sums = {prefix: sum(float(parameter.grad.detach().abs().sum()) for name, parameter in model.named_parameters()
                                 if name.startswith(prefix + '.') and parameter.grad is not None)
                     for prefix in ('refiner', 'role_head', 'gain_head')}
    if experiment == 'final' and any(value <= 0 for value in gradient_sums.values()):
        raise RuntimeError(f'完整模型新增模块未收到有效训练梯度：{gradient_sums}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['clip_grad_norm']), error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    with torch.no_grad(), autocast_context(device, environment['precision']):
        eval_outputs = model(images, targets=targets_device, epoch=smoke_epoch, profile=config['profile'])
        records = postprocess(eval_outputs, targets, image_size=config['image_size'])
    report = {'smoke_passed': True, 'real_optimizer_update': True, 'images': len(records),
              'fields': sum(len(record['fields']) for record in records), 'loss': float(loss.detach()),
              'gpu_verified': device.type == 'cuda', 'test_only': environment['test_only'],
              'memory': gpu_memory_report(device), 'trained_metrics_available': False,
              'experiment': experiment, 'smoke_epoch': smoke_epoch, 'role_enabled': role_enabled,
              'loss_components': {key: float(value.detach()) for key, value in losses.items()},
              'new_module_gradient_absolute_sum': gradient_sums,
              'smoke_scope': 'single ephemeral optimizer update; formal training starts again from COCO',
              'weight_loading': weight_report}
    if experiment == 'final':
        report['refined_queries_per_image'] = int(eval_outputs['selected_indices'].shape[1])
        if report['refined_queries_per_image'] != int(config['detail_queries']):
            raise RuntimeError('完整模型推理细化预算不等于固定64个查询。')
    output = result_root(root, config['profile'], environment['test_only'])
    write_json(output / ('smoke_final_report.json' if experiment == 'final' else 'smoke_report.json'), report)
    log(report)
    del model, criterion, optimizer, loss, losses, outputs, eval_outputs, images, targets_device
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report


@contextmanager
def exclusive_run(root):
    if os.environ.get('CGIS_LOCK_INHERITED') == '1':
        # start.sh acquired fd 9 before prepare/smoke and passed the same open description.
        if os.name != 'nt':
            os.fstat(9)
        yield
        return
    if os.name == 'nt':
        # Direct Windows entry is restricted to diagnostic prepare/smoke; server uses flock.
        yield
        return
    import fcntl
    lock_path = root / 'runs' / '.run.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('已有CGIS准备/训练进程占用锁，不能重复启动。') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def status(root):
    path = root / 'runs' / 'launcher.json'
    launcher = read_json(path) if path.exists() else {'state': 'not_started'}
    pid = launcher.get('pid')
    if pid:
        try:
            os.kill(int(pid), 0)
            launcher['process_exists'] = True
        except (ProcessLookupError, PermissionError):
            launcher['process_exists'] = False
    launcher['models'] = {}
    for item in (root / 'runs').glob('*/*/*/status.json'):
        launcher['models'][str(item.relative_to(root))] = read_json(item)
    log(launcher)
    return launcher


def ensure_ocr_runtime(root, *, allow_install=True):
    """Prepare OCR when entering the final model; baseline never needs this step."""
    root = Path(root).resolve()
    if allow_install:
        log({'stage': 'install_ocr', 'note': 'baseline is retained; preparing the final-model OCR dependency',
             'eta_pipeline': 'pending OCR installation and measurement'})
        subprocess.run(['bash', str(root / 'scripts/install_ocr.sh')], cwd=str(root), check=True)
    paths_file = root / 'reports/runtime_paths.json'
    if paths_file.exists():
        executable = read_json(paths_file).get('tesseract')
        if executable:
            os.environ['CGIS_TESSERACT'] = str(executable)
    from .weak_roles import _engine
    executable, _, information = _engine(root)
    os.environ['CGIS_TESSERACT'] = str(executable)
    return information


def main(argv=None):
    parser = argparse.ArgumentParser(description='CGIS four-field pseudo-label experiments')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=('prepare', 'smoke', 'run', 'status'), default='run')
    parser.add_argument('--profile', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--experiment', choices=('baseline', 'final', 'both'), default='both')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--precision', choices=('bf16', 'float32'), default='bf16')
    parser.add_argument('--test-only', action='store_true', help='CPU diagnostic outputs only; never GPU verification')
    parser.add_argument('--diagnostic-batches', type=int, default=None,
                        help='test-only: bound each split to this many batches, excluded from official runs')
    parser.add_argument('--diagnostic-epochs', type=int, default=None,
                        help='test-only: override epochs for actual local CPU recovery checks')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == 'status':
        return status(root)
    config = load_config(root, args.profile)
    if args.diagnostic_batches is not None or args.diagnostic_epochs is not None:
        if not args.test_only or args.device != 'cpu':
            parser.error('diagnostic overrides require --test-only --device cpu')
        if args.diagnostic_batches is not None:
            if args.diagnostic_batches < 1:
                parser.error('--diagnostic-batches must be positive')
            config['diagnostic_batches'] = args.diagnostic_batches
            config['num_workers'] = 0
        if args.diagnostic_epochs is not None:
            if args.diagnostic_epochs < 1:
                parser.error('--diagnostic-epochs must be positive')
            config['epochs'] = args.diagnostic_epochs
    output = result_root(root, args.profile, args.test_only)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(root):
        try:
            from .data import prepare_data
            environment = inspect_environment(args.device, args.precision, args.test_only)
            torch.set_num_threads(4)
            metadata = prepare_data(root, args.profile)
            write_json(output / 'environment.json', environment)
            log({'stage': 'prepare', 'profile': args.profile, 'reference_type': 'pseudo',
                 'data': {key: value for key, value in metadata.items() if key != 'splits'},
                 'environment': environment, 'output': str(output)})
            if args.stage == 'prepare':
                return metadata
            if args.stage == 'smoke':
                environment = verify_cuda_compute(environment)
                write_json(output / 'environment.json', environment)
                return smoke(root, config, environment)
            if args.device == 'cpu' and not (args.test_only and config.get('diagnostic_batches') and args.diagnostic_epochs):
                raise RuntimeError('CPU训练仅开放 --test-only --diagnostic-batches N --diagnostic-epochs N 限量诊断；正式实验请在4090运行。')
            smoke_report_path = output / 'smoke_report.json'
            previous_smoke = read_json(smoke_report_path) if smoke_report_path.exists() else {}
            valid_smoke = previous_smoke.get('smoke_passed') and (previous_smoke.get('gpu_verified') or args.test_only)
            if not valid_smoke:
                environment = verify_cuda_compute(environment)
                smoke(root, config, environment)
            results = {}
            selected = ('baseline', 'final') if args.experiment == 'both' else (args.experiment,)
            for index, experiment in enumerate(selected):
                role_report = None
                if experiment == 'final':
                    from .weak_roles import prepare_weak_roles
                    ensure_ocr_runtime(root, allow_install=not environment['test_only'])
                    log({'stage': 'weak_roles', 'profile': args.profile,
                         'eta_pipeline': 'pending OCR and final-model measurement'})
                    role_report = prepare_weak_roles(root, args.profile)
                    if not any(role_report['enabled_categories']):
                        raise RuntimeError('所有类别角色监督覆盖不足，完整模型停止；baseline结果保留。')
                    smoke(root, config, environment, experiment='final', role_report=role_report)
                results[experiment] = run_model(root, config, metadata, environment, experiment,
                                                role_report, queue_models=len(selected) - index - 1)
                summary_reports(output, results)
                export_results(root, output, args.profile)
            # Also support completing the other full model in a later invocation.
            if args.profile == 'full':
                for experiment in ('baseline', 'final'):
                    completion = output / experiment / 'completion.json'
                    if experiment not in results and completion.exists():
                        saved = read_json(completion)
                        manifest = read_json(output / experiment / 'manifest.json')
                        if saved.get('fingerprint') != manifest.get('fingerprint'):
                            raise RuntimeError('另一模型完成状态的指纹不一致。')
                        # Recheck its recorded source/config/input identity before admitting test.
                        recorded_identity = manifest['identity']
                        if recorded_identity['source'] != source_identity(root) or recorded_identity['data'] != data_identity(root, metadata):
                            raise RuntimeError('另一模型使用了不同源码或数据，不能自动合并测试。')
                        results[experiment] = saved
                if set(results) == {'baseline', 'final'}:
                    summary_reports(output, results)
                    evaluate_locked_test(root, config, environment, results)
            write_json(output / 'pipeline_status.json', {'state': 'complete', 'experiments': list(results),
                       'reference_type': 'pseudo', 'test_evaluated': args.profile == 'full' and len(results) == 2})
            export_results(root, output, args.profile)
            return results
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                state = 'interrupted'
            else:
                state = 'failed'
            failure = {'state': state, 'type': type(error).__name__, 'error': str(error),
                       'traceback': traceback.format_exc(), 'pid': os.getpid(),
                       'resume': 'rerun the same start.sh command; last complete epoch is restored'}
            coverage_path = root / 'data/weak_roles' / args.profile / 'coverage.json'
            if type(error).__name__ == 'WeakRoleDataError' and coverage_path.exists():
                coverage = read_json(coverage_path)
                if coverage.get('completed') and not any(coverage.get('role_enabled', [])):
                    failure['reason_category'] = 'insufficient_weak_supervision'
                    failure['baseline_complete'] = (output / 'baseline/completion.json').exists()
                    failure['coverage_report'] = str(coverage_path)
                    failure['resume'] = ('Inspect the OCR coverage and rejection report first. '
                                         'An unchanged rerun reuses the same insufficient evidence. '
                                         'Completed baseline results are preserved.')
            write_json(output / 'failure.json', failure)
            write_json(output / 'pipeline_status.json', {'state': state, 'error': str(error)})
            export_results(root, output, args.profile)
            log(failure)
            raise


if __name__ == '__main__':
    main()

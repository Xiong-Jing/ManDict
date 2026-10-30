"""Checks for the isolated 4090 environment; CPU mode is explicitly diagnostic."""
from __future__ import annotations

import platform
import subprocess
import sys

import numpy as np
import torch
import torchvision


EXPECTED = {'python': '3.10', 'torch': '2.1.2', 'torchvision': '0.16.2',
            'numpy': '1.26.4', 'cuda_runtime': '12.1'}


def inspect_environment(device='cuda', precision='bf16', test_only=False, require_ocr=False):
    report = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'numpy': np.__version__,
        'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
        'requested_device': device, 'precision': precision, 'test_only': bool(test_only),
        'gpu_test_passed': False, 'environment_check_passed': False,
    }
    if device == 'cpu':
        if not test_only or precision != 'float32':
            raise RuntimeError('CPU 只用于 --test-only --precision float32 本地诊断，不能标记为4090验证。')
        report['environment_check_passed'] = True
        return report
    if device != 'cuda' or precision != 'bf16':
        raise RuntimeError('正式训练固定使用 --device cuda --precision bf16。')
    actual = {'python': '.'.join(platform.python_version().split('.')[:2]),
              'torch': torch.__version__.split('+')[0],
              'torchvision': torchvision.__version__.split('+')[0],
              'numpy': np.__version__, 'cuda_runtime': torch.version.cuda}
    errors = [f'{name}: expected {value}, got {actual[name]}'
              for name, value in EXPECTED.items() if actual[name] != value]
    if errors:
        raise RuntimeError('环境版本不匹配：' + '; '.join(errors))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请确认租用GPU实例并使用CUDA版PyTorch。')
    properties = torch.cuda.get_device_properties(0)
    report.update(device_name=properties.name, device_memory_bytes=properties.total_memory,
                  bf16_supported=torch.cuda.is_bf16_supported())
    if properties.total_memory < 22 * 1024 ** 3:
        raise RuntimeError('正式配置需要至少22GiB实际GPU显存。')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('此GPU不支持BF16。')
    if require_ocr:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True, check=True)
        first_line = result.stdout.splitlines()[0]
        if not first_line.startswith('tesseract 5.'):
            raise RuntimeError(f'需要Tesseract 5，当前：{first_line}')
        report['tesseract'] = first_line
    report['environment_check_passed'] = True
    return report


def verify_cuda_compute(report):
    """A real CUDA forward/backward, not a driver/version-only check."""
    if report['requested_device'] != 'cuda':
        return report
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = (x @ x.T).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item() or not torch.isfinite(x.grad).all().item():
        raise RuntimeError('实际CUDA BF16前向/反向出现非有限值。')
    report['gpu_test_passed'] = True
    del loss, x
    torch.cuda.empty_cache()
    return report

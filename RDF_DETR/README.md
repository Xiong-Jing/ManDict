# Four-Field Dictionary Detection

This anonymous code package accompanies a four-category dictionary-field detection study. The detector identifies Chinese headwords, parts of speech, Manchu word forms and romanized transcriptions. It produces bounding boxes, class labels and confidence scores; this experiment does not transcribe text or detect example sentences.

The four experimental configurations use the same adapted RT-DETR-R18-vd backbone, input pipeline and detection supervision:

| Experiment | Description |
| --- | --- |
| `baseline` | Four-class adapted RT-DETR-R18-vd |
| `detail` | Baseline with detail-gain selection and local refinement |
| `context` | Baseline with context-guided classification and contrastive supervision |
| `combined` | Both modules, the complete model evaluated in this study |

The current study uses `combined`; the legacy `final` branch in the shared model code is not the model used for the reported four-arm experiments. The legacy OCR preparation component is excluded, so the standalone `cgis.runner` CLI is not supported here; use `cgis.ablation_runner` as shown below. The four arms start from the same COCO initialization rather than continuing a trained baseline.

## Contents

- `cgis/`: detector, data loader, training, losses, metrics and the four-arm runner.
- `configs/`: pilot and full training settings.
- `vendor/`: pinned RT-DETR source, runtime adaptation, provenance and third-party license.
- `tests/`: synthetic CPU tests. They do not establish model accuracy or reproduce the full experiment.

The dataset, annotations, checkpoints and pretrained weights are **not included**. No network download or training is performed merely by importing the code. Full reproducibility requires access to the actual dataset and frozen pseudo-label resources described below; the authors' resources are not supplied in this package.

## Environment

The experiment used Linux, Python 3.10, PyTorch 2.1.2, TorchVision 0.16.2, CUDA runtime 12.1 and NumPy 1.26.4. A CUDA GPU with BF16 support is required for the full run. To create an isolated environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.1.2' 'torchvision==0.16.2' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.lock
```

For the synthetic tests, additionally install `pytest` in the isolated environment.

## External data and weights

Supply the original prepared resources with their exact manifest and referenced files:

```text
data/
  import_manifest.json
  images/*.png
  splits/{pilot,full}/{train,val,test}.txt
  derived/{train,val,test}.json
  derived/{train,val,test}_id_mapping.json
  frozen/pseudo_four_fields_{train,val,test}.json
  frozen/pseudo_ignore_{train,val,test}.json
weights/
  rtdetr_r18vd_coco.pth
```

`import_manifest.json` contains the `files` mapping of relative file paths to SHA-256 digests and the `image_sha256` mapping from image filenames to digests. All entries are verified before training. The supplied resources must also satisfy the fixed image counts: full train/validation/test = 9,275/1,159/1,159, with 11,593 unique PNG images; pilot train/validation/test = 500/100/0. Split files contain one image filename per line. Train, validation and test splits must be disjoint.

Each derived JSON contains `images`, `annotations` and `categories` arrays. Images specify `id`, `file_name`, `width` and `height`. Annotations use `image_id`, 1-based `category_id`, original-image `[x,y,width,height]` boxes, `ignore`, and `pseudo_weight` for positive annotations. Category IDs 1–4 correspond to the four classes in the order listed above. Frozen source annotations and ID mappings are also required where listed by the manifest. Ignore regions are uncertainty masks, not confirmed negative sentence labels. Data schema alone does not reconstruct the original pseudo labels or their source; obtaining the actual prepared resources remains a prerequisite.

The public COCO initialization is the official `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` release, to be placed at `weights/rtdetr_r18vd_coco.pth`. Its recorded SHA-256 is `3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0`. It can be obtained from the [official RT-DETR release](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth). This is an upstream pretrained weight, not a trained checkpoint from this study.

## Training and evaluation

After preparing the exact external resources and activating the environment, run from the repository root:

```bash
# 500 training and 100 validation images, 20 epochs per arm.
python -m cgis.ablation_runner --stage run --profile pilot --experiment all

# 9,275 training and 1,159 validation images, 60 epochs per arm.
# After all four arms complete, the runner evaluates the fixed test split.
python -m cgis.ablation_runner --stage run --profile full --experiment all
```

The runner accepts `baseline`, `detail`, `context`, `combined` or `all` for `--experiment`. `configs/pilot.yaml` and `configs/full.yaml` contain the settings, including 896-pixel letterbox input, physical batch 2, gradient accumulation 4 and BF16. The full runner selects each model's checkpoint and confidence threshold on validation data before evaluating test data. For independent comparison, use a clean results directory and the same configuration, data, seed and COCO initialization for all arms. The anonymous package has a different source fingerprint from the original workspace, so original experiment checkpoints and imported baseline results cannot be assumed to resume or pass identity checks here.

Outputs are written under `runs/ablations_v1/<profile>/`: histories, selected checkpoints, predictions, JSON/CSV/Markdown comparison tables, and, when all full arms finish, locked test metrics. Results are measured against frozen pseudo labels: report them as Pseudo-P/R/F1 and ignore-aware Pseudo-AP rather than human-verified accuracy. The validation threshold is selected per model. Loss definitions vary across arms and stages, so absolute total loss is not a direct performance ranking.

To run the included CPU tests without a dataset or pretrained checkpoint:

```bash
python -m pytest tests -q
```

These tests use synthetic inputs and random model initialization. They do not validate GPU execution, dataset availability or experimental performance.

## Third-party attribution

The RT-DETR subset and its runtime adaptations are pinned to upstream commit `df2a96c8c3a098567a1149bf0248da6efe6d495d`. See `vendor/UPSTREAM.json`, `vendor/RUNTIME_PATCHES.json`, `vendor/COMPATIBILITY.md` and `vendor/LICENSE` for provenance and licensing. The upstream source retains its original attribution and notices.

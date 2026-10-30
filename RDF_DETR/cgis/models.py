"""Pinned official RT-DETR-R18-vd and the experimental RDF-DETR extensions.

The official operators are vendored unchanged apart from isolated registry imports.
Letterbox validity, hidden-query exposure and experimental modules live here; see
vendor/COMPATIBILITY.md. No component downloads weights during forward/build.
"""
from __future__ import annotations

from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from vendor.rtdetr_runtime.presnet import PResNet
from vendor.rtdetr_runtime.hybrid_encoder import HybridEncoder
from vendor.rtdetr_runtime.rtdetr_decoder import RTDETRTransformer, MLP
from vendor.rtdetr_runtime.denoising import get_contrastive_denoising_training_group
from vendor.rtdetr_runtime.box_ops import box_cxcywh_to_xyxy
from .context import CONTEXT_POLICY, select_context_targets


def stage_for(epoch: int, profile: str) -> str:
    warm, utility = (3, 5) if profile == 'pilot' else (10, 15)
    return 'fine_warmup' if epoch <= warm else ('gain_warmup' if epoch <= utility else 'joint')


def valid_rectangles(images, targets=None):
    """Valid xyxy in canvas pixels; centered letterbox is explicitly supported."""
    height, width = images.shape[-2:]
    values = []
    for index in range(len(images)):
        target = targets[index] if targets is not None else {}
        if 'valid_box' in target:
            value = torch.as_tensor(target['valid_box'], device=images.device, dtype=torch.float32)
        else:
            valid = target.get('valid_size', target.get('resized_size', (height, width)))
            valid = torch.as_tensor(valid, device=images.device, dtype=torch.float32)
            value = torch.stack((valid.new_zeros(()), valid.new_zeros(()), valid[1], valid[0]))
        if value.shape != (4,) or not bool(((value[2:] - value[:2]) > 0).all()):
            raise ValueError('Each valid_box must contain positive-area xyxy canvas coordinates')
        values.append(value)
    return torch.stack(values)


def feature_valid_mask(feature, rectangles, canvas):
    height, width = feature.shape[-2:]
    ch, cw = canvas
    y = (torch.arange(height, device=feature.device, dtype=torch.float32) + .5) * ch / height
    x = (torch.arange(width, device=feature.device, dtype=torch.float32) + .5) * cw / width
    y, x = torch.meshgrid(y, x, indexing='ij')
    return ((x[None] >= rectangles[:, 0, None, None]) & (x[None] < rectangles[:, 2, None, None]) &
            (y[None] >= rectangles[:, 1, None, None]) & (y[None] < rectangles[:, 3, None, None]))[:, None]


class LetterboxTransformer(RTDETRTransformer):
    """Official decoder with invalid-letterbox-anchor exclusion and query access.

    All attention/box updates remain the upstream implementations. If a very thin
    image has fewer than 300 valid encoder anchors, valid indices are repeated to
    retain 300 queries rather than silently selecting padded/invalid anchors.
    """
    def __init__(self):
        super().__init__(num_classes=4, hidden_dim=256, num_queries=300,
                         feat_channels=[256, 256, 256], feat_strides=[8, 16, 32],
                         num_levels=3, num_decoder_layers=3, num_denoising=100,
                         eval_spatial_size=None, aux_loss=True)
        self._captured_queries = None
        self.decoder.layers[-1].register_forward_hook(self._capture_queries)

    def _capture_queries(self, _module, _inputs, output):
        self._captured_queries = output

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        anchors, geometric_valid = self._generate_anchors(spatial_shapes, device=memory.device)
        allowed = geometric_valid.expand(memory.shape[0], -1, -1) & self._padding_valid[..., None]
        memory = memory * allowed.to(memory.dtype)
        output_memory = self.enc_output(memory)
        enc_class = self.enc_score_head(output_memory)
        enc_coord = self.enc_bbox_head(output_memory) + anchors
        selection_scores = enc_class.float().max(-1).values.masked_fill(~allowed[..., 0], -float('inf'))
        topk = []
        for batch in range(len(memory)):
            indices = allowed[batch, :, 0].nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                raise ValueError('No valid encoder anchors: valid image region is too small')
            count = min(self.num_queries, indices.numel())
            chosen = selection_scores[batch].topk(count).indices
            if count < self.num_queries:
                chosen = chosen.repeat(math.ceil(self.num_queries / count))[:self.num_queries]
            topk.append(chosen)
        topk = torch.stack(topk)
        reference = enc_coord.gather(1, topk[..., None].expand(-1, -1, 4))
        boxes = reference.sigmoid()
        logits = enc_class.gather(1, topk[..., None].expand(-1, -1, self.num_classes))
        target = output_memory.gather(1, topk[..., None].expand(-1, -1, self.hidden_dim)).detach()
        if denoising_class is not None:
            # Embedding lookup stays fp32 under autocast. Preserve the decoder's
            # activation dtype so MultiheadAttention does not canonicalize a bool
            # DN mask into fp32 while its projected Q/K/V are bf16 (torch 2.1).
            target = torch.cat((denoising_class.to(target.dtype), target), 1)
            reference = torch.cat((denoising_bbox_unact, reference), 1)
        return target, reference.detach(), boxes, logits

    def forward(self, feats, targets=None, valid_masks=None):
        memory, shapes, starts = self._get_encoder_input(feats)
        self._padding_valid = (torch.cat([mask.flatten(1) for mask in valid_masks], 1)
                               if valid_masks is not None else
                               torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device))
        memory = memory * self._padding_valid[..., None].to(memory.dtype)
        if self.training and self.num_denoising > 0:
            dn_class, dn_box, attention, dn_meta = get_contrastive_denoising_training_group(
                targets, self.num_classes, self.num_queries, self.denoising_class_embed,
                num_denoising=self.num_denoising, label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale)
        else:
            dn_class, dn_box, attention, dn_meta = None, None, None, None
        target, reference, enc_boxes, enc_logits = self._get_decoder_input(memory, shapes, dn_class, dn_box)
        boxes, logits = self.decoder(target, reference, memory, shapes, starts,
                                      self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
                                      attn_mask=attention, memory_mask=self._padding_valid)
        queries = self._captured_queries
        self._captured_queries = None
        if self.training and dn_meta is not None:
            dn_boxes, boxes = torch.split(boxes, dn_meta['dn_num_split'], dim=2)
            dn_logits, logits = torch.split(logits, dn_meta['dn_num_split'], dim=2)
            queries = queries[:, dn_meta['dn_num_split'][0]:]
        result = {'pred_logits': logits[-1], 'pred_boxes': boxes[-1], 'query_features': queries}
        if self.training and self.aux_loss:
            result['aux_outputs'] = self._set_aux_loss(logits[:-1], boxes[:-1])
            result['aux_outputs'].extend(self._set_aux_loss([enc_logits], [enc_boxes]))
            if dn_meta is not None:
                result['dn_aux_outputs'] = self._set_aux_loss(dn_logits, dn_boxes)
                result['dn_meta'] = dn_meta
        return result


def _roi_patches(feature, boxes, canvas, size=7, rectangles=None):
    """Normalized cxcywh boxes -> float32 ROI patches, with letterbox mask."""
    batch, count = boxes.shape[:2]
    if count == 0:
        return feature.new_empty((0, feature.shape[1], size, size)), feature.new_empty((0, 1, size, size))
    ch, cw = canvas
    xyxy = box_cxcywh_to_xyxy(boxes.float())
    # Canvas and feature dimensions are divisible by their stride in supplied presets.
    xyxy = xyxy * xyxy.new_tensor([feature.shape[-1], feature.shape[-2]] * 2)
    batch_idx = torch.arange(batch, device=feature.device)[:, None].expand(batch, count).reshape(-1, 1)
    rois = torch.cat((batch_idx.float(), xyxy.reshape(-1, 4)), 1)
    if rectangles is None:
        mask = torch.ones_like(feature[:, :1], dtype=torch.float32)
    else:
        mask = feature_valid_mask(feature, rectangles, canvas).float()
    # ROIAlign kernels on torch/torchvision 2.1 CPU and CUDA share float32 support.
    with torch.autocast(device_type=feature.device.type, enabled=False):
        patches = roi_align(feature.float() * mask, rois.float(), size, spatial_scale=1., aligned=True)
        masks = roi_align(mask, rois.float(), size, spatial_scale=1., aligned=True)
    return patches, masks


class RoleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.classifier = nn.Sequential(nn.Linear(64 * 2 + 6, 128), nn.GELU(), nn.Linear(128, 4))
        # Initially nearly neutral; unsupported classes are EXACTLY neutral via mask.
        nn.init.constant_(self.classifier[-1].bias, 2.0)

    def forward(self, feature, boxes, rectangles, canvas):
        batch, count = boxes.shape[:2]
        if count == 0:
            return feature.new_empty((batch, 0, 4))
        boxes = boxes.detach()
        inner, inner_mask = _roi_patches(feature, boxes, canvas, rectangles=rectangles)
        outer_boxes = torch.cat((boxes[..., :2], boxes[..., 2:] * 1.8), -1)
        outer, outer_mask = _roi_patches(feature, outer_boxes, canvas, rectangles=rectangles)
        axis = (torch.arange(7, device=feature.device).float() + .5) / 7 - .5
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        ring = ((xx.abs() > .5 / 1.8) | (yy.abs() > .5 / 1.8))[None, None]
        outer_mask = outer_mask * ring
        inner = self.encoder(inner)
        outer = self.encoder(outer * ring)
        inner = (inner * inner_mask).sum((-1, -2)) / inner_mask.sum((-1, -2)).clamp_min(1e-6)
        outer = (outer * outer_mask).sum((-1, -2)) / outer_mask.sum((-1, -2)).clamp_min(1e-6)
        aspect = torch.log((boxes[..., 2] + 1e-6) / (boxes[..., 3] + 1e-6)).clamp(-8, 8)
        area = (boxes[..., 2] * boxes[..., 3]).clamp_min(1e-8).log().clamp_min(-16)
        geometry = torch.cat((boxes, aspect[..., None], area[..., None]), -1).reshape(-1, 6)
        return self.classifier(torch.cat((inner, outer, geometry), -1)).reshape(batch, count, 4)


class DetailRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(64, 256, 1)
        self.roi_encoder = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.fuse = nn.Sequential(nn.Linear(256 + 64, 256), nn.GELU(), nn.LayerNorm(256))
        self.class_delta = nn.Linear(256, 4)
        self.box_delta = nn.Linear(256, 4)
        nn.init.zeros_(self.class_delta.weight)
        nn.init.zeros_(self.class_delta.bias)
        nn.init.zeros_(self.box_delta.weight)
        nn.init.zeros_(self.box_delta.bias)
        self.last_roi_count = 0

    def forward(self, c2, query, boxes, logits, selected, rectangles, canvas):
        selected_boxes = boxes.gather(1, selected[..., None].expand(-1, -1, 4))
        selected_queries = query.gather(1, selected[..., None].expand(-1, -1, query.shape[-1]))
        self.last_roi_count = int(selected.numel())
        fine_feature = self.project(c2)
        patches, _mask = _roi_patches(fine_feature, selected_boxes.detach(), canvas, rectangles=rectangles)
        local = self.roi_encoder(patches).flatten(1).reshape(query.shape[0], selected.shape[1], -1)
        fine_query = selected_queries + self.fuse(torch.cat((selected_queries, local), -1))
        refined_logits = logits.gather(1, selected[..., None].expand(-1, -1, 4)) + self.class_delta(fine_query)
        bounded_delta = self.box_delta(fine_query).tanh() * .5
        refined_boxes = (torch.logit(selected_boxes.clamp(1e-5, 1 - 1e-5)) + bounded_delta).sigmoid()
        all_logits = logits.scatter(1, selected[..., None].expand(-1, -1, 4), refined_logits)
        all_boxes = boxes.scatter(1, selected[..., None].expand(-1, -1, 4), refined_boxes)
        return {'pred_logits': all_logits, 'pred_boxes': all_boxes}


class ContextHead(nn.Module):
    """Bounded four-class residual from a candidate and its valid outer ring.

    The prediction path uses image features and proposed geometry only. Existing
    positive labels are consulted solely by ``auxiliary`` during training.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GELU(),
                                     nn.Conv2d(64, 64, 3, padding=1), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(64 * 2 + 6, 128), nn.GELU(), nn.LayerNorm(128))
        self.embedding = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64))
        self.class_delta = nn.Linear(128, 4)
        self.gate = nn.Linear(128, 4)
        nn.init.zeros_(self.class_delta.weight)
        nn.init.zeros_(self.class_delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.last_roi_count = 0

    def _pooled(self, feature, boxes, rectangles, canvas):
        batch, count = boxes.shape[:2]
        if count == 0:
            return feature.new_empty((0, 134))
        boxes = boxes.detach()
        inner, inner_mask = _roi_patches(feature, boxes, canvas, rectangles=rectangles)
        outer_boxes = torch.cat((boxes[..., :2], boxes[..., 2:] * 1.8), -1)
        outer, outer_mask = _roi_patches(feature, outer_boxes, canvas, rectangles=rectangles)
        axis = (torch.arange(7, device=feature.device, dtype=torch.float32) + .5) / 7 - .5
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        ring = ((xx.abs() > .5 / 1.8) | (yy.abs() > .5 / 1.8))[None, None]
        outer_mask = outer_mask * ring
        inner = self.encoder(inner)
        outer = self.encoder(outer * ring)
        inner = (inner * inner_mask).sum((-1, -2)) / inner_mask.sum((-1, -2)).clamp_min(1e-6)
        outer = (outer * outer_mask).sum((-1, -2)) / outer_mask.sum((-1, -2)).clamp_min(1e-6)
        aspect = torch.log((boxes[..., 2] + 1e-6) / (boxes[..., 3] + 1e-6)).clamp(-8, 8)
        area = (boxes[..., 2] * boxes[..., 3]).clamp_min(1e-8).log().clamp_min(-16)
        geometry = torch.cat((boxes, aspect[..., None], area[..., None]), -1).reshape(-1, 6)
        return torch.cat((inner, outer, geometry), -1)

    def forward(self, feature, boxes, rectangles, canvas):
        self.last_roi_count = int(boxes.shape[0] * boxes.shape[1])
        pooled = self._pooled(feature, boxes, rectangles, canvas)
        hidden = self.fusion(pooled)
        residual = 2. * self.gate(hidden).sigmoid() * self.class_delta(hidden).tanh()
        residual = residual.reshape(*boxes.shape[:2], 4)
        return residual, {'candidate_context_rois': self.last_roi_count,
                          'residual_absolute_mean': float(residual.detach().abs().mean()) if residual.numel() else 0.,
                          'residual_bound': 2.,
                          'placement': CONTEXT_POLICY['placement']}

    def auxiliary(self, feature, targets, rectangles, canvas):
        selected, diagnostics = select_context_targets(targets)
        views, labels, weights, boxes, image_ids = [[], []], [], [], [], []
        for batch, index in enumerate(selected):
            if not len(index):
                continue
            target = targets[batch]
            chosen = target['boxes'][index]
            pooled = self._pooled(feature[batch:batch + 1], chosen[None], rectangles[batch:batch + 1], canvas)
            # Independent mild feature dropout preserves box semantics, unlike
            # inventing text/background labels from unverified image regions.
            for view in range(2):
                augmented = F.dropout(pooled, p=.1, training=True)
                views[view].append(self.embedding(self.fusion(augmented)))
            labels.append(target['labels'][index].long())
            weights.append(target.get('weights', chosen.new_ones(len(target['boxes'])))[index])
            boxes.append(chosen.detach())
            image_ids.append(torch.full((len(index),), batch, device=feature.device, dtype=torch.long))
        if labels:
            return {'embeddings': torch.stack([torch.cat(view) for view in views]),
                    'labels': torch.cat(labels), 'weights': torch.cat(weights),
                    'boxes': torch.cat(boxes), 'image_ids': torch.cat(image_ids),
                    'selection': diagnostics}
        # Keep an actual differentiable zero for empty/full-ignore training.
        empty = feature.sum() * 0 + self.embedding[-1].weight.sum() * 0
        return {'embeddings': empty.expand(2, 0, 64),
                'labels': torch.empty(0, device=feature.device, dtype=torch.long),
                'weights': feature.new_empty(0), 'boxes': feature.new_empty((0, 4)),
                'image_ids': torch.empty(0, device=feature.device, dtype=torch.long),
                'selection': diagnostics}


class DictionaryDetector(nn.Module):
    def __init__(self, experiment='baseline', role_enabled=None):
        super().__init__()
        if experiment not in ('baseline', 'final', 'detail', 'context', 'combined'):
            raise ValueError('experiment must be baseline, final, detail, context or combined')
        self.experiment = experiment
        self.backbone = PResNet(depth=18, variant='d', return_idx=[0, 1, 2, 3],
                               freeze_at=-1, freeze_norm=False, pretrained=False)
        self.encoder = HybridEncoder(in_channels=[128, 256, 512], feat_strides=[8, 16, 32],
                                     hidden_dim=256, expansion=.5, depth_mult=1.,
                                     enc_act='gelu', act='silu', eval_spatial_size=None)
        self.decoder = LetterboxTransformer()
        self.enable_extensions = experiment in ('final', 'detail', 'combined')
        self.enable_context = experiment in ('context', 'combined')
        enabled = [False] * 4 if role_enabled is None else role_enabled
        if len(enabled) != 4:
            raise ValueError('role_enabled must contain four booleans')
        self.register_buffer('role_enabled', torch.tensor(enabled, dtype=torch.bool))
        if self.enable_extensions:
            self.refiner = DetailRefiner()
            self.gain_head = nn.Sequential(nn.Linear(256 + 4 + 4, 128), nn.GELU(), nn.Linear(128, 1))
        if experiment == 'final':
            self.role_head = RoleHead()
        if self.enable_context:
            self.context_head = ContextHead()

    def forward(self, images, targets=None, epoch=1, profile='pilot'):
        canvas = images.shape[-2:]
        rectangles = valid_rectangles(images, targets)
        features = self.backbone(images)
        encoded = self.encoder(features[1:])
        masks = [feature_valid_mask(feature, rectangles, canvas)[:, 0] for feature in encoded]
        result = self.decoder(encoded, targets if self.training else None, valid_masks=masks)
        if self.enable_context:
            # Context first: both coarse and fine have exactly the same context
            # prior. The gain oracle therefore measures the actual subsequent
            # detail residual under one fixed coarse Hungarian assignment.
            residual, diagnostics = self.context_head(encoded[0], result['pred_boxes'], rectangles, canvas)
            result['context_base_outputs'] = {'pred_logits': result['pred_logits'], 'pred_boxes': result['pred_boxes']}
            result['pred_logits'] = result['pred_logits'] + residual
            result['context_diagnostics'] = diagnostics
            result['variant'] = self.experiment
            if self.training and targets is not None:
                result['context_aux'] = self.context_head.auxiliary(encoded[0], targets, rectangles, canvas)
        if not self.enable_extensions:
            return result
        coarse = {'pred_logits': result['pred_logits'], 'pred_boxes': result['pred_boxes']}
        query = result['query_features']
        gain_input = torch.cat((query, coarse['pred_logits'].sigmoid(), coarse['pred_boxes']), -1)
        gains = self.gain_head(gain_input).squeeze(-1)
        count = query.shape[1]
        selected = gains.topk(min(64, count), dim=1).indices
        if self.training:
            # All normal queries supply paired coarse/fine training targets; no DN queries.
            all_queries = torch.arange(count, device=images.device)[None].expand(len(images), -1)
            fine = self.refiner(features[0], query, coarse['pred_boxes'], coarse['pred_logits'],
                                all_queries, rectangles, canvas)
            final = {key: coarse[key].scatter(1, selected[..., None].expand(-1, -1, 4),
                                              value.gather(1, selected[..., None].expand(-1, -1, 4)))
                     for key, value in fine.items()}
            if stage_for(epoch, profile) != 'joint':
                final = coarse
        else:
            fine = self.refiner(features[0], query, coarse['pred_boxes'], coarse['pred_logits'],
                                selected, rectangles, canvas)
            final = fine
        result.update(final)
        result.update(coarse_outputs=coarse, fine_outputs=fine, gain_pred=gains, selected_indices=selected,
                      stage=stage_for(epoch, profile), role_enabled=self.role_enabled)
        if self.experiment != 'final':
            result['variant'] = self.experiment
        if self.experiment == 'final' and bool(self.role_enabled.any()):
            role_logits = self.role_head(encoded[0], result['pred_boxes'], rectangles, canvas)
            factor = torch.where(self.role_enabled[None, None], role_logits.sigmoid(), torch.ones_like(role_logits))
            adjusted = torch.logit((result['pred_logits'].sigmoid() * factor).clamp(1e-7, 1 - 1e-7))
            # The unsupported classes retain bitwise-identical logits, not a logit(sigmoid()) round trip.
            result['role_adjusted_logits'] = torch.where(self.role_enabled[None, None], adjusted, result['pred_logits'])
            result['role_logits'] = role_logits
            if self.training and targets is not None:
                pairs = []
                for batch, target in enumerate(targets):
                    positive = target.get('role_positive_boxes')
                    negative = target.get('role_negative_boxes')
                    labels = target.get('role_pair_labels')
                    if positive is None or len(positive) == 0:
                        continue
                    if negative is None or labels is None or len(positive) != len(negative):
                        raise ValueError('Role positive/negative boxes and labels must have equal length')
                    boxes = torch.cat((positive, negative), 0)[None]
                    logits = self.role_head(encoded[0][batch:batch + 1], boxes,
                                            rectangles[batch:batch + 1], canvas)[0]
                    n = len(positive)
                    labels = labels.long()
                    pair_weights = target.get('role_pair_weights', torch.ones(n, device=images.device))
                    active = self.role_enabled[labels]
                    pairs.append({'positive': logits[:n].gather(1, labels[:, None]).squeeze(1)[active],
                                  'negative': logits[n:].gather(1, labels[:, None]).squeeze(1)[active],
                                  'weights': pair_weights[active]})
                result['role_pair_logits'] = pairs
        return result


def _state_dictionary(blob):
    if isinstance(blob, dict) and 'ema' in blob and blob['ema'] is not None:
        value = blob['ema']
        if isinstance(value, dict):
            return value.get('module', value)
    if isinstance(blob, dict):
        return blob.get('model', blob.get('state_dict', blob))
    raise ValueError('Unrecognized official checkpoint format')


def load_coco_pretrained(model, path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Official COCO checkpoint missing: {path}')
    upstream = _state_dictionary(torch.load(path, map_location='cpu'))
    current = model.state_dict()
    skipped, loaded, unexpected = [], {}, []
    reset_names = ('decoder.enc_score_head.', 'decoder.dec_score_head.', 'decoder.denoising_class_embed.')
    extension_names = ('refiner.', 'role_head.', 'gain_head.', 'context_head.', 'role_enabled')
    for key, value in upstream.items():
        key = key.removeprefix('module.')
        if key.startswith(reset_names):
            skipped.append(key)
        elif key in current and value.shape == current[key].shape:
            loaded[key] = value
        else:
            unexpected.append(key)
    initialized_counters = [key for key in current if key not in loaded and key.endswith('.num_batches_tracked')]
    missing_base = [key for key in current if key not in loaded and
                    not key.startswith(reset_names + extension_names) and key not in initialized_counters]
    if missing_base or unexpected:
        raise ValueError(f'Checkpoint is not pinned official R18-vd: missing={missing_base[:12]}, unexpected={unexpected[:12]}')
    model.load_state_dict(loaded, strict=False)
    return {'checkpoint': str(path), 'loaded_tensors': len(loaded), 'reset_four_class_tensors': skipped,
            'missing_base_tensors': missing_base, 'unexpected_tensors': unexpected,
            'initialized_batch_counters': initialized_counters,
            'official_commit': 'df2a96c8c3a098567a1149bf0248da6efe6d495d'}


def build_model(experiment, pretrained_path=None, role_enabled=None, load_pretrained=True):
    model = DictionaryDetector(experiment=experiment, role_enabled=role_enabled)
    if load_pretrained:
        if pretrained_path is None:
            pretrained_path = Path(__file__).resolve().parents[1] / 'weights' / 'rtdetr_r18vd_coco.pth'
        report = load_coco_pretrained(model, pretrained_path)
    else:
        report = {'loaded_tensors': 0, 'pretrained': False, 'test_only_random_initialization': True}
    report.update(experiment=experiment, parameters=sum(p.numel() for p in model.parameters()),
                  normal_queries=300, decoder_layers=3, denoising_queries=100,
                  backbone='Official PResNet18-vd', role_enabled=model.role_enabled.tolist())
    if experiment in ('detail', 'context', 'combined'):
        report.update(variant=experiment, detail_enabled=model.enable_extensions,
                      context_enabled=model.enable_context, uses_ocr=False,
                      context_policy=CONTEXT_POLICY if model.enable_context else None)
    return model, report

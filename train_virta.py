"""ViRTA training with VPRM, ATM, Fine-Grained Adaptive Fusion Module and full English articles."""
import argparse
import math
from lr_warmup_cosine import warmup_cosine_factor
import hashlib
import json
import random
import numpy as np
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from classification_metrics import classification_metrics, metric_table
from ViRTA import ViRTA
from virta_dataset import build_dataset, collate_fn, has_explicit_validation, configure_data_root
from prepare_tenfold import FOLDS, protocol


def _normalized_text(value):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value)).lower()


def _image_digest(value, csv_path):
    image_path = Path(str(value).split("|")[0])
    if not image_path.is_absolute():
        csv_path = Path(csv_path)
        if csv_path.stem in {"gossip_train", "gossip_val", "gossip_test"} and len(image_path.parts) == 1:
            split = csv_path.stem.removeprefix("gossip_")
            image_path = csv_path.parent / "Images" / f"gossip_{split}" / image_path.name
        else:
            image_path = csv_path.parent / image_path
    if not image_path.is_file():
        return None
    digest = hashlib.sha256()
    with image_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_aware_split(frame, csv_path, validation_ratio, seed):
    """Split duplicate-connected samples as indivisible, label-stratified groups."""
    frame = frame.reset_index(drop=True)
    size = len(frame)
    parent = list(range(size))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owners = {}
    for index, row in frame.iterrows():
        text = _normalized_text(row.get("text", row.get("content", "")))
        digest = _image_digest(row.get("image", row.get("images", "")), csv_path)
        for key in (("text", text), ("image", digest)):
            if not key[1]:
                continue
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    grouped = defaultdict(list)
    for index in range(size):
        grouped[find(index)].append(index)
    groups = list(grouped.values())
    group_labels = []
    for group in groups:
        counts = frame.iloc[group]["label"].astype(int).value_counts()
        group_labels.append(int(counts.idxmax()))
    train_groups, validation_groups = train_test_split(
        list(range(len(groups))), test_size=validation_ratio, random_state=seed, stratify=group_labels
    )
    train_indices = [index for group in train_groups for index in groups[group]]
    validation_indices = [index for group in validation_groups for index in groups[group]]
    train_roots = {find(index) for index in train_indices}
    validation_roots = {find(index) for index in validation_indices}
    if train_roots & validation_roots:
        raise RuntimeError("duplicate groups leaked across training and validation")
    return train_indices, validation_indices


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def model_forward(model, batch, image_key="image"):
    return model(batch.get("input_ids"), batch.get("attention_mask"), batch.get("token_type_ids"), batch[image_key])


def focal_loss(logits, labels, alpha, gamma=2.0):
    """Alpha-balanced softmax focal loss; p_t comes from unweighted log probabilities."""
    log_pt = torch.nn.functional.log_softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    return (-alpha[labels] * (1 - log_pt.exp()).pow(gamma) * log_pt).mean()


def binary_classification_loss(logits, labels):
    """BCE on binary log odds, preserving the model's two-class predictions."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Binary classification requires two class logits per sample")
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1] - logits[:, 0], labels.to(dtype=logits.dtype))


def fusion_classification_loss(logits, aux, labels, class_weights=None,
                               lambda_branch=.2, lambda_fusion_balance=.02, label_smoothing=.05,
                               criterion=None):
    """Final and auxiliary classification losses plus the existing fusion regularizer."""
    if criterion is None:
        criterion = partial(torch.nn.functional.cross_entropy, weight=class_weights,
                            label_smoothing=label_smoothing)
    classification = criterion(logits, labels)
    branch_losses = [criterion(aux["branch_logits"][:, index], labels)
                     for index in range(3)]
    auxiliary = torch.stack(branch_losses).mean()
    mean_weights = aux["weights"].mean(0)
    fusion_balance = ((mean_weights - 1 / 3) ** 2).sum()
    total = classification + lambda_branch * auxiliary + lambda_fusion_balance * fusion_balance
    return total, {"classification": classification.detach(), "branch_auxiliary": auxiliary.detach(),
                   "fusion_balance": fusion_balance.detach()}


def evaluate(model, loader, device, class_weights=None, collect_risk=False, criterion=None,
             predictions_path=None, features_path=None):
    model.eval(); predictions, labels, weights = [], [], []
    loss_sum = 0.0
    risk_features, error_targets, news_ids, classifier_features = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device); logits, aux = model_forward(model, batch)
            loss = (criterion(logits, batch["label"]) if criterion is not None else
                    torch.nn.functional.cross_entropy(logits, batch["label"], weight=class_weights))
            loss_sum += loss.item() * len(batch["label"])
            prediction = logits.argmax(1)
            predictions.extend(prediction.cpu().tolist()); labels.extend(batch["label"].cpu().tolist())
            if predictions_path is not None or features_path is not None:
                news_ids.extend(batch['news_id'])
            if features_path is not None:
                classifier_features.append(aux['fused_features'].detach().cpu().numpy())
            weights.append(aux["weights"].cpu())
            if collect_risk:
                risk_features.append(aux["risk_features"].cpu())
                error_targets.append((prediction != batch["label"]).float().cpu())
    result = {"loss": loss_sum / max(len(labels), 1), "accuracy": accuracy_score(labels, predictions),
              "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
              "mean_weights": torch.cat(weights).mean(0).tolist()}
    result.update(classification_metrics(labels, predictions))
    if collect_risk:
        result["risk_features"] = torch.cat(risk_features).numpy()
        result["error_targets"] = torch.cat(error_targets).numpy()
    if predictions_path is not None:
        Path(predictions_path).write_text(json.dumps({'labels': labels, 'predictions': predictions,
                                                     'news_ids': news_ids}, ensure_ascii=False))
    if features_path is not None:
        np.savez_compressed(features_path, features=np.concatenate(classifier_features),
                            labels=np.asarray(labels, dtype=np.int64), news_ids=np.asarray(news_ids))
    return result


def _f1_threshold(targets, probabilities):
    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities)))
    scores = [f1_score(targets, probabilities >= threshold, zero_division=0) for threshold in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def calibrate_risk(model, loader, device, class_weights=None, folds=FOLDS, seed=0, criterion=None):
    """Calibrate the risk head and report an out-of-sample error-detection F1.

    The coefficients and threshold that ship in the checkpoint are still fit on every
    validation row (that is what runs at inference). ``error_f1_in_sample`` scores those
    same rows and is therefore optimistic; ``error_f1_cross_fitted`` refits the calibrator
    on K-1 folds and predicts the held-out fold, so it never scores the rows it was fit on
    and is the number that should be reported.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    protocol(folds)
    metrics = evaluate(model, loader, device, class_weights, collect_risk=True, criterion=criterion)
    features, targets = metrics.pop("risk_features"), metrics.pop("error_targets")
    summary = {"threshold": 1.0, "error_f1_in_sample": 0.0, "error_f1_cross_fitted": None,
               "folds": 0, "error_count": int(targets.sum()), "validation": metrics}
    if len(np.unique(targets)) < 2:
        return summary

    def fit(x, y):
        return LogisticRegression(max_iter=1000, random_state=seed).fit(x, y)

    calibrator = fit(features, targets)
    probabilities = calibrator.predict_proba(features)[:, 1]
    threshold, in_sample_f1 = _f1_threshold(targets, probabilities)
    model.risk_coefficients.copy_(torch.tensor(calibrator.coef_[0], device=device, dtype=model.risk_coefficients.dtype))
    model.risk_intercept.copy_(torch.tensor(calibrator.intercept_[0], device=device, dtype=model.risk_intercept.dtype))
    model.risk_threshold.copy_(torch.tensor(threshold, device=device, dtype=model.risk_threshold.dtype))
    summary.update(threshold=threshold, error_f1_in_sample=in_sample_f1)
    if len(targets) < FOLDS:
        summary['cross_fit_skipped'] = 'Fewer than ten validation samples; no reduced-fold fallback'
        return summary
    # Cross-fitting: each held-out fold is flagged by a calibrator and threshold it never saw.
    flagged = np.ones(len(targets), dtype=bool)
    for fit_index, hold_index in KFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(features):
        if len(np.unique(targets[fit_index])) < 2:
            continue
        fold_calibrator = fit(features[fit_index], targets[fit_index])
        fold_probabilities = fold_calibrator.predict_proba(features[fit_index])[:, 1]
        fold_threshold, _ = _f1_threshold(targets[fit_index], fold_probabilities)
        flagged[hold_index] = fold_calibrator.predict_proba(features[hold_index])[:, 1] >= fold_threshold
    summary.update(folds=FOLDS,
                   error_f1_cross_fitted=float(f1_score(targets, flagged, zero_division=0)))
    return summary


def train_extended_text_positions(encoders):
    """English-CLIP's mask2 activates only positions 20:248; keep all other weights frozen."""
    if encoders.language != "en":
        raise ValueError("Extended text position training requires the English-CLIP encoder")
    mask = encoders.clip.mask2
    if mask.shape != (248, 1) or torch.count_nonzero(mask[:20]) or not torch.all(mask[20:] == 1):
        raise ValueError("Expected English-CLIP's fixed first-20 / trainable last-228 position mask")
    encoders.requires_grad_(False)
    positions = encoders.clip.positional_embedding_res
    positions.requires_grad_(True)
    return positions


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def main(args):
    from cross_validation import run, verify_worker
    if getattr(args, '_cv_worker', False):
        verify_worker(args)
        return train_one_fold(args)
    return run(args)


def train_one_fold(args):
    configure_data_root(args.dataset, args.data_root)
    if args.dataset == "gossipcop":
        if args.train_text_positions:
            print("GossipCop text positions 0:248 are frozen; ignoring --train-text-positions", flush=True)
        args.train_text_positions = False
    args.classification_loss = args.classification_loss or ("bce" if args.dataset == "gossipcop" else "ce")
    if args.classification_loss == "bce":
        args.label_smoothing = 0.0
        args.lambda_branch = args.lambda_fusion_balance = 0.0
    args.no_validation = getattr(args, 'no_validation', False)
    if args.no_validation and args.dataset not in {'gossipcop', 'twitter'}:
        raise ValueError('--no-validation is enabled for GossipCop and Twitter only')
    if args.classification_loss == "focal":
        if args.dataset not in {"gossipcop", "twitter"}:
            raise ValueError("Focal Loss is enabled for GossipCop and Twitter")
        if not math.isfinite(args.focal_gamma) or args.focal_gamma < 0:
            raise ValueError("focal-gamma must be finite and nonnegative")
        if args.label_smoothing != 0:
            raise ValueError("Canonical Focal Loss requires --label-smoothing 0")
    if args.train_text_positions and (args.dataset == "weibo" or args.train_backbones):
        raise ValueError("Train extended English text positions with all other encoders frozen")
    if args.vprm_checkpoint and args.train_backbones:
        raise ValueError("Keep the pretrained VPRM visual encoder frozen during task adaptation")
    if args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("batch-size 和 accumulation-steps 必须大于零")
    if args.epochs < 1:
        raise ValueError("epochs 必须大于零")
    if not args.vprm_checkpoint or not Path(args.vprm_checkpoint).is_file():
        raise FileNotFoundError("Full-model training requires a pretrained --vprm-checkpoint")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.output is None:
        args.output = f"virta_runs/{args.dataset}/best.pth"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.no_validation:
        args.patience = None
    elif args.patience is None:
        args.patience = 2 if args.dataset in {"gossipcop", "twitter"} else 3
    if not args.no_validation and args.patience < 1:
        raise ValueError("patience must be positive")
    Path(str(args.output) + ".config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_train_set = build_dataset(args.dataset, "train", True)
    test_set = None if args.no_validation else build_dataset(args.dataset, "test", False)
    if args.no_validation:
        train_set, validation_set = source_train_set, None
        print(f'fixed_epochs={args.epochs} validation=disabled early_stopping=disabled '
              'checkpoint=last_epoch risk_calibration=disabled test=after_training', flush=True)
    elif has_explicit_validation(args.dataset):
        source_validation_set = build_dataset(args.dataset, "val", False)
        train_set, validation_set = source_train_set, source_validation_set
        print(f"使用固定划分，验证集为独立文件: {source_validation_set.csv_path}", flush=True)
    else:
        source_validation_set = build_dataset(args.dataset, "train", False)
        labels = source_train_set.frame["label"].astype(int).tolist()
        train_indices, validation_indices = duplicate_aware_split(
            source_train_set.frame, source_train_set.csv_path, args.validation_ratio, args.seed)
        train_set, validation_set = Subset(source_train_set, train_indices), Subset(source_validation_set, validation_indices)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=collate_fn, drop_last=False,
                              worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed))
    train_labels = (source_train_set.frame["label"].astype(int).tolist()
                    if args.no_validation or has_explicit_validation(args.dataset)
                    else [labels[index] for index in train_indices])
    print(f"natural_sampling counts={[train_labels.count(0), train_labels.count(1)]} "
          f"batches={len(train_loader)} samples_per_epoch={len(train_set)} "
          "replacement=False drop_last=False", flush=True)
    validation_loader = (DataLoader(validation_set, args.batch_size, shuffle=False, num_workers=args.workers,
                                    collate_fn=collate_fn) if validation_set is not None else None)
    test_loader = (DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers,
                              collate_fn=collate_fn) if test_set is not None else None)
    if not len(train_loader):
        raise ValueError("训练样本少于 batch-size，请减小 batch-size")
    criterion = None
    loss_config = {"type": args.classification_loss, "label_smoothing": args.label_smoothing,
                   "class_weights": None}
    if args.classification_loss == "bce":
        criterion = binary_classification_loss
        loss_config.update(binary_logit="real_logit - fake_logit", labels={"0": "fake", "1": "real"},
                           lambda_branch=0.0, lambda_fusion_balance=0.0,
                           applies_to="fused prediction only; same BCE for training, validation and test")
    elif args.classification_loss == "focal":
        counts = torch.bincount(torch.tensor(train_labels), minlength=2).to(device=device, dtype=torch.float32)
        if counts.numel() != 2 or (counts <= 0).any():
            raise ValueError("Focal alpha requires both binary classes in the training split")
        alpha = counts.reciprocal()
        alpha = alpha / alpha.sum()
        criterion = partial(focal_loss, alpha=alpha, gamma=args.focal_gamma)
        loss_config.update(gamma=args.focal_gamma, alpha=alpha.tolist(),
                           alpha_source="normalized inverse training class counts only",
                           train_class_counts=counts.to(torch.int64).tolist(),
                           labels={"0": "fake", "1": "real"},
                           applies_to="final and all three auxiliary classifiers; validation/test classification loss")
    Path(str(args.output) + ".loss.json").write_text(json.dumps(loss_config, indent=2))
    print(f"classification_loss={loss_config}", flush=True)
    print(f"dataset={args.dataset} train={len(train_set)} val={len(validation_set) if validation_set is not None else 0} "
          f"test={len(test_set) if test_set is not None else 'deferred'} encoders=CLIP output={args.output}", flush=True)
    model = ViRTA(dataset=args.dataset, dropout=args.dropout, vprm_checkpoint=args.vprm_checkpoint,
                 atm_keep_ratio=args.atm_keep_ratio).to(device)
    print('visual_restoration=VPRM', flush=True)
    image_tokens = 1 + int(196 * args.atm_keep_ratio)
    print(f'ATM keep_ratio={args.atm_keep_ratio} '
          f'image_tokens={image_tokens} text_tokens_per_chunk={model.text_slots+1}', flush=True)
    if not args.train_backbones:
        model.encoders.requires_grad_(False)
    if args.freeze_text:
        for name,p in model.encoders.named_parameters():
            if not name.startswith('clip.visual.'):
                p.requires_grad_(False)
    if args.train_text_positions:
        train_extended_text_positions(model.encoders)
        print(f"trainable_text_positions=20:248 effective_parameters={228 * model.encoders.text_width} "
              f"positional_weight_decay=0 text_lr={args.text_lr} other_encoder_weights_frozen=True", flush=True)
    if args.dataset == "gossipcop":
        # Rows 0:20 were already fixed; freeze the residual used for rows 20:248 too.
        model.encoders.clip.positional_embedding_res.requires_grad_(False)
        print("frozen_text_positions=0:248 trainable_text_position_parameters=0", flush=True)
    visual_ids = {id(p) for p in model.encoders.clip.visual.parameters()}
    text_parameters = [p for p in model.encoders.parameters() if p.requires_grad and id(p) not in visual_ids]
    vision_parameters = [p for p in model.encoders.clip.visual.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in model.encoders.parameters()}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = []
    if text_parameters:
        text_group = {"params": text_parameters, "lr": args.text_lr}
        if args.train_text_positions:
            # mask2 gives the first 20 rows zero gradients; no decay keeps them unchanged.
            text_group["weight_decay"] = 0.0
        groups.append(text_group)
    if vision_parameters: groups.append({"params": vision_parameters, "lr": args.vision_lr})
    groups.append({"params": other, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_updates = args.epochs * updates_per_epoch
    warmup_updates = args.lr_warmup_epochs * updates_per_epoch
    peak_lrs = [group["lr"] for group in optimizer.param_groups]
    warmup_cosine_factor(0, total_updates, warmup_updates, args.lr_start_ratio, args.lr_min_ratio)
    global_update = 0
    schedule_config = dict(updates_per_epoch=updates_per_epoch, total_updates=total_updates,
                           warmup_updates=warmup_updates, peak_lrs=peak_lrs,
                           start_ratio=args.lr_start_ratio, min_ratio=args.lr_min_ratio)
    Path(str(args.output) + ".lr.json").write_text(json.dumps(schedule_config, indent=2))
    print(f"lr_schedule=warmup_cosine {schedule_config}", flush=True)
    class_weights = None
    # Sampling remains natural; Focal alpha, when selected, belongs only to the loss.
    best, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    for epoch in range(args.epochs):
        model.train()
        if not args.train_backbones:
            model.encoders.clip.visual.eval()
        if args.freeze_text or not args.train_backbones:
            model.encoders.eval()
        optimizer.zero_grad(set_to_none=True); running = {"total": 0.0}; steps = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            batch = move(batch, device); logits, aux = model_forward(model, batch)
            if args.classification_loss == "bce":
                loss = criterion(logits, batch["label"])
                parts = {"classification": loss.detach()}
            else:
                loss, parts = fusion_classification_loss(
                    logits, aux, batch["label"], class_weights,
                    args.lambda_branch, args.lambda_fusion_balance, args.label_smoothing, criterion=criterion)
            group_start = (steps // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, len(train_loader) - group_start)
            (loss / group_size).backward()
            if (steps + 1) % args.accumulation_steps == 0 or steps + 1 == len(train_loader):
                factor = warmup_cosine_factor(global_update, total_updates, warmup_updates,
                                               args.lr_start_ratio, args.lr_min_ratio)
                last_lrs = [peak * factor for peak in peak_lrs]
                for group, lr in zip(optimizer.param_groups, last_lrs):
                    group["lr"] = lr
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                global_update += 1
            running["total"] += loss.item(); steps += 1
            for name, value in parts.items():
                running[name] = running.get(name, 0.0) + value.item()
        print(f"optimizer_updates={global_update}/{total_updates} lr_used={last_lrs}", flush=True)
        if args.no_validation:
            print(f"epoch={epoch + 1} train_losses="
                  f"{{{', '.join(f'{k}: {v/max(steps,1):.4f}' for k,v in running.items())}}}", flush=True)
            continue
        metrics = evaluate(model, validation_loader, device, class_weights, criterion=criterion)
        print(f"epoch={epoch + 1} train_losses={{{', '.join(f'{k}: {v/max(steps,1):.4f}' for k,v in running.items())}}} validation={metrics}")
        significant_f1 = metrics["macro_f1"] > best + args.min_delta
        tied_but_better_loss = abs(metrics["macro_f1"] - best) <= args.min_delta and metrics["loss"] < best_loss - args.loss_min_delta
        if significant_f1 or tied_but_better_loss:
            best = max(best, metrics["macro_f1"])
            best_loss, best_epoch, stale = metrics["loss"], epoch + 1, 0
            torch.save(model.state_dict(), args.output)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stopping epoch={epoch + 1} best_epoch={best_epoch} best_macro_f1={best:.6f} "
                      f"best_loss={best_loss:.6f} patience={args.patience} min_delta={args.min_delta}")
                break
    if args.no_validation:
        torch.save(model.state_dict(), args.output)
        Path(str(args.output) + '.epoch.json').write_text(json.dumps(
            {'epoch': args.epochs, 'selection': 'fixed_last_epoch', 'validation': False,
             'early_stopping': False, 'risk_calibration': False}, indent=2))
        print(f'fixed_training_complete epochs={args.epochs} checkpoint={args.output}', flush=True)
        test_set = build_dataset(args.dataset, 'test', False)
        test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers,
                                 collate_fn=collate_fn)
    else:
        model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
        calibration = None
        if args.classification_loss != "bce":
            calibration = calibrate_risk(model, validation_loader, device, class_weights,
                                         seed=args.seed, criterion=criterion)
            torch.save(model.state_dict(), args.output)
        Path(str(args.output) + '.epoch.json').write_text(json.dumps(
            {'epoch': best_epoch, 'trained_epochs': epoch + 1, 'selection': 'validation_macro_f1_then_loss',
             'validation': True, 'early_stopping': epoch + 1 < args.epochs,
             'patience': args.patience, 'best_macro_f1': best, 'best_loss': best_loss,
             'risk_calibration': calibration is not None}, indent=2))
        print(f"risk_calibration={calibration}")
    features_path = (str(args.output) + '.features.npz'
                     if getattr(args, 'tsne_first_fold', False) and args.cv_fold == 0 else None)
    test_metrics = evaluate(model, test_loader, device, class_weights, criterion=criterion,
                            predictions_path=str(args.output) + '.predictions.json', features_path=features_path)
    print(f"test={test_metrics}")
    print(metric_table(test_metrics))
    Path(str(args.output) + ".test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    if features_path is not None:
        from plot_classifier_tsne import plot_features
        plot_features(features_path, Path(args.output).parent / 'tsne', seed=args.seed,
                      metadata={'dataset': args.dataset, 'fold': 1,
                                'atm_keep_ratio': args.atm_keep_ratio, 'checkpoint': str(args.output),
                                'selection': json.loads(Path(str(args.output) + '.epoch.json').read_text())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", choices=("weibo", "gossipcop", "twitter"), default="weibo")
    parser.add_argument('--cv-count', type=int, choices=(FOLDS,), default=FOLDS,
                        help='Only standard ten-fold 80/10/10 train/validation/test is supported')
    parser.add_argument('--tsne-first-fold', action='store_true', help='Plot the first fold test features entering the final classifier')
    parser.add_argument('--discard-fold-weights', action='store_true', help='Remove each completed fold checkpoint after saving metrics and features')
    parser.add_argument('--cv-splits', help='Shared standard ten-fold manifest with split seed 666')
    parser.add_argument('--cv-resume', action='store_true', help='Keep completed folds; restart incomplete folds from scratch')
    parser.add_argument('--cv-prepare-only', action='store_true', help='Prepare the splits and immutable run plan without training')
    parser.add_argument('--cv-fold', type=int, choices=range(10), help=argparse.SUPPRESS)
    parser.add_argument('--_cv-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--label-smoothing", type=float, default=.05)
    parser.add_argument("--classification-loss", choices=("ce", "focal", "bce"), default=None,
                        help="GossipCop defaults to BCE only; other datasets default to the existing CE objective")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=.02)
    parser.add_argument("--lambda-branch", type=float, default=.2)
    parser.add_argument("--lambda-fusion-balance", type=float, default=.02)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--patience", type=int, default=None)
    parser.add_argument('--no-validation', action='store_true',
                        help='GossipCop/Twitter: train the full training CSV for fixed epochs; test the final epoch without validation or calibration')
    parser.add_argument("--min-delta", type=float, default=1e-3); parser.add_argument("--loss-min-delta", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.1); parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--text-lr", type=float, default=1e-5); parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument("--lr-warmup-epochs", type=int, default=3, help="学习率线性预热轮数")
    parser.add_argument("--lr-start-ratio", type=float, default=0.01)
    parser.add_argument("--lr-min-ratio", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--vprm-checkpoint", help="Matching-language three-stage VPRM checkpoint; required for full-model training")
    parser.add_argument('--atm-keep-ratio', type=float, choices=tuple(i/10 for i in range(1,11)), default=.5)
    parser.add_argument("--freeze-text", action="store_true", help="Keep text frozen even with --train-backbones; text is frozen by default")
    parser.add_argument("--train-text-positions", action="store_true",
                        help="Train only English-CLIP positions 20:248; ignored for GossipCop, whose positions stay frozen")
    parser.add_argument("--train-backbones", action="store_true",
                        help="fine-tune both CLIP encoders; default freezes all encoders and VPRM")
    parser.add_argument("--data-root", help="Directory containing this dataset's train/val/test CSVs")
    parser.add_argument("--output", default=None)
    main(parser.parse_args())

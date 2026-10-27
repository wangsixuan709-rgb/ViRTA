"""Evaluate a trained ViRTA checkpoint and export every misclassified test row."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from classification_metrics import classification_metrics, metric_table
from ViRTA import ViRTA
from virta_dataset import build_dataset, collate_fn, configure_data_root
from train_virta import model_forward, move
from prepare_tenfold import FOLDS, protocol


def main(args):
    if Path(str(args.checkpoint) + '.cv.json').is_file():
        from cross_validation import export_saved_predictions
        export_saved_predictions(args.checkpoint, args.output)
        print('Exported all ten saved test evaluations without repeating inference')
        return
    config_path = Path(str(args.checkpoint) + '.config.json')
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    protocol(config.get('cv_count', FOLDS))
    data_root = args.data_root or config.get('data_root')
    if config.get('cv_splits') and args.data_root is not None:
        if Path(args.data_root).resolve() != Path(config['data_root']).resolve():
            raise ValueError('Use the held-out test data root recorded for this checkpoint fold')
    configure_data_root(args.dataset, data_root)
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    epoch_metadata = Path(str(args.checkpoint) + '.epoch.json')
    risk_calibrated = (json.loads(epoch_metadata.read_text()).get('risk_calibration', True)
                       if epoch_metadata.exists() else True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    dataset = build_dataset(args.dataset, "test", False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn)
    ratio=float(state.get('atm_keep_ratio',state.get('atrm_keep_ratio',.5)))
    model = ViRTA(dataset=args.dataset, atm_keep_ratio=ratio).to(device)
    model.load_state_dict(state)
    del state
    model.eval()

    records, labels, predictions = [], [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            logits, aux = model_forward(model, batch)
            probabilities = torch.softmax(logits, -1)
            prediction = probabilities.argmax(-1)
            branch_probs = aux["branch_probs"]
            branch_predictions = branch_probs.argmax(-1)
            size = len(prediction)
            for local in range(size):
                row_index = offset + local
                row = dataset.frame.iloc[row_index]
                label = int(batch["label"][local])
                pred = int(prediction[local])
                labels.append(label); predictions.append(pred)
                if pred == label:
                    continue
                text = str(row.get("text", row.get("content", "")))
                image = str(row.get("image", row.get("images", "")))
                sample_id = row.get("id", row.get("news_id", row.get("Unnamed: 0", row_index)))
                branches = branch_predictions[local].detach().cpu().tolist()
                record = {
                    "dataset": args.dataset, "row_index": row_index, "sample_id": sample_id,
                    "error_direction": "fake_to_real" if label == 0 else "real_to_fake",
                    "label": label, "prediction": pred,
                    "prob_fake": float(probabilities[local, 0]), "prob_real": float(probabilities[local, 1]),
                    "confidence": float(probabilities[local, pred]),
                    "text_prediction": branches[0], "image_prediction": branches[1],
                    "multimodal_prediction": branches[2],
                    "correct_branch_count": sum(value == label for value in branches),
                    "branch_disagreement": len(set(branches)) > 1,
                    "text_prob_real": float(branch_probs[local, 0, 1]),
                    "image_prob_real": float(branch_probs[local, 1, 1]),
                    "multimodal_prob_real": float(branch_probs[local, 2, 1]),
                    "text_weight": float(aux["weights"][local, 0]),
                    "image_weight": float(aux["weights"][local, 1]),
                    "multimodal_weight": float(aux["weights"][local, 2]),
                    "conflict_score": float(aux["conflict_score"][local]),
                    "fused_entropy": float(aux["fused_entropy"][local]),
                    "risk_score": float(aux["risk_score"][local]) if risk_calibrated else None,
                    "risk_flag": bool(aux["risk_flag"][local]) if risk_calibrated else None,
                    "text_mode": "chunks" if args.dataset != "weibo" else "tokens256",
                    "input_token_count": int(batch["token_count"][local]),
                    "input_chunk_count": int(batch["chunk_mask"][local].sum()),
                    "truncated": args.dataset == "weibo" and int(batch["token_count"][local]) > dataset.max_length,
                    "r_v": float(aux["r_v"][local]), "r_t": float(aux["r_t"][local]),
                    "min_visual_evidence": float(aux["a"][local].min()),
                    "min_text_evidence": float(aux["b"][local][aux["text_mask"][local]].min()),
                    "text_char_count": len(text), "text": text, "image": image,
                }
                records.append(record)
            offset += size

    errors = pd.DataFrame(records)
    if records:
        errors = errors.sort_values(["error_direction", "confidence"], ascending=[True, False])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output, index=False)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    summary = {
        "visual_restoration": "VPRM", "atm_keep_ratio": ratio,
        "risk_calibration_available": risk_calibrated,
        "dataset": args.dataset, "checkpoint": str(Path(args.checkpoint)), "device": str(device),
        "test_rows": len(labels), "errors": len(errors),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix_rows_true_fake_real": matrix,
        "fake_to_real": matrix[0][1], "real_to_fake": matrix[1][0],
        "all_errors_csv": str(output),
    }
    summary.update(classification_metrics(labels, predictions))
    pd.DataFrame([{k: summary[k] for k in ["accuracy", "fake_precision", "fake_recall", "fake_f1", "real_precision", "real_recall", "real_f1"]}]).to_csv(output.with_suffix(".metrics.csv"), index=False)
    print(metric_table(summary))
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("weibo", "gossipcop", "twitter"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Directory containing this dataset's test CSV")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())

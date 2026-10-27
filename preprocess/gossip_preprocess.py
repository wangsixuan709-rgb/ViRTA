"""Normalize local Gossip CSV/JSON records for ViRTA; preview by default."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def read_records(path):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        value = [dict(row, news_id=row.get("news_id", key)) for key, row in value.items()]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: JSON must be a record array or an ID-to-record object")
    return value


def normalize(path, split, args, mapping):
    rows, rejected = [], []
    for index, row in enumerate(read_records(path)):
        for key in (args.text_column, args.image_column, args.label_column):
            if key not in row:
                raise ValueError(f"{path} row {index + 1}: missing column {key}")
        raw_label = str(row[args.label_column]).strip().lower()
        if raw_label not in mapping:
            raise ValueError(f"{path} row {index + 1}: unmapped label {raw_label!r}")
        content = str(row[args.text_column] or "")
        value = str(row[args.image_column] or "").strip()
        image = Path(value)
        if not image.is_absolute():
            # This matches virta_dataset.py for the existing split CSVs.
            image = (args.image_root / image if args.image_root else
                     args.data_root / "Images" / f"gossip_{split}" / image.name)
        reason = "empty_text" if not content.strip() else ""
        if not reason:
            try:
                with Image.open(image) as source:
                    source.convert("RGB").load()
            except (OSError, ValueError):
                reason = "no_readable_image"
        if reason:
            rejected.append({"source": str(path), "row": index + 1, "reason": reason})
            continue
        news_id = row.get("news_id", row.get("id", row.get("Unnamed: 0", index)))
        rows.append({"news_id": str(news_id), "content": content,
                     "image": str(image.resolve()), "label": mapping[raw_label]})
    if not rows:
        raise ValueError(f"{path}: no valid records; check input and image paths")
    return rows, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/gossip")
    parser.add_argument("--train-input", type=Path)
    parser.add_argument("--test-input", type=Path)
    parser.add_argument("--input", type=Path, help="Unsplit CSV/JSON; uses a seeded stratified train/test split")
    parser.add_argument("--image-root", type=Path, help="Resolve relative input image paths under this directory")
    parser.add_argument("--text-column", default="content")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--label-map", default='{"0":0,"1":1,"fake":0,"real":1}', help="JSON mapping from source labels to 0=fake, 1=real")
    parser.add_argument("--test-size", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    mapping = json.loads(args.label_map)
    if not isinstance(mapping, dict) or not mapping or any(type(v) is not int or v not in (0, 1) for v in mapping.values()):
        parser.error("--label-map must be a JSON object with integer values 0 or 1")
    mapping = {str(k).strip().lower(): v for k, v in mapping.items()}
    if args.input:
        if args.train_input or args.test_input or not args.image_root:
            parser.error("--input requires --image-root and cannot be combined with split inputs")
        if not 0 < args.test_size < 1:
            parser.error("--test-size must be between 0 and 1")
        from sklearn.model_selection import train_test_split
        rows, rejected = normalize(args.input, "all", args, mapping)
        train, test = train_test_split(rows, test_size=args.test_size, random_state=args.seed,
                                      stratify=[row['label'] for row in rows])
    else:
        if bool(args.train_input) != bool(args.test_input):
            parser.error("Pass both --train-input and --test-input")
        train, rejected = normalize(args.train_input or args.data_root / "gossip_train.csv", "train", args, mapping)
        test, test_rejected = normalize(args.test_input or args.data_root / "gossip_test.csv", "test", args, mapping)
        rejected.extend(test_rejected)
    outputs = {"gossip_train.csv": train, "gossip_test.csv": test}
    report = {"outputs": {name: {"rows": len(rows), "labels": dict(Counter(row['label'] for row in rows))}
                          for name, rows in outputs.items()}, "label_map": mapping,
              "rejected": len(rejected), "reasons": dict(Counter(row['reason'] for row in rejected)),
              "split": {"mode": "stratified" if args.input else "preserved",
                        "seed": args.seed if args.input else None,
                        "test_size": args.test_size if args.input else None}}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write:
        destination = args.output_dir or args.data_root
        paths = [destination / name for name in outputs] + [destination / "gossip_preprocess_audit.json"]
        if not args.overwrite and any(path.exists() for path in paths):
            raise FileExistsError("Output exists; use a new --output-dir or explicitly pass --overwrite")
        destination.mkdir(parents=True, exist_ok=True)
        for name, rows in outputs.items():
            path = destination / name
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
        paths[-1].write_text(json.dumps({**report, "rejected_records": rejected}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

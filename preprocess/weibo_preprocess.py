"""Prepare three-line Weibo tweets for the current ViRTA loader."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image


def prepare(root):
    outputs, rejected = {}, []
    for split in ("train", "test"):
        rows = []
        for category, label in (("rumor", 0), ("nonrumor", 1)):
            source = root / "tweets" / f"{split}_{category}.txt"
            lines = source.read_text(encoding="utf-8-sig").splitlines()
            if len(lines) % 3:
                raise ValueError(f"{source}: expected three lines per record, got {len(lines)}")
            for offset in range(0, len(lines), 3):
                news_id = lines[offset].split("|")[0].strip()
                text = lines[offset + 2].strip()
                selected = None
                for url in lines[offset + 1].split("|"):
                    name = Path(unquote(urlsplit(url.strip()).path)).name
                    if name.lower() in ("", "null", "none"):
                        continue
                    candidate = root / f"{category}_images" / name
                    try:
                        with Image.open(candidate) as image:
                            image.convert("RGB").load()
                        selected = candidate
                        break
                    except (OSError, ValueError):
                        continue
                reason = "missing_id" if not news_id else "empty_text" if not text else "no_readable_image" if selected is None else ""
                if reason:
                    rejected.append({"source": str(source), "line": offset + 1,
                                     "id": news_id, "reason": reason})
                    continue
                rows.append({"id": news_id, "image": str(selected.resolve()), "label": label,
                             "text": text, "clip_text": text,
                             "label_name": "fake" if label == 0 else "real", "has_image": 1})
        if not rows:
            raise ValueError(f"No valid {split} records; check the raw data and image root")
        outputs[f"{split}.csv"] = rows
    return outputs, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/weibo/weibo")
    parser.add_argument("--output-dir", type=Path, help="Default: data root")
    parser.add_argument("--write", action="store_true", help="Write CSVs and rejection audit; otherwise preview only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs, rejected = prepare(args.data_root.resolve())
    destination = args.output_dir or args.data_root
    report = {"outputs": {name: {"rows": len(rows), "labels": dict(Counter(row['label'] for row in rows))}
                          for name, rows in outputs.items()},
              "rejected": len(rejected), "reasons": dict(Counter(row['reason'] for row in rejected))}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write:
        paths = [destination / name for name in outputs] + [destination / "weibo_preprocess_audit.json"]
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

"""
Trích xuất toàn bộ văn bản OCR từ ảnh trong một thư mục (không dùng câu hỏi VQA).

Cách dùng:
    python scripts/extract_ocr_st_images.py --config configs/internvl2_2b.yaml

    python scripts/extract_ocr_st_images.py --config configs/internvl2_2b.yaml \\
        --image_dir ./data/st_images --output results/st_images_ocr.json

    # Có checkpoint fine-tune:
    python scripts/extract_ocr_st_images.py --config configs/internvl2_2b.yaml \\
        --checkpoint checkpoints/internvl2_2b/best_model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.adapters import get_adapter
from src.dataset import VQADataCollator

DEFAULT_OCR_PROMPT = """Nhiệm vụ: chỉ trích xuất toàn bộ văn bản có trong hình (OCR).
- Liệt kê mọi chữ, số, ký hiệu đọc được, theo thứ tự hợp lý (ví dụ từ trên xuống, trái sang phải).
- Giữ nguyên xuống dòng nếu cần để phản ánh bố cục.
- Không giải thích, không trả lời câu hỏi khác; chỉ xuất nội dung chữ trong ảnh.
- Nếu không có chữ nào, trả lời đúng một dòng: (không có văn bản)"""


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(image_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for ext in extensions:
        files.extend(sorted(image_dir.glob(f"*{ext}")))
        files.extend(sorted(image_dir.glob(f"*{ext.upper()}")))
    # trùng tên khác hoa/thường
    seen: set[str] = set()
    unique: list[Path] = []
    for p in files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


@torch.no_grad()
def run_ocr(
    adapter,
    image_paths: list[Path],
    ocr_prompt: str,
    max_length: int,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
) -> list[dict]:
    collator = VQADataCollator(adapter, max_length=max_length, training=False)
    device = next(adapter.model.parameters()).device
    results: list[dict] = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="OCR"):
        batch_paths = image_paths[start : start + batch_size]
        raw_batch = []
        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            raw_batch.append({
                "image": image,
                "question": ocr_prompt,
                "answer": "",
                "question_id": path.stem,
                "image_id": path.stem,
            })

        inputs = collator(raw_batch)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        preds = adapter.generate(
            inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
        )

        for path, pred in zip(batch_paths, preds):
            results.append({
                "file": path.name,
                "path": str(path.resolve()),
                "ocr_text": pred,
            })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR toàn bộ ảnh trong thư mục (multimodal).")
    parser.add_argument("--config", type=str, required=True, help="YAML config (model + evaluation).")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/st_images",
        help="Thư mục chứa ảnh (mặc định: ./data/st_images).",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint LoRA (tuỳ chọn).")
    parser.add_argument(
        "--output",
        type=str,
        default="results/st_images_ocr.json",
        help="File JSON kết quả.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt OCR tùy chỉnh (mặc định: prompt trích xuất OCR đầy đủ).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".jpg,.jpeg,.png,.webp,.bmp",
        help="Danh sách đuôi file, cách nhau bởi dấu phẩy.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Ghi đè evaluation.batch_size.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Ghi đè evaluation.max_new_tokens.")
    parser.add_argument("--num_beams", type=int, default=None, help="Ghi đè evaluation.num_beams.")

    args = parser.parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg.get("evaluation", {})
    train_cfg = cfg.get("training", {})

    image_dir = Path(args.image_dir).resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục: {image_dir}")

    exts = tuple(
        e.strip() if e.strip().startswith(".") else f".{e.strip()}"
        for e in args.ext.split(",")
        if e.strip()
    )
    paths = list_images(image_dir, exts)
    if not paths:
        raise SystemExit(f"Không có ảnh ({', '.join(exts)}) trong {image_dir}")

    batch_size = args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 4)
    # OCR thường dài hơn câu trả lời VQA; mặc định 512 (config evaluation thường chỉ 64).
    max_new_tokens = (
        args.max_new_tokens if args.max_new_tokens is not None else 512
    )
    num_beams = args.num_beams if args.num_beams is not None else eval_cfg.get("num_beams", 1)
    max_length = train_cfg.get("max_length", 2048)

    ocr_prompt = (args.prompt or DEFAULT_OCR_PROMPT).strip()

    adapter = get_adapter(cfg["model"]["type"])
    adapter.load_for_inference(cfg, args.checkpoint)

    print(f"Model     : {cfg['model']['name']}")
    print(f"Images    : {len(paths):,} files in {image_dir}")
    print(f"Batch     : {batch_size}, max_new_tokens: {max_new_tokens}")

    results = run_ocr(
        adapter,
        paths,
        ocr_prompt,
        max_length=max_length,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": cfg["model"]["name"],
        "checkpoint": args.checkpoint,
        "image_dir": str(image_dir),
        "ocr_prompt": ocr_prompt,
        "num_images": len(results),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {len(results):,} kết quả → {out_path.resolve()}")


if __name__ == "__main__":
    main()

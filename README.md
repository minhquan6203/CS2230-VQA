# ViTextVQA — Fine-tuning Vision-Language Models

Fine-tuning các Vision-Language Model trên tập dữ liệu **ViTextVQA** (Vietnamese Text-based Visual Question Answering).

---

## Models được hỗ trợ

| Config | Model | Params | Tiếng Việt | VRAM (train) |
|--------|-------|:------:|:----------:|:------------:|
| `configs/qwen2vl_2b.yaml` | Qwen/Qwen2-VL-2B-Instruct | 2B | ✅ Tốt | ~8 GB |
| `configs/internvl2_2b.yaml` | OpenGVLab/InternVL2-2B | 2B | ✅ Tốt | ~7 GB |
| `configs/internvl2_4b.yaml` | OpenGVLab/InternVL2-4B | 4B | ✅ Tốt | ~12 GB |
| `configs/smolvlm_500m.yaml` | HuggingFaceTB/SmolVLM-500M-Instruct | 500M | ⚠️ Hạn chế | ~4 GB |
| `configs/smolvlm_500m_full.yaml` | HuggingFaceTB/SmolVLM-500M-Instruct | 500M | ⚠️ Hạn chế | ~6 GB |
| `configs/smolvlm_2b.yaml` | HuggingFaceTB/SmolVLM-Instruct | 2B | ⚠️ Hạn chế | ~8 GB |
| `configs/smolvlm2_2b.yaml` | HuggingFaceTB/SmolVLM2-2.2B-Instruct | 2.2B | ⚠️ Hạn chế | ~8 GB |

> **Khuyến nghị:** Dùng **Qwen2-VL-2B** hoặc **InternVL2-4B** cho kết quả tốt nhất trên tiếng Việt.

---

## Cấu trúc thư mục

```
CS2230-VQA/
├── configs/
│   ├── qwen2vl_2b.yaml
│   ├── internvl2_2b.yaml
│   ├── internvl2_4b.yaml
│   ├── smolvlm_500m.yaml
│   ├── smolvlm_500m_full.yaml
│   ├── smolvlm_2b.yaml
│   └── smolvlm2_2b.yaml
├── data/
│   ├── ViTextVQA_train.json       # 35,159 samples
│   ├── ViTextVQA_dev.json         # 5,155 samples
│   ├── ViTextVQA_test_gt .json
│   └── st_images/                 # ảnh *.jpg
├── src/
│   ├── adapters/
│   │   ├── __init__.py            # Factory: get_adapter(type)
│   │   ├── base.py                # Abstract BaseAdapter
│   │   ├── qwen2vl.py             # Qwen2-VL adapter
│   │   ├── internvl2.py           # InternVL2 / InternVL2.5 adapter
│   │   └── smolvlm.py             # SmolVLM / SmolVLM2 adapter
│   ├── dataset.py                 # ViTextVQADataset + VQADataCollator
│   ├── metrics.py                 # ANLS + Exact Match
│   ├── train.py                   # Training script
│   └── evaluate.py                # Inference + Evaluation script
├── checkpoints/                   # LoRA checkpoints (gitignored)
├── results/                       # Predictions JSON output
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

Nếu dùng **InternVL2**, cài thêm:
```bash
pip install einops timm
```

> **Lưu ý:** `bitsandbytes` yêu cầu CUDA. Trên Colab dùng GPU T4 (16 GB) là đủ cho tất cả model 2B với QLoRA.

---

## Training

```bash
# Qwen2-VL-2B (khuyến nghị cho tiếng Việt)
python -m src.train --config configs/qwen2vl_2b.yaml

# InternVL2-2B
python -m src.train --config configs/internvl2_2b.yaml

# InternVL2-4B (cần ~12 GB VRAM)
python -m src.train --config configs/internvl2_4b.yaml

# SmolVLM-500M + LoRA (nhanh nhất)
python -m src.train --config configs/smolvlm_500m.yaml

# SmolVLM-500M full fine-tune (không LoRA)
python -m src.train --config configs/smolvlm_500m_full.yaml
```

Checkpoint tốt nhất theo **ANLS trên dev set** được lưu tại `checkpoints/<model>/best_model/`.

---

## Evaluation

```bash
# Zero-shot (base model, không fine-tune)
python -m src.evaluate --config configs/qwen2vl_2b.yaml

# Fine-tuned model
python -m src.evaluate \
    --config configs/qwen2vl_2b.yaml \
    --checkpoint checkpoints/qwen2vl_2b/best_model

# Chỉ định file output
python -m src.evaluate \
    --config configs/internvl2_2b.yaml \
    --checkpoint checkpoints/internvl2_2b/best_model \
    --results_file results/internvl2_2b_test.json
```

Output terminal:
```
=======================================================
  Model  : Qwen/Qwen2-VL-2B-Instruct
  Mode   : Fine-tuned  →  checkpoints/qwen2vl_2b/best_model
=======================================================
  ANLS         : 65.42%
  Exact Match  : 51.30%
  Num samples  : 4013
=======================================================
```

---

## Metrics

| Metric | Mô tả |
|--------|-------|
| **ANLS** | Average Normalized Levenshtein Similarity — metric chính của TextVQA. Đo độ tương đồng chuỗi, threshold = 0.5 (nếu similarity < 0.5 → tính bằng 0). |
| **Exact Match** | Tỉ lệ khớp chính xác sau khi chuẩn hóa (lowercase, Unicode NFC, bỏ ký tự đặc biệt). |

---

## Kỹ thuật tối ưu GPU

| Kỹ thuật | Tác dụng |
|----------|----------|
| **QLoRA** (4-bit NF4 + LoRA r=16) | Giảm ~75% VRAM cho weights, chỉ train ~1% tham số |
| **Gradient checkpointing** | Giảm activation memory (~20% chậm hơn) |
| **bf16 / fp16** | Mixed precision, giảm ~50% memory cho activations |
| **batch_size=1 + grad_accum** | Peak VRAM thấp, effective batch size vẫn lớn |
| **max_pixels** (Qwen2-VL) | Kiểm soát số image patches, giảm sequence length |
| **max_num_tiles** (InternVL2) | Giới hạn số tiles, kiểm soát VRAM và tốc độ |

---

## Thêm model mới

1. Tạo `src/adapters/<model_type>.py` kế thừa `BaseAdapter`:

```python
from .base import BaseAdapter

class MyModelAdapter(BaseAdapter):
    def load(self, cfg): ...
    def load_for_inference(self, cfg, checkpoint=None): ...
    def process_batch(self, items, max_length=1024, training=True): ...
    def generate(self, inputs, max_new_tokens=64, num_beams=1): ...
```

2. Đăng ký trong `src/adapters/__init__.py`:

```python
from .my_model import MyModelAdapter

_REGISTRY = {
    ...,
    "mymodel": MyModelAdapter,
}
```

3. Tạo `configs/mymodel.yaml` với `model.type: "mymodel"`.

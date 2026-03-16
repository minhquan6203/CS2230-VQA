"""
Adapter cho InternVL3.5 (1B, 2B, 4B, 8B, ...).
Supported models:
  - OpenGVLab/InternVL3_5-1B
  - OpenGVLab/InternVL3_5-2B
  - OpenGVLab/InternVL3_5-4B
  - OpenGVLab/InternVL3_5-8B

Đặc điểm:
  - Kế thừa toàn bộ logic xử lý ảnh / batch / generate từ InternVL2Adapter
  - LLM backbone chuyển sang Qwen3 → dùng dtype khi load model
  - Cascade RL training pipeline (CPT + SFT + CascadeRL)
  - Yêu cầu transformers>=4.52.1
"""

import torch
from transformers import AutoModel, AutoTokenizer

from .internvl2 import InternVL2Adapter, _patch_meta_linspace, IMG_CONTEXT_TOKEN


class InternVL3_5Adapter(InternVL2Adapter):
    """InternVL3.5 — kế thừa InternVL2, override load cho Qwen3 LLM backbone."""

    # process_batch, generate, _expand_image_placeholder, _apply_qlora
    # đều kế thừa từ InternVL2Adapter (interface giống hệt).

    def load(self, cfg: dict) -> None:
        """Training mode: full fine-tune hoặc QLoRA tuỳ config."""
        model_name = cfg["model"]["name"]
        self._max_num_tiles = cfg["model"].get("max_num_tiles", 6)
        use_lora = "lora" in cfg
        use_quant = "quantization" in cfg

        print(f"[InternVL3.5] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        load_kwargs = dict(
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self._get_device_map(cfg["model"].get("device")),
        )
        if use_quant:
            print(f"[InternVL3.5] Loading model (4-bit): {model_name}")
            load_kwargs["quantization_config"] = self._build_bnb_config(cfg["quantization"])
        else:
            print(f"[InternVL3.5] Loading model (bf16): {model_name}")

        with _patch_meta_linspace():
            model = AutoModel.from_pretrained(model_name, **load_kwargs)

        model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._num_image_token = model.num_image_token

        if use_lora:
            self.model = self._apply_qlora(model, cfg["lora"])
        else:
            model.train()
            self.model = model
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"trainable params: {trainable:,} || all params: {total:,} "
                f"|| trainable%: {trainable / total * 100:.4f}"
            )

    def load_for_inference(self, cfg: dict, checkpoint: str | None = None) -> None:
        """Inference mode: full precision, optional LoRA merge."""
        from peft import PeftModel

        model_name = cfg["model"]["name"]
        self._max_num_tiles = cfg["model"].get("max_num_tiles", 6)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print(f"[InternVL3.5] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        print(f"[InternVL3.5] Loading base model: {model_name}")
        with _patch_meta_linspace():
            base = AutoModel.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
                device_map=self._get_device_map(cfg["model"].get("device")),
            )
        base.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._num_image_token = base.num_image_token

        if checkpoint:
            print(f"[InternVL3.5] Merging LoRA từ: {checkpoint}")
            self.model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        else:
            self.model = base
        self.model.eval()

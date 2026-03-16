"""
Adapter cho Qwen3.5 (0.8B, 2B, ...).
Supported models:
  - Qwen/Qwen3.5-2B
  - Qwen/Qwen3.5-0.8B

Đặc điểm:
  - Unified Vision-Language Foundation (early fusion)
  - Hybrid architecture: Gated DeltaNet + Gated Attention + FFN
  - Dùng AutoProcessor + AutoModelForImageTextToText
  - Yêu cầu transformers mới nhất (pip install transformers@git+...main)
"""

import torch

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
from transformers import AutoProcessor

from .base import BaseAdapter


class Qwen3_5Adapter(BaseAdapter):

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    def load(self, cfg: dict) -> None:
        """Training mode: full fine-tune hoặc QLoRA tuỳ config."""
        model_name = cfg["model"]["name"]
        use_lora = "lora" in cfg
        use_quant = "quantization" in cfg

        print(f"[Qwen3.5] Loading processor: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.pad_token_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        load_kwargs = dict(
            device_map=self._get_device_map(cfg["model"].get("device")),
            torch_dtype=torch.bfloat16,
        )
        if use_quant:
            print(f"[Qwen3.5] Loading model (4-bit): {model_name}")
            load_kwargs["quantization_config"] = self._build_bnb_config(cfg["quantization"])
        else:
            print(f"[Qwen3.5] Loading model (bf16): {model_name}")

        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)

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
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        proc_src = checkpoint or model_name
        print(f"[Qwen3.5] Loading processor từ: {proc_src}")
        self.processor = AutoProcessor.from_pretrained(proc_src)
        self.pad_token_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        print(f"[Qwen3.5] Loading base model: {model_name}")
        base = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self._get_device_map(cfg["model"].get("device")),
        )
        if checkpoint:
            print(f"[Qwen3.5] Merging LoRA từ: {checkpoint}")
            self.model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        else:
            self.model = base
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  process_batch                                                       #
    # ------------------------------------------------------------------ #

    def _build_texts(
        self, items: list[dict], training: bool
    ) -> tuple[list[str], list[str], list]:
        """Trả về (full_texts, prompt_texts, images)."""
        full_texts, prompt_texts, images = [], [], []

        for item in items:
            messages_user = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["question"]},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                messages_user, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)
            images.append(item["image"])

            if training:
                messages_full = messages_user + [
                    {"role": "assistant", "content": item.get("answer", "")}
                ]
                full_text = self.processor.apply_chat_template(
                    messages_full, tokenize=False, add_generation_prompt=False
                )
            else:
                full_text = prompt_text
            full_texts.append(full_text)

        return full_texts, prompt_texts, images

    def process_batch(
        self,
        items: list[dict],
        max_length: int = 512,
        training: bool = True,
    ) -> dict:
        full_texts, prompt_texts, images = self._build_texts(items, training)

        enc = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        result = dict(enc)
        seq_len = enc.input_ids.shape[1]
        if seq_len > max_length:
            for k, v in result.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == seq_len:
                    result[k] = v[:, :max_length]

        if training:
            input_ids = result["input_ids"]
            attention_mask = result["attention_mask"]
            labels = input_ids.clone()
            for i, (full_text, prompt_text) in enumerate(zip(full_texts, prompt_texts)):
                answer_ids = self.processor.tokenizer.encode(
                    full_text[len(prompt_text):], add_special_tokens=False
                )
                real_len = int(attention_mask[i].sum().item())
                prompt_len = max(0, real_len - len(answer_ids))
                labels[i, :prompt_len] = -100
            labels[input_ids == self.pad_token_id] = -100
            result["labels"] = labels

        return result

    # ------------------------------------------------------------------ #
    #  Generate                                                            #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = 64,
        num_beams: int = 1,
    ) -> list[str]:
        input_len = inputs["input_ids"].shape[1]
        gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        generated = self.model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=self.pad_token_id,
        )
        return [
            self.processor.tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
            for seq in generated
        ]

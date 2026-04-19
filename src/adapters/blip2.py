"""
Adapter cho BLIP-2 (Bootstrapping Language-Image Pre-training, 2nd edition).
Supported models:
  - Salesforce/blip2-opt-2.7b
  - Salesforce/blip2-opt-6.7b
  - Salesforce/blip2-flan-t5-xl
  - Salesforce/blip2-flan-t5-xxl

Đặc điểm:
  - Dùng Blip2ForConditionalGeneration + Blip2Processor
  - Không dùng chat template — prompt dạng "Question: ... Answer:"
  - Hỗ trợ QLoRA: OPT backbone dùng TaskType.CAUSAL_LM,
                   Flan-T5 backbone dùng TaskType.SEQ_2_SEQ_LM
  - BLIP-2 tự prepend -100 cho query tokens trong forward() —
    labels chỉ cần cùng chiều dài với input_ids.
  - generate() trả về chỉ token IDs của phần được sinh (không kèm input/query tokens),
    nên decode trực tiếp mà không cần slice.
"""

import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from .base import BaseAdapter


class BLIP2Adapter(BaseAdapter):

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    def load(self, cfg: dict) -> None:
        """Training mode: full fine-tune hoặc QLoRA tuỳ config."""
        model_name = cfg["model"]["name"]
        self._load_system_prompt(cfg)
        use_lora = "lora" in cfg
        use_quant = "quantization" in cfg

        print(f"[BLIP2] Loading processor: {model_name}")
        self.processor = Blip2Processor.from_pretrained(model_name)
        tok = self.processor.tokenizer
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        load_kwargs: dict = dict(
            device_map=self._get_device_map(cfg["model"].get("device")),
            torch_dtype=torch.bfloat16,
        )
        if use_quant:
            print(f"[BLIP2] Loading model (4-bit): {model_name}")
            load_kwargs["quantization_config"] = self._build_bnb_config(cfg["quantization"])
        else:
            print(f"[BLIP2] Loading model (full precision): {model_name}")

        model = Blip2ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

        if use_lora:
            self.model = self._apply_blip2_lora(model, cfg["lora"])
        else:
            model.train()
            self.model = model
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"trainable params: {trainable:,} || all params: {total:,} || "
                f"trainable%: {trainable / total * 100:.4f}"
            )

    def load_for_inference(self, cfg: dict, checkpoint: str | None = None) -> None:
        """Inference mode: full precision, optional LoRA merge."""
        from peft import PeftModel

        model_name = cfg["model"]["name"]
        self._load_system_prompt(cfg)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        proc_src = checkpoint or model_name
        print(f"[BLIP2] Loading processor từ: {proc_src}")
        self.processor = Blip2Processor.from_pretrained(proc_src)
        tok = self.processor.tokenizer
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        print(f"[BLIP2] Loading base model: {model_name}")
        base = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self._get_device_map(cfg["model"].get("device")),
        )
        if checkpoint:
            print(f"[BLIP2] Merging LoRA từ: {checkpoint}")
            self.model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        else:
            self.model = base
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  LoRA — phân biệt OPT (causal) vs. Flan-T5 (seq2seq)               #
    # ------------------------------------------------------------------ #

    def _apply_blip2_lora(self, model, lora_cfg: dict):
        """
        Override _apply_qlora vì BLIP-2 có thể dùng Flan-T5 (SEQ_2_SEQ_LM)
        thay vì OPT (CAUSAL_LM) làm language backbone.
        """
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        is_enc_dec = model.language_model.config.is_encoder_decoder
        task_type = TaskType.SEQ_2_SEQ_LM if is_enc_dec else TaskType.CAUSAL_LM

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type=task_type,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    # ------------------------------------------------------------------ #
    #  process_batch                                                       #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, question: str) -> str:
        """Tạo prompt VQA theo format BLIP-2 (không dùng chat template)."""
        prefix = f"{self.system_prompt}\n" if self.system_prompt else ""
        return f"{prefix}Question: {question} Answer:"

    def process_batch(
        self,
        items: list[dict],
        max_length: int = 512,
        training: bool = True,
    ) -> dict:
        prompts = [self._build_prompt(item["question"]) for item in items]
        images = [item["image"] for item in items]

        if training:
            answers = [item.get("answer", "") for item in items]
            # Thêm dấu cách giữa prompt và answer
            full_texts = [f"{p} {a}" for p, a in zip(prompts, answers)]

            enc = self.processor(
                images=images,
                text=full_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            labels = enc.input_ids.clone()
            tok = self.processor.tokenizer
            for i, (full_text, prompt) in enumerate(zip(full_texts, prompts)):
                # Encode phần answer (suffix sau prompt) để xác định prompt_len
                answer_suffix = full_text[len(prompt):]
                answer_ids = tok.encode(answer_suffix, add_special_tokens=False)
                real_len = int(enc.attention_mask[i].sum().item())
                prompt_len = max(0, real_len - len(answer_ids))
                labels[i, :prompt_len] = -100
            # Mask padding
            labels[enc.input_ids == self.pad_token_id] = -100

            result = dict(enc)
            result["labels"] = labels
            return result
        else:
            enc = self.processor(
                images=images,
                text=prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            return dict(enc)

    # ------------------------------------------------------------------ #
    #  Generate                                                            #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = 64,
        num_beams: int = 1,
    ) -> list[str]:
        """
        BLIP-2 internals: Q-Former output được prepend dưới dạng embeddings (không
        phải token IDs) trước khi đưa vào LM. LM.generate() nhận inputs_embeds nên
        output sequences chỉ chứa token IDs của phần được sinh → decode trực tiếp.
        """
        gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        generated = self.model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        return [
            self.processor.tokenizer.decode(seq, skip_special_tokens=True).strip()
            for seq in generated
        ]

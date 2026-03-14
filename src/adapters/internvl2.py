"""
Adapter cho InternVL2 và InternVL2.5.
Supported models:
  - OpenGVLab/InternVL2-2B
  - OpenGVLab/InternVL2-4B
  - OpenGVLab/InternVL2-8B
  - OpenGVLab/InternVL2_5-4B
  - OpenGVLab/InternVL2_5-8B

Đặc điểm:
  - Dùng trust_remote_code=True (model có custom code riêng)
  - Tokenizer thuần (không có AutoProcessor)
  - Dynamic resolution: ảnh chia tiles, mỗi tile = num_image_token IMG_CONTEXT tokens
  - pixel_values shape: [total_tiles, 3, image_size, image_size]
  - image_flags: [total_tiles] — đánh dấu tile nào là ảnh thật
"""

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseAdapter


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

INTERNVL_SYSTEM = (
    "Bạn là trợ lý AI hữu ích, chuyên trả lời các câu hỏi "
    "về nội dung trong ảnh bằng tiếng Việt."
)


# ------------------------------------------------------------------ #
#  Image preprocessing                                                #
# ------------------------------------------------------------------ #

def _build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_best_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for r in target_ratios:
        diff = abs(aspect_ratio - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
        elif diff == best_diff and area > 0.5 * image_size ** 2 * r[0] * r[1]:
            best = r
    return best


def _dynamic_preprocess(
    image: Image.Image, min_num: int = 1, max_num: int = 6,
    image_size: int = 448, use_thumbnail: bool = True
) -> list[Image.Image]:
    w, h = image.size
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tr = _find_best_ratio(w / h, target_ratios, w, h, image_size)
    tw, th = image_size * tr[0], image_size * tr[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = [
        resized.crop((
            (i % cols) * image_size, (i // cols) * image_size,
            (i % cols + 1) * image_size, (i // cols + 1) * image_size,
        ))
        for i in range(tr[0] * tr[1])
    ]
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _load_pixel_values(
    image: Image.Image, max_num_tiles: int, image_size: int
) -> torch.Tensor:
    tiles = _dynamic_preprocess(
        image, max_num=max_num_tiles, image_size=image_size, use_thumbnail=True
    )
    transform = _build_transform(image_size)
    return torch.stack([transform(t) for t in tiles])  # [num_tiles, 3, H, W]


# ------------------------------------------------------------------ #
#  Adapter                                                            #
# ------------------------------------------------------------------ #

class InternVL2Adapter(BaseAdapter):

    image_size: int = 448
    max_num_tiles: int = 6
    num_image_token: int = 256

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _setup_tokenizer(self, source: str):
        tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False)
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        return tok

    def _read_model_cfg(self, cfg: dict):
        mc = cfg["model"]
        self.image_size = mc.get("image_size", 448)
        self.max_num_tiles = mc.get("max_num_tiles", 6)

    def _cache_num_image_token(self, model):
        """Đọc num_image_token từ model config, lưu vào self."""
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        self.num_image_token = getattr(base.config, "num_image_token", 256)

    # ------------------------------------------------------------------ #
    #  Load                                                               #
    # ------------------------------------------------------------------ #

    def load(self, cfg: dict) -> None:
        self._read_model_cfg(cfg)
        model_name = cfg["model"]["name"]
        use_lora = "lora" in cfg
        use_quant = "quantization" in cfg

        print(f"[InternVL2] Loading tokenizer: {model_name}")
        self.processor = self._setup_tokenizer(model_name)

        load_kwargs = dict(
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if use_quant:
            print(f"[InternVL2] Loading model (4-bit): {model_name}")
            load_kwargs["quantization_config"] = self._build_bnb_config(cfg["quantization"])
        else:
            print(f"[InternVL2] Loading model (bf16): {model_name}")

        model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self._cache_num_image_token(model)

        if use_lora:
            self.model = self._apply_qlora(model, cfg["lora"])
        else:
            model.train()
            self.model = model
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {trainable/total*100:.4f}")

    def load_for_inference(self, cfg: dict, checkpoint: str | None = None) -> None:
        from peft import PeftModel

        self._read_model_cfg(cfg)
        model_name = cfg["model"]["name"]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        tok_src = checkpoint or model_name
        print(f"[InternVL2] Loading tokenizer từ: {tok_src}")
        self.processor = self._setup_tokenizer(tok_src)

        print(f"[InternVL2] Loading base model: {model_name}")
        base = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        self._cache_num_image_token(base)

        if checkpoint:
            print(f"[InternVL2] Merging LoRA từ: {checkpoint}")
            self.model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        else:
            self.model = base
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Text formatting                                                    #
    # ------------------------------------------------------------------ #

    def _image_placeholder(self, num_tiles: int) -> str:
        """<img><IMG_CONTEXT>×(num_tiles×num_image_token)</img>"""
        ctx = IMG_CONTEXT_TOKEN * (self.num_image_token * num_tiles)
        return f"\n{IMG_START_TOKEN}{ctx}{IMG_END_TOKEN}\n"

    def _build_texts(
        self, question: str, num_tiles: int, answer: str = "", training: bool = True
    ) -> tuple[str, str]:
        img_ph = self._image_placeholder(num_tiles)
        prompt = (
            f"<|im_start|>system\n{INTERNVL_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{img_ph}{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full = prompt + answer + "<|im_end|>" if (training and answer) else prompt
        return full, prompt

    # ------------------------------------------------------------------ #
    #  process_batch                                                      #
    # ------------------------------------------------------------------ #

    def process_batch(
        self,
        items: list[dict],
        max_length: int = 1024,
        training: bool = True,
    ) -> dict:
        all_input_ids, all_masks, all_labels = [], [], []
        all_pixel_values, all_image_flags = [], []

        for item in items:
            pixel_values = _load_pixel_values(
                item["image"], self.max_num_tiles, self.image_size
            )
            num_tiles = pixel_values.shape[0]
            all_pixel_values.append(pixel_values)
            all_image_flags.append(torch.ones(num_tiles, dtype=torch.long))

            answer = item.get("answer", "") if training else ""
            full_text, prompt_text = self._build_texts(
                item["question"], num_tiles, answer, training
            )

            enc = self.processor(full_text, return_tensors="pt")
            ids = enc.input_ids[0]
            mask = enc.attention_mask[0]

            if training:
                labels = self._compute_labels(ids, full_text, prompt_text, self.processor)
                all_labels.append(labels)

            all_input_ids.append(ids)
            all_masks.append(mask)

        max_len = max(t.size(0) for t in all_input_ids)
        result = {
            "input_ids": self._pad_1d(all_input_ids, self.pad_token_id, max_len),
            "attention_mask": self._pad_1d(all_masks, 0, max_len),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
            "image_flags": torch.cat(all_image_flags, dim=0),
        }
        if all_labels:
            result["labels"] = self._pad_1d(all_labels, -100, max_len)
        return result

    # ------------------------------------------------------------------ #
    #  Generate                                                           #
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
            self.processor.decode(seq[input_len:], skip_special_tokens=True).strip()
            for seq in generated
        ]

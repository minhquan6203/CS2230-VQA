"""
Factory để tạo adapter tương ứng từ config.

Thêm model mới:
  1. Tạo file src/adapters/<model_type>.py kế thừa BaseAdapter
  2. Đăng ký vào _REGISTRY bên dưới
  3. Tạo config yaml trong configs/
"""

from .base import BaseAdapter
from .blip2 import BLIP2Adapter
from .internvl2 import InternVL2Adapter
from .internvl3_5 import InternVL3_5Adapter
from .qwen2vl import Qwen2VLAdapter
from .qwen3_5 import Qwen3_5Adapter
from .smolvlm import SmolVLMAdapter

_REGISTRY: dict[str, type[BaseAdapter]] = {
    "blip2": BLIP2Adapter,
    "internvl2": InternVL2Adapter,
    "internvl3_5": InternVL3_5Adapter,
    "qwen2vl": Qwen2VLAdapter,
    "qwen3_5": Qwen3_5Adapter,
    "smolvlm": SmolVLMAdapter,
}


def get_adapter(model_type: str) -> BaseAdapter:
    """
    Trả về adapter instance từ model type string.

    Args:
        model_type: một trong các key trong _REGISTRY
                    (lấy từ field `model.type` trong config.yaml)
    """
    model_type = model_type.lower()
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[model_type]()

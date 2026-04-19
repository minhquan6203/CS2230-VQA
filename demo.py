import gradio as gr
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=dtype,
    device_map=None
).to(device)
model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

SYSTEM_PROMPT = """
Bạn là một hệ thống VQA (Visual Question Answering) chuyên xử lý tiếng Việt.

Nhiệm vụ:
- Trả lời câu hỏi dựa trên hình ảnh được cung cấp.
- Nếu câu hỏi không liên quan đến hình ảnh, hãy trả lời bình thường.
- Nếu thông tin trong ảnh không đủ → nói rõ "không đủ thông tin".

Yêu cầu:
- Trả lời NGẮN GỌN, CHÍNH XÁC.
- Ưu tiên tiếng Việt.
- Không suy đoán khi không chắc chắn.
- Nếu có OCR (text trong ảnh), hãy tận dụng.

Format:
- Trả lời trực tiếp, không giải thích dài dòng.
"""

def qwen_chat_fn(message, history):
    text = message.get("text", "")
    files = message.get("files", [])

    messages = []

    # system prompt
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    })

    # history
    for hist_item in history:
        messages.append({
            "role": hist_item["role"],
            "content": [{"type": "text", "text": hist_item["content"]}]
        })

    # current input
    current_content = []

    # images
    if files:
        for file_path in files:
            try:
                image = Image.open(file_path).convert("RGB")
                current_content.append({
                    "type": "image",
                    "image": image
                })
            except Exception as e:
                print(f"Error loading image: {e}")

    # text
    if text:
        current_content.append({
            "type": "text",
            "text": text
        })

    if not current_content:
        return ""

    messages.append({
        "role": "user",
        "content": current_content
    })

    # tokenize
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )

    # decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


demo = gr.ChatInterface(
    fn=qwen_chat_fn,
    type="messages",
    multimodal=True,
    title="🇻🇳 Qwen3-VL Vi VQA",
    description="""
Chat với mô hình Qwen3-VL với tiếng Việt.

- Hỗ trợ tiếng Việt
- Hỏi đáp dựa trên ảnh (VQA)
- OCR + reasoning

Upload ảnh + đặt câu hỏi để bắt đầu.
"""
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

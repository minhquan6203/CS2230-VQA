"""
Metrics cho ViTextVQA:
- F1 Score (set-based, token-level): metric chính
- Exact Match (EM): metric phụ
"""

import re
import string
import unicodedata


def normalize_text(text: str) -> str:
    """Xóa punctuation, lowercase, strip."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text


def preprocess_sentence(sentence: str) -> str:
    """Chuẩn hóa text: lowercase, NFC, tách punctuation thành token riêng."""
    sentence = sentence.lower()
    sentence = unicodedata.normalize("NFC", sentence)
    sentence = re.sub(r'[\u201c\u201d]', '"', sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r'"', ' " ', sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def _normalize(text: str) -> str:
    return preprocess_sentence(normalize_text(text))


def f1_score(prediction: str, ground_truths: list[str]) -> float:
    """
    Set-based token-level F1 cho 1 câu hỏi.
    Lấy max F1 với tất cả ground truth answers.
    """
    pred_tokens = _normalize(prediction).split()

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = _normalize(gt).split()
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            best_f1 = max(best_f1, int(pred_tokens == gt_tokens))
            continue

        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue

        precision = len(common) / len(set(pred_tokens))
        recall = len(common) / len(set(gt_tokens))
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def exact_match_score(prediction: str, ground_truths: list[str]) -> float:
    """Exact match sau khi chuẩn hóa."""
    pred = _normalize(prediction)
    return float(any(_normalize(gt) == pred for gt in ground_truths))


def compute_metrics(predictions: list[str], ground_truths: list[list[str]]) -> dict:
    """
    Tính F1 và EM trên toàn bộ tập dữ liệu.

    Args:
        predictions: danh sách câu trả lời dự đoán
        ground_truths: danh sách các câu trả lời đúng (mỗi câu hỏi có thể có nhiều đáp án)

    Returns:
        {"f1": float, "exact_match": float, "num_samples": int}
    """
    assert len(predictions) == len(ground_truths), (
        f"Số lượng predictions ({len(predictions)}) "
        f"không khớp ground_truths ({len(ground_truths)})"
    )

    total_f1 = 0.0
    total_em = 0.0

    for pred, gts in zip(predictions, ground_truths):
        total_f1 += f1_score(pred, gts)
        total_em += exact_match_score(pred, gts)

    n = len(predictions)
    return {
        "f1": total_f1 / n if n > 0 else 0.0,
        "exact_match": total_em / n if n > 0 else 0.0,
        "num_samples": n,
    }

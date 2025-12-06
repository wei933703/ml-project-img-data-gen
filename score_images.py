# score_images.py
import os
import csv
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image

import ImageReward
import clip
import hpsv2
from transformers import CLIPProcessor, CLIPModel


def get_device() -> torch.device:
    """
    自動偵測可用裝置：CUDA -> MPS -> CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_prompts_csv(csv_path: str) -> List[Tuple[str, str]]:
    """
    從 CSV 讀取 (id, prompt)。
    """
    records: List[Tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_id = str(row["id"]).strip()
            prompt = str(row["prompt"]).strip()
            if prompt_id and prompt:
                records.append((prompt_id, prompt))
    return records


# ---------------------------
# 模型載入
# ---------------------------

def load_image_reward_model():
    """
    載入 ImageReward 模型。
    """
    model = ImageReward.load("ImageReward-v1.0")
    return model


def load_clip_model(
    device: torch.device | None = None,
):
    """
    載入 CLIP 模型與 Processor。
    """
    if device is None:
        device = get_device()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return clip_model, processor

    # clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    # return clip_model, clip_preprocess


# ---------------------------
# 各種評分 function
# ---------------------------

def score_image_reward(
    image_reward_model: ImageReward.ImageReward,
    prompt: str,
    image_path: str,
) -> float:
    """
    使用 ImageReward 取得分數。
    """
    # ImageReward.score 接受路徑或 PIL 物件的 list，回傳 list 分數 [oai_citation:6‡GitHub](https://github.com/p1atdev/ImageReward-PickScore?utm_source=chatgpt.com)
    with torch.no_grad():
        score = image_reward_model.score(prompt, image_path)
    print(f"    ImageReward scores: {score}")
    return float(score)


def score_clip(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    prompt: str,
    image_path: str,
    device: torch.device | None = None,
) -> float:
    """
    使用 CLIP 計算圖文相似度分數（logits_per_image）。
    """
    if device is None:
        device = get_device()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: (batch_size, num_texts)
        score = logits_per_image[0, 0].item()

    print(f"    CLIP score: {score}")
    return float(score)


def score_hps(
    prompt: str,
    image_path: str,
) -> float:
    """
    使用 HPSv2 的 score 函數。
    hpsv2.score 可以接受單一影像路徑或列表，回傳評分結果 [oai_citation:7‡PyPI](https://pypi.org/project/hpsv2/)
    """
    # 這裡簡化為單張圖片
    result = hpsv2.score(image_path, prompt, hps_version="v2.1")
    
    print(f"    HPSv2 score: {result[0] if isinstance(result, (list, tuple)) else result}")

    # result 可能是 float 或 list，保險起見處理一下
    if isinstance(result, (list, tuple)):
        return float(result[0])
    return float(result)


# ---------------------------
# 主流程：對所有圖片評分並寫 CSV
# ---------------------------

def score_all_images(
    prompts_csv: str = "prompts.csv",
    images_dir: str = "outputs",
    output_csv: str = "scores.csv",
) -> None:
    """
    對所有 (id, prompt) 的圖片進行 ImageReward、CLIP、HPSv2 評分，並輸出到 CSV。
    """
    device = get_device()
    print(f"Using device: {device}")

    prompts = read_prompts_csv(prompts_csv)

    print("[Load] ImageReward model ...")
    image_reward_model = load_image_reward_model()
    print("[Load] ImageReward model loaded.")

    print("[Load] CLIP model ...")
    clip_model, clip_processor = load_clip_model(device=device)
    print("[Load] CLIP model loaded.")

    fieldnames = ["id", "prompt", "image_path", "image_reward", "clip", "hps"]

    rows: List[Dict[str, Any]] = []

    for prompt_id, prompt in prompts:
        image_path = os.path.join(images_dir, f"{prompt_id}.png")
        if not os.path.exists(image_path):
            print(f"[Warn] image not found for id={prompt_id}: {image_path}, skip.")
            continue

        print(f"[Score] id={prompt_id}, image={image_path}")

        try:
            ir_score = score_image_reward(image_reward_model, prompt, image_path)
        except Exception as e:
            print(f"  [Error] ImageReward failed for {prompt_id}: {e}")
            ir_score = float("nan")
            raise e

        try:
            clip_score = score_clip(clip_model, clip_processor, prompt, image_path, device=device)
        except Exception as e:
            print(f"  [Error] CLIP failed for {prompt_id}: {e}")
            clip_score = float("nan")
            raise e

        try:
            hps_score = score_hps(prompt, image_path)
        except Exception as e:
            print(f"  [Error] HPSv2 failed for {prompt_id}: {e}")
            hps_score = float("nan")
            raise e

        rows.append(
            {
                "id": prompt_id,
                "prompt": prompt,
                "image_path": image_path,
                "image_reward": ir_score,
                "clip": clip_score,
                "hps": hps_score,
            }
        )

    # 寫出到 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[Done] Scores saved to {output_csv}")


def main():
    score_all_images(
        prompts_csv=os.path.join("prompts", "normal_prompts.csv"),
        images_dir=os.path.join("outputs", "normal"),
        output_csv="scores_normal.csv",
    )

    score_all_images(
        prompts_csv=os.path.join("prompts", "normal_prompts.csv"),
        images_dir=os.path.join("outputs", "modified"),
        output_csv="scores_modified.csv",
    )


if __name__ == "__main__":
    main()
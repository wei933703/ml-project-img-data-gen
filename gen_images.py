# generate_images.py
import os
import csv
from typing import List, Tuple

import torch
from diffusers import StableDiffusionPipeline


def get_device() -> torch.device:
    """
    自動偵測可用裝置：CUDA -> MPS -> CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sd15_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device: torch.device | None = None,
) -> StableDiffusionPipeline:
    """
    載入 Stable Diffusion 1.5 模型。
    """
    if device is None:
        device = get_device()

    # CPU 通常用 float32，GPU/MPS 用 float16 省記憶體
    # dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    dtype = torch.float16 if device.type in ("cuda") else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # 依需求啟用/關閉
    )
    pipe.to(device)

    # 一些省記憶體的小技巧
    if device.type == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()

    return pipe


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


def generate_images_from_prompts(
    pipe: StableDiffusionPipeline,
    prompts: List[Tuple[str, str]],
    output_dir: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = 140122,
    height: int = 512,
    width: int = 512,
) -> None:
    """
    給定 Stable Diffusion pipeline 與 (id, prompt) 清單，批次生成圖片。

    生成檔名格式：{id}.png
    """
    os.makedirs(output_dir, exist_ok=True)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    for prompt_id, prompt in prompts:
        out_path = os.path.join(output_dir, f"{prompt_id}.png")

        print(f"[SD1.5] Generating image for id={prompt_id}, prompt={prompt!r} -> {out_path}")
        if torch.cuda.is_available():
            with torch.autocast(str(pipe.device), enabled=(pipe.device.type != "cpu")):
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                ).images[0]
        else:
            with torch.no_grad():
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]

        image.save(out_path)
    
    print("[Done] Image generation completed.")


def main():
    HEIGHT = 512
    WIDTH = 512

    device = get_device()
    print(f"Using device: {device}")

    pipe = load_sd15_pipeline(device=device)

    normal_csv_path = os.path.join("prompts", "normal_prompts.csv")
    normal_output_dir = os.path.join("outputs", "normal")
    prompts = read_prompts_csv(normal_csv_path)
    generate_images_from_prompts(pipe, prompts, output_dir=normal_output_dir, height=HEIGHT, width=WIDTH)

    modified_csv_path = os.path.join("prompts", "modified_prompts.csv")
    modified_output_dir = os.path.join("outputs", "modified")
    prompts = read_prompts_csv(modified_csv_path)
    generate_images_from_prompts(pipe, prompts, output_dir=modified_output_dir, height=HEIGHT, width=WIDTH)


if __name__ == "__main__":
    main()
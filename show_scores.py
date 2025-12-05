import csv
from typing import List, Dict, Any


def read_scores_csv(csv_path: str = "scores.csv") -> List[Dict[str, Any]]:
    """
    從 scores.csv 讀取記錄，欄位預期為：
    id,prompt,image_path,image_reward,clip,hps
    """
    records: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 嘗試把分數轉成 float（若失敗就保持原字串）
            for key in ("image_reward", "clip", "hps"):
                try:
                    row[key] = float(row[key])
                except Exception:
                    pass
            records.append(row)
    return records


def display_scores_table(records: List[Dict[str, Any]]) -> None:
    """
    將 (id, prompt, image_reward, clip, hps) 統一顯示。
    這裡先用簡單的文字表格輸出，你也可以改成 pandas 或 rich table。
    """
    # 設定欄寬
    id_width = 6
    ir_width = 12
    clip_width = 12
    hps_width = 12
    prompt_width = 60  # 避免太長，可以再調整

    header = (
        f"{'id':<{id_width}} "
        f"{'ImageReward':>{ir_width}} "
        f"{'CLIP':>{clip_width}} "
        f"{'HPSv2':>{hps_width}} "
        f"{'prompt':<{prompt_width}}"
    )
    print(header)
    print("-" * len(header))

    for row in records:
        pid = str(row.get("id", ""))
        prompt = str(row.get("prompt", "")).replace("\n", " ")

        ir = row.get("image_reward", "")
        clip = row.get("clip", "")
        hps = row.get("hps", "")

        # 分數格式化
        def fmt_score(x):
            try:
                return f"{float(x):.4f}"
            except Exception:
                return "NaN"

        line = (
            f"{pid:<{id_width}} "
            f"{fmt_score(ir):>{ir_width}} "
            f"{fmt_score(clip):>{clip_width}} "
            f"{fmt_score(hps):>{hps_width}} "
            f"{prompt[:prompt_width]:<{prompt_width}}"
        )
        print(line)


def main():
    records = read_scores_csv("scores.csv")
    display_scores_table(records)


if __name__ == "__main__":
    main()
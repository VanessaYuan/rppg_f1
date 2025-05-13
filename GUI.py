import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
from datetime import datetime
import subprocess
from tkinter import ttk
import Preprocess as pp


def browse_folder():
    foldername = filedialog.askdirectory(title="選擇健康資料資料夾")
    if foldername:
        file_label.config(text=f"Selected Folder: {foldername}")

def execute_action():
    root_folder = file_label.cget("text").replace("Selected Folder: ", "")
    if not os.path.isdir(root_folder):
        messagebox.showwarning("錯誤", "請選擇健康資料資料夾")
        return

    all_data = []

    # 🔢 預估總檔案數量
    total_files = 0
    for case_name in os.listdir(root_folder):
        case_path = os.path.join(root_folder, case_name)
        if not os.path.isdir(case_path):
            continue

        for region_name in os.listdir(case_path):
            region_path = os.path.join(case_path, region_name)
            if not os.path.isdir(region_path):
                continue

            total_files += len([f for f in os.listdir(region_path) if f.endswith(".txt")])

    progress_bar["maximum"] = total_files
    progress_bar["value"] = 0
    root.update()

    # 🚀 開始處理資料夾
    processed = 0
    for case_name in os.listdir(root_folder):
        case_path = os.path.join(root_folder, case_name)
        if not os.path.isdir(case_path):
            continue

        for region_name in os.listdir(case_path):
            region_path = os.path.join(case_path, region_name)
            if not os.path.isdir(region_path):
                continue

            for filename in os.listdir(region_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(region_path, filename)

                    try:
                        features = pp.main_all_features(file_path)

                        color = "unknown"
                        for c in ["r", "g", "b"]:
                            if f"_{c}" in filename:
                                color = c
                                break

                        row = features + [f'="{case_name}', region_name, color]
                        all_data.append(row)

                    except Exception as e:
                        print(f"處理失敗: {file_path}，錯誤訊息: {e}")

                    processed += 1
                    progress_bar["value"] = processed
                    root.update()

    feature_names = [
        "sdnn", "rmssd",
        "tRatio_1", "tRatio_2",
        "hRatio_1", "hRatio_2",
        "aRatio_1", "aRatio_2",
        "SampEn", "ApEn",
        "nlf", "nhf", "lf_hf_ratio",
        "heart_rate_bpm",
        "record_folder", "ROI", "color"
    ]

    df = pd.DataFrame(all_data, columns=feature_names)

    output_folder = 'C:\\output_features\\'
    os.makedirs(output_folder, exist_ok=True)

    output_filename = "Features_all.csv"
    full_path = os.path.join(output_folder, output_filename)

    file_exists = os.path.exists(full_path)
    df.to_csv(full_path, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')

    messagebox.showinfo("儲存成功", f"特徵已儲存至：\n{full_path}")
    subprocess.Popen(f'explorer "{output_folder}"')
    print(f"總共輸出了 {len(all_data)} 筆資料")

    progress_bar["value"] = total_files
    root.update()


"""介面"""
root = tk.Tk()
root.title("RPPG前處理工具_4.0_健康資料版本")
root.geometry("500x600")

browse_button = tk.Button(root, text="選取『健康資料』資料夾", command=browse_folder)
browse_button.pack(pady=5)

file_label = tk.Label(root, text="Selected Folder: None", wraplength=400, anchor="w")
file_label.pack(pady=5)

# Time Window（目前未使用，但可以保留作為延伸）
time_window_label = tk.Label(root, text="Time Window:")
time_window_label.pack(pady=(10, 0))
time_window_entry = tk.Entry(root)
time_window_entry.pack(pady=5)
time_window_entry.insert(0, "100")

# 進度條
progress_label = tk.Label(root, text="處理進度：")
progress_label.pack()

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

execute_button = tk.Button(root, text="執行", command=execute_action)
execute_button.pack(pady=20)

root.mainloop()

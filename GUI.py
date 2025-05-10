import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os
from datetime import datetime
import subprocess
from tkinter import ttk
import Preprocess as pp


def browse_file_1():
    foldername = filedialog.askdirectory(title="選擇資料夾")
    if foldername:
        file_label_1.config(text=f"Selected Folder: {foldername}")

"""4版"""
def execute_action():
    case_path = file_label_1.cget("text").replace("Selected Folder: ", "")
    if not os.path.isdir(case_path):
        messagebox.showwarning("錯誤", "請選擇一個資料夾")
        return
    
    file_name = os.path.basename(case_path)  # e.g., "15-3"
    all_data = []

    # 🔢 預先計算總檔案數（為進度條準備）
    total_files = 0
    for region_name in os.listdir(case_path):
        region_path  = os.path.join(case_path, region_name)
        if not os.path.isdir(region_path):
            continue

        total_files += len([f for f in os.listdir(region_path) if f.endswith(".txt")])

    progress_bar["maximum"] = total_files
    progress_bar["value"] = 0
    root.update()  # 更新 GUI 畫面

    # 🚀 開始處理檔案
    processed = 0
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

                    row = features + [f'="{file_name}', region_name, color]
                    all_data.append(row)

                except Exception as e:
                    print(f"處理失敗: {file_path}，錯誤訊息: {e}")

                # 更新進度條
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

    # now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_filename = f"features_all_{now}.csv"
    output_filename = f"Features_all.csv"
    full_path = os.path.join(output_folder, output_filename)

    # 如果檔案已存在，就以 append 模式寫入；否則就建立新檔案
    file_exists = os.path.exists(full_path)         
    df.to_csv(full_path, mode='a', header=not file_exists, index=False)

    messagebox.showinfo("儲存成功", f"特徵已儲存至：\n{full_path}")
    subprocess.Popen(f'explorer "{output_folder}"')

    # 處理完成，歸零進度條（或滿格顯示完成）
    progress_bar["value"] = total_files
    root.update()


"""介面"""
# 建立主視窗
root = tk.Tk()
root.title("RPPG前處理工具_1.0")
root.geometry("500x600")

# 檔案 1 選取區
browse_button_1 = tk.Button(root, text="選取小資料夾", command=browse_file_1)
browse_button_1.pack(pady=5)
file_label_1 = tk.Label(root, text="Selected File: None", wraplength=300, anchor="w")
file_label_1.pack(pady=5)

# Time Window 輸入框
time_window_label = tk.Label(root, text="Time Window:")
time_window_label.pack(pady=(10, 0))
time_window_entry = tk.Entry(root)
time_window_entry.pack(pady=5)
time_window_entry.insert(0, "100")

# 進度條 Label
progress_label = tk.Label(root, text="處理進度：")
progress_label.pack()

# 進度條本體
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)


# # 第幾個 Time Window 輸入框
# time_window_index_label = tk.Label(root, text="第幾個 Time Window:")
# time_window_index_label.pack(pady=(10, 0))
# time_window_index_entry = tk.Entry(root)
# time_window_index_entry.pack(pady=5)
# time_window_index_entry.insert(0, "1")

# # 區間 From 輸入框
# 區間_from = tk.Label(root, text="區間_from:")
# 區間_from.pack(pady=(10, 0))
# 區間_from_entry = tk.Entry(root)
# 區間_from_entry.pack(pady=5)
# 區間_from_entry.insert(0, "80")

# # 區間 To 輸入框
# 區間_to = tk.Label(root, text="區間_to:")
# 區間_to.pack(pady=(10, 0))
# 區間_to_entry = tk.Entry(root)
# 區間_to_entry.pack(pady=5)
# 區間_to_entry.insert(0, "300")

# # 產生 Excel 檔案 Checkbutton
# check_var = tk.IntVar()
# check_var_label = tk.Label(root, text="產生 Excel 檔")
# check_var_label.pack()
# check_box = tk.Checkbutton(root, variable=check_var)
# check_box.pack()

# 執行按鈕
execute_button = tk.Button(root, text="執行", command=execute_action)
execute_button.pack(pady=20)

# 啟動主迴圈
root.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import subprocess
from tkinter import ttk
import Preprocess as pp


def browse_folder():
    foldername = filedialog.askdirectory(title="選擇大資料資料夾")
    if foldername:
        file_label.config(text=f"Selected Folder: {foldername}")


def extract_metadata(filepath):
    parts = filepath.split(os.sep)
    raw_case = parts[-3]  # e.g., '15-3'
    roi = parts[-2]   # e.g., 'cheeck'
    filename = parts[-1]
    color = filename.split('_')[-1].split('.')[0]  # 'r', 'g', or 'b'
    
    case = f"case_{raw_case}"  # 防止被當成日期
    return case, roi, color

"""window size 選擇"""
# def extract_segment_features(signal, window_size):
#     segments = []
#     for start in range(0, len(signal), window_size):
#         end = start + window_size
#         if end > len(signal):
#             break
#         segment = signal[start:end]
#         segments.append(segment)
#     return segments

"""window size 選擇 + 決定overlap多少點"""
def extract_segment_features(signal, window_size, overlap_points):
    segments = []
    step_size = window_size - overlap_points  # 每次滑動多少點（要保證 > 0）

    if step_size <= 0:
        raise ValueError("Overlap 太大，必須小於 window_size")

    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        segment = signal[start:end]
        segments.append(segment)

    return segments

def process_file(filepath, window_size, overlap_points):
    signal = pp.read_from_file(filepath)

    time_processed = pp.preProcessing_timeDomain(signal)
    freq_processed = pp.preProcessing_freqDomain(signal)

    time_segments = extract_segment_features(time_processed, window_size, overlap_points)
    freq_segments = extract_segment_features(freq_processed, window_size, overlap_points)

    case, roi, color = extract_metadata(filepath)

    rows = []
    for i, (time_seg, freq_seg) in enumerate(zip(time_segments, freq_segments)):
        time_features = pp.time_features_cal(time_seg)
        freq_features = pp.freq_features_cal(freq_seg)
        row = {
            'case': case,
            'roi': roi,
            'color': color,
            'segment': f'seg{i}',
        }
        row.update({f"time_{k}": v for k, v in time_features.items()})
        row.update({f"freq_{k}": v for k, v in freq_features.items()})
        rows.append(row)
    return rows

def execute_action():
    folder_path = file_label.cget("text").replace("Selected Folder: ", "")
    if not os.path.isdir(folder_path):
        messagebox.showerror("錯誤", "請選擇正確的資料夾路徑")
        return

    window_size = int(segment_window_entry.get())
    overlap_points = int(overlap_entry.get())
    all_rows = []

    # 準備輸出資料夾
    output_folder = 'C:\\output_features\\'
    os.makedirs(output_folder, exist_ok=True)

    # 收集所有檔案路徑以便計算進度
    all_txt_files = []
    for dirpath, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                all_txt_files.append(os.path.join(dirpath, file))

    total_files = len(all_txt_files)
    if total_files == 0:
        messagebox.showwarning("結果", "找不到任何 .txt 檔案")
        return

    progress_bar["maximum"] = total_files
    progress_bar["value"] = 0
    root.update_idletasks()

    # 處理每個檔案
    for idx, filepath in enumerate(all_txt_files):
        try:
            rows = process_file(filepath, window_size, overlap_points)
            all_rows.extend(rows)
        except Exception as e:
            print(f"處理檔案失敗：{filepath}，錯誤訊息：{e}")
        
        progress_bar["value"] = idx + 1
        root.update_idletasks()

    if not all_rows:
        messagebox.showwarning("結果", "沒有成功處理任何檔案。")
        return

    # ===== 寬表格（WIDE）格式輸出 =====
    df = pd.DataFrame(all_rows)

    id_vars = ['case', 'segment']
    value_vars = [col for col in df.columns if col.startswith('time_') or col.startswith('freq_')]

    new_rows = []
    for _, row in df.iterrows():
        base_info = {'case': row['case'], 'segment': row['segment']}
        roi = row['roi']
        color = row['color']
        new_row = base_info.copy()
        for col in value_vars:
            new_name = f"{col[5:]}_{roi}_{color}"  # 移除 time_/freq_ 前綴
            new_row[new_name] = row[col]
        new_rows.append(new_row)

    df_wide = pd.DataFrame(new_rows)
    df_final = df_wide.groupby(['case', 'segment'], as_index=False).first()

    timestamp = datetime.now().strftime('%Y%m%d')  # 加入時間戳記
    output_filename_wide = f"Features_by_segment_WIDE_{timestamp}.csv"
    output_path = os.path.join(output_folder, output_filename_wide)
    
    # 如果檔案不存在，就加上欄位標題；若已存在，就只寫內容（不重複寫欄位）
    write_header = not os.path.exists(output_path)
    df_final.to_csv(output_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
    

    messagebox.showinfo("完成", f"特徵已儲存至：\n{output_folder}")
    subprocess.Popen(f'explorer "{output_folder}"')
    print(f"window:", window_size)
    print(f"overlap:",overlap_points)
    print(f"本次輸出 {len(df_final)} 筆 case-segment 特徵（寬表格，每筆一列）")


"""介面"""
root = tk.Tk()
root.title("RPPG前處理工具_4.0_健康資料版本")
root.geometry("500x500")

browse_button = tk.Button(root, text="選取大資料夾", command=browse_folder)
browse_button.pack(pady=5)

file_label = tk.Label(root, text="Selected Folder: None", wraplength=400, anchor="w")
file_label.pack(pady=5)

# Time Window
segment_window_label = tk.Label(root, text="Time Window Size(points):")
segment_window_label.pack(pady=(10, 0))
segment_window_entry = tk.Entry(root)
segment_window_entry.pack(pady=5)
segment_window_entry.insert(0, "180")  # 預設為 180 點

# Overlap Entry
overlap_label = tk.Label(root, text="Overlap Points (u):")
overlap_label.pack()
overlap_entry = tk.Entry(root)
overlap_entry.pack()
overlap_entry.insert(0, "50")  # 預設重疊 50 點

# 進度條
progress_label = tk.Label(root, text="處理進度：")
progress_label.pack()

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

# 執行按鈕
execute_button = tk.Button(root, text="執行", command=execute_action)
execute_button.pack(pady=20)

root.mainloop()

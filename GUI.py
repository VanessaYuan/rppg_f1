import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
from datetime import datetime
import subprocess
from tkinter import ttk
import Preprocess as pp


def browse_folder():
    foldername = filedialog.askdirectory(title="選擇健康資料資料夾")
    if foldername:
        file_label.config(text=f"Selected Folder: {foldername}")

"""讀取檔案(3版)"""
def read_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    """原始數據長度"""
    total_original = len(lines)
    print(f"\n--原始數據點數: {total_original}")

    # if total_original < 1000:  
    #     print("數據長度不足")
    #     return np.array([])

    # """去掉最前面 100 個點和最後面 50 個點"""
    # lines = lines[100:-150]
    # total_after_trim = len(lines)
    # print(f"--去頭 100、去尾 50 後剩餘: {total_after_trim}")

    # """只保留第 200 到 1000 點"""
    # lines = lines[200:1001]
    # total_after_trim = len(lines)
    # print(f"--篩選後剩餘: {total_after_trim}")

    """解析數據"""
    parsed_data = []  # 存儲 (value, timestamp)
    values = []  # 只存數值部分
    first_timestamp = None  # 記錄第一個時間戳

    for line in lines:
        data_tpm = line.strip().split(",")
        try:
            value = float(data_tpm[1])  # 解析數值
            timestamp = datetime.strptime(data_tpm[0], "%H:%M:%S.%f")  # 解析時間
            
            if first_timestamp is None:
                first_timestamp = timestamp  # 記錄第一個時間
            
            # 計算與第一個時間的時間差（以秒為單位）
            time_second = (timestamp - first_timestamp).total_seconds()

            parsed_data.append((value, time_second))  # 存數據
            values.append(value)  # 存數值
        except Exception as e:
            print(f"數據解析錯誤: {line.strip()}，錯誤: {e}")

    """計算平均值與標準差"""
    mean_val = np.mean(values)
    std_val = np.std(values)
    lower_bound = mean_val - 2 * std_val
    upper_bound = mean_val + 2 * std_val
    print(f"--離群值範圍: 小於 {lower_bound:.2f} 或 大於 {upper_bound:.2f} 的數據將被排除")

    """篩選數據"""
    listTemp = [(value, time_second) for value, time_second in parsed_data if lower_bound <= value <= upper_bound]
    
    # """重新編排 y 軸"""
    # listTemp = [(value, new_y, time_second) for new_y, (value, time_second) in enumerate(filtered_data)]

    """計算離群值數量"""
    outlier_count = len(values) - len(listTemp)
    total_after_outlier_removal = len(listTemp)
    print(f"--離群值數量: {outlier_count}")
    print(f"--扣除離群值後剩餘: {total_after_outlier_removal}")
    
    # print(f"輸出格式為\n{listTemp}")  ## list

    # """繪製訊號圖"""
    # plt.figure(figsize=(12, 5))

    # # 從list中提取值
    # filtered_values = [val for val, _ in listTemp]
    # time_second_index = [time_second for _, time_second in listTemp]

    # plt.plot(time_second_index, filtered_values, color="b", markersize=3, linestyle="-", label="Signal")

    # # 添加均值虛線
    # plt.axhline(mean_val, color="r", linestyle="--", label="Mean")

    # # 標記離群值範圍
    # plt.fill_between(time_second_index, lower_bound, upper_bound, color="gray", alpha=0.2, label="Mean ± 2STD")

    # plt.xlabel("Time (seconds)")
    # plt.ylabel("Value")
    # plt.title("Filtered Signal")
    # plt.legend()
    # plt.grid(True)

    # plt.show()

    return np.array(listTemp)  # 回傳整理後的數據

def extract_metadata(filepath):
    parts = filepath.split(os.sep)
    case = parts[-3]  # e.g., '15-3'
    roi = parts[-2]   # e.g., 'cheeck'
    filename = parts[-1]
    color = filename.split('_')[-1].split('.')[0]  # 'r', 'g', or 'b'
    return case, roi, color

def extract_segment_features(signal, window_size):
    segments = []
    for start in range(0, len(signal), window_size):
        end = start + window_size
        if end > len(signal):
            break
        segment = signal[start:end]
        segments.append(segment)
    return segments

def process_file(filepath, window_size):
    signal = read_from_file(filepath)

    time_processed = pp.preProcessing_timeDomain(signal)
    freq_processed = pp.preProcessing_freqDomain(signal)

    time_segments = extract_segment_features(time_processed, window_size)
    freq_segments = extract_segment_features(freq_processed, window_size)

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
        row.update({f"time_{k}": v for k, v in freq_features.items()})
        rows.append(row)
    return rows

def execute_action():
    folder_path = file_label.cget("text").replace("Selected Folder: ", "")
    if not os.path.isdir(folder_path):
        messagebox.showerror("錯誤", "請選擇正確的資料夾路徑")
        return

    window_size = int(segment_window_entry.get())
    all_rows = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                try:
                    rows = process_file(filepath, window_size)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"處理檔案失敗：{filepath}，錯誤訊息：{e}")

    if not all_rows:
        messagebox.showwarning("結果", "沒有成功處理任何檔案。")
        return

    # 整理為寬表格格式（欄位包含 roi + color）
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

    output_folder = 'C:\\output_features\\'
    os.makedirs(output_folder, exist_ok=True)
    output_filename = "Features_by_segment_WIDE.csv"
    df_final.to_csv(os.path.join(output_folder, output_filename), index=False, encoding='utf-8-sig')

    messagebox.showinfo("完成", f"特徵已儲存至：\n{output_folder}")
    subprocess.Popen(f'explorer "{output_folder}"')

    print(f"總共輸出了 {len(df_final)} 筆 case-segment 特徵（每筆一列，欄位包含 ROI + 通道）")


    df = pd.DataFrame(all_rows)
    output_folder = 'C:\\output_features\\'
    os.makedirs(output_folder, exist_ok=True)
    output_filename = "Features_by_segment.csv"
    df.to_csv(os.path.join(output_folder, output_filename), index=False, encoding='utf-8-sig')

    messagebox.showinfo("完成", f"特徵已儲存至：\n{output_folder}")
    subprocess.Popen(f'explorer "{output_folder}"')

    print(f"總共輸出了 {len(all_rows)} 段 segment 特徵（每段一列）")



"""介面"""
root = tk.Tk()
root.title("RPPG前處理工具_4.0_健康資料版本")
root.geometry("500x600")

browse_button = tk.Button(root, text="選取『健康資料』資料夾", command=browse_folder)
browse_button.pack(pady=5)

file_label = tk.Label(root, text="Selected Folder: None", wraplength=400, anchor="w")
file_label.pack(pady=5)

# Time Window（目前未使用，但可以保留作為延伸）
segment_window_label = tk.Label(root, text="Time Window Size(points):")
segment_window_label.pack(pady=(10, 0))
segment_window_entry = tk.Entry(root)
segment_window_entry.pack(pady=5)
segment_window_entry.insert(0, "300")  # 預設為 300 點

# 進度條
progress_label = tk.Label(root, text="處理進度：")
progress_label.pack()

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

execute_button = tk.Button(root, text="執行", command=execute_action)
execute_button.pack(pady=20)

root.mainloop()

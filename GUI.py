import tkinter as tk
import Preprocess as pp
from tkinter import filedialog, messagebox
import pandas as pd
import os
from datetime import datetime
import subprocess

def browse_file_1():
    filename = filedialog.askopenfilename(title="Select File 1")
    if filename:
        file_label_1.config(text=filename)


def execute_action():
    # # 如果未選取檔案，跳出提示
    # if file_label_1.cget("text") == "Selected File: None" or file_label_2.cget("text") == "Selected File: None":
    #     messagebox.showinfo("執行結果", "請先選取兩個檔案")
    #     return

    time_window = time_window_entry.get()
    # time_window_index = time_window_index_entry.get()
    
    file_1 = file_label_1.cget("text").replace("Selected File: ", "")

    
    # pp.main_all_features(file_1)
    # pp.main_all_features(file_2)

    # 特徵整合
    features_1 = pp.main_all_features(file_1)  # list

    # 合併所有特徵
    all_features_all = [features_1]

    # 欄位名稱（依你的特徵順序自訂）
    feature_names = [
        "sdnn", "rmssd", 
        "tRatio_1", "tRatio_2",
        "hRatio_1", "hRatio_2",
        "aRatio_1", "aRatio_2",
        "SampEn", "ApEn",
        "nlf", "nhf", "lf_hf_ratio", 
        "heart_rate_bpm1", "heart_rate_bpm2", "heart_rate_bpm3",
        
    ]  # 如果特徵數不同，要補上正確順序！

    # 轉成 DataFrame
    df = pd.DataFrame(all_features_all, columns=feature_names)

    # 檢查是否要產生 Excel/CSV
    # if check_var.get():
    #     save_path = filedialog.asksaveasfilename(
    #         defaultextension=".csv",
    #         filetypes=[("CSV files", "*.csv")],
    #         title="儲存特徵為 CSV"
    #     )
    #     if save_path:
    #         df.to_csv(save_path, index=False)
    #         messagebox.showinfo("成功", f"特徵已儲存到：\n{save_path}")
    #     else:
    #         messagebox.showwarning("取消儲存", "未選擇儲存路徑")
    
        # 是否儲存
    if check_var.get():
        # 建立資料夾（如果不存在）
        output_folder = 'C:\\output_features\\'
        os.makedirs(output_folder, exist_ok=True)

        # 固定檔案名稱
        output_filename = "features_log.csv"
        full_path = os.path.join(output_folder, output_filename)

        # 如果檔案已存在就 append，不寫欄位名稱
        if os.path.exists(full_path):
            df.to_csv(full_path, mode='a', header=False, index=False)
        else:
            df.to_csv(full_path, index=False)

        messagebox.showinfo("儲存成功", f"特徵已附加至：\n{full_path}")

        # 自動開啟資料夾
        subprocess.Popen(f'explorer "{output_folder}"')

    
    """每次都輸出一個csv"""
    # # 是否儲存
    # if check_var.get():
    #     # 📁 建立輸出資料夾（若不存在）
    #     output_folder = 'C:\\output_features\\'
    #     os.makedirs(output_folder, exist_ok=True)

    #     # 📝 自動產生檔案名稱（可改為自訂格式）
    #     now = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     output_filename = f"features_{now}.csv"
        
    #     full_path = os.path.join(output_folder, output_filename)

    #     # 儲存 CSV
    #     df.to_csv(full_path, index=False)

    #     messagebox.showinfo("儲存成功", f"特徵已儲存至：\n{full_path}")

    #     subprocess.Popen(f'explorer "{output_folder}"')

    # pp.all_features(timeFeatures_list, fqFeatures_list)

    # pp.preProcessing(
    #     int(time_window),
    #     # int(time_window_index),
    #     file_label_1.cget("text"),
    #     # int(區間_from_entry.get()),
    #     # int(區間_to_entry.get())
    # )
    # pp.preProcessing(
    #     int(time_window),
    #     # int(time_window_index),
    #     file_label_2.cget("text"),
    #     # int(區間_from_entry.get()),
    #     # int(區間_to_entry.get())
    # )

    # 執行按鈕動作時的提示框
    # result = f"Time Window: {time_window}\n"
    # result += f"Time Window Index: {time_window_index}\n"
    # result += f"File 1: {file_label_1.cget('text')}\n"
    # result += f"File 2: {file_label_2.cget('text')}\n"
    
    # messagebox.showinfo("執行結果", result)

"""介面"""
# 建立主視窗
root = tk.Tk()
root.title("RPPG前處理工具_1.0")
root.geometry("500x600")

# 檔案 1 選取區
browse_button_1 = tk.Button(root, text="選取檔案 1", command=browse_file_1)
browse_button_1.pack(pady=5)
file_label_1 = tk.Label(root, text="Selected File: None", wraplength=300, anchor="w")
file_label_1.pack(pady=5)

# Time Window 輸入框
time_window_label = tk.Label(root, text="Time Window:")
time_window_label.pack(pady=(10, 0))
time_window_entry = tk.Entry(root)
time_window_entry.pack(pady=5)
time_window_entry.insert(0, "100")

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

# 產生 Excel 檔案 Checkbutton
check_var = tk.IntVar()
check_var_label = tk.Label(root, text="產生 Excel 檔")
check_var_label.pack()
check_box = tk.Checkbutton(root, variable=check_var)
check_box.pack()

# 執行按鈕
execute_button = tk.Button(root, text="執行", command=execute_action)
execute_button.pack(pady=20)

# 啟動主迴圈
root.mainloop()

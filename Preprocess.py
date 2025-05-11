import datetime
import statistics
from turtle import pd
import cv2
import numpy as np
from imutils import face_utils
import imutils
import os
import time
from datetime import datetime
from scipy import signal, sparse
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from BaselineRemoval import BaselineRemoval
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
import traceback

import antropy as ant

from scipy.spatial import KDTree
from datetime import datetime
from scipy.signal import find_peaks

from scipy.optimize import linear_sum_assignment

#PTT
peaks_list = []

# R = (AC/DC)/(AC/DC)
acdc_list = []

"""圖形顯示"""
def plot_data(listTemp, title):
    plt.figure(figsize=(12, 5))

    # 擷取數據
    filtered_values = [val for val, _ in listTemp]  # Y 軸的數值
    # y軸_index = [y for _, y, _ in listTemp]  # X 軸索引
    time_second_index = [time_diff for _, time_diff in listTemp ]

    # 畫折線圖
    plt.plot(time_second_index, filtered_values, color="b", markersize=3, label="Signal")  

    # 設定標籤與標題
    plt.xlabel("Time(seconds)")
    plt.ylabel("Value")
    plt.title("Signal Plot after " + title)

    # 加入圖例 & 格線
    plt.legend()
    plt.grid(True)

    # 顯示圖表
    plt.show()

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


"""三角平滑化"""
def smoothTriangle(data, degree):
        triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
        smoothed=[]
    
        for i in range(degree, len(data) - degree * 2):
            point=data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point)/np.sum(triangle))

        # Handle boundaries
        smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])  ## list

        return np.array(smoothed)  ## 回傳 numpy array  

"""平滑化"""
def smooth_signal(listTemp,timeWindow):
        #平滑化
        #用.copy()複製一份listTemp_處理前，以便後續畫圖比較
        # listTemp_處理前 = listTemp.copy()
        
        # window_length 控制平滑範圍大小，數值越大，平滑效果越強。
        # polyorder 決定多項式的擬合複雜度，常用低階（如 2 或 3）。
        # mode 用於選擇邊界處理方式，默認 'interp'。
        listTemp[:,0] = scipy.signal.savgol_filter(listTemp[:,0].astype(float), timeWindow, 2)

        plot_data(listTemp[:,0])
        #將信號畫成圖型 #平滑化
        # 兩圖合併(listTemp_處理前,listTemp,"Smoothing")
        return listTemp

"""帶通濾波"""
def bandPass_filter(signal, fs, lowcut, highcut, order):
    """
    帶通濾波器：使用 Butterworth 濾波器
    :param signal: 需要處理的信號 (numpy array)
    :param fs: 採樣頻率 (預設 30Hz)
    :param lowcut: 低頻截止頻率 (預設 0.4Hz)
    :param highcut: 高頻截止頻率 (預設 3.5Hz)
    :param order: 濾波器階數 (預設 2)
    :return: 過濾後的信號 (numpy array)
    """

    nyquist = 0.5 * fs  # 奈奎斯特頻率
    low = lowcut / nyquist
    high = highcut / nyquist

    # 設計數位帶通濾波器 (analog=False)
    b, a = scipy.signal.butter(order, [low, high], btype='bandpass', analog=False)

    # 使用零相位濾波器
    y = scipy.signal.filtfilt(b, a, signal)

    return y  # 確保回傳的是 numpy array

"""帶通濾波"""
def butterworth_bandpass_filter(self, data, lowcut, highcut, fs, order=1):
        b, a = self.butterworth_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y
       
"""巴特濾波器"""
def butterworth_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band',analog='True')
    return b, a
    #order 是濾波器的階數，階數越大，濾波效果越好，但是計算量也會跟著變大。
    #所產生的濾波器參數 a 和 b 的長度，等於 order+1。
    #Wn 是正規化的截止頻率，介於 0 和 1 之間，當取樣頻率是 fs 時，所能處理的
    #最高頻率是 fs/2，所以如果實際的截止頻率是 f = 1000，那麼 Wn = f/(fs/2)。
    #function 是一個字串，function = 'low' 代表是低通濾波器，function = 'high' 代表是高通濾波。
    #fs=12,wn=f/(fs/2),如果截止頻率大於6,就高於正規化的截止頻率

def butterworth_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butterworth_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


"""學姊的 基線校正 (按週期分割並減去波谷值)"""
def baseline_correction(signal,window_size):
    corrected_signal = np.zeros_like(signal)
    # window_size = 100  # 假設每個週期約 100 點
    for i in range(0, len(signal), window_size):
        segment = signal[i:i + window_size]
        if len(segment) > 0:
            corrected_signal[i:i + window_size] = segment - np.min(segment)
    return corrected_signal


"""基線校正(每個時窗去做基線校正)"""
def baseline_removal(listTemp,polynomial_degree,window_size):
    # polynomial_degree = 2  # 只適用於 ModPoly 和 IModPoly
    # window_size = 50  # 設定每 50 個點執行一次基線校正

    for i in range(0, len(listTemp), window_size):
        sub_array = listTemp[i:i+window_size, 0]  # 取出當前的50個點
        if len(sub_array) < 2:  # 避免長度太短導致算法錯誤
            continue
        baseObj = BaselineRemoval(sub_array)
        listTemp[i:i+window_size, 0] = baseObj.ModPoly(polynomial_degree)  # 進行基線校正
    return listTemp


def baseline_removal2(listTemp, window_size):
    """基線校正(每個時窗去做基線校正，使用最小值法)"""
    
    for i in range(0, len(listTemp), window_size):
        segment = listTemp[i:i+window_size, 0]  # 取出當前窗口內的數據
        if len(segment) < 2:  # 避免長度太短導致錯誤
            continue

        min_value = np.min(segment)  # 計算當前窗口的最小值
        listTemp[i:i+window_size, 0] = segment - np(min_value)  # 減去最小值進行基線校正
    
    return listTemp


"""基線校準(进行AsLS基线校正)"""
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y) 
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2)) 
    D = lam * D.dot(D.T) 
    w = np.ones(L) 
    for i in range(niter): 
        W = diags(w, 0, shape=(L, L)) 
        Z = W + D 
        z = spsolve(Z, w*y) 
        w = p * (y > z) + (1-p) * (y < z) 
    return z
#基線校準(主要呼叫)
def baseline_correction(input_data): 
    corrected_spectra = np.zeros_like(input_data) 
    for i in range(input_data.shape[0]): 
        baseline_values = baseline_als(input_data[i, :]) 
        corrected_spectra[i, :] = input_data[i, :] - baseline_values 
    return corrected_spectra

"""基線校準(上一屆作法)"""
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    """
    Remove linear trend along axis from data.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng()
    >>> npoints = 1000
    >>> noise = rng.standard_normal(npoints)
    >>> x = 3 + 2*np.linspace(0, 1, npoints) + noise
    >>> (signal.detrend(x) - noise).max()
    0.06  # random

    """
    if type not in ['linear', 'l', 'constant', 'c']:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = np.asarray(data)
    dtype = data.dtype.char
    if dtype not in 'dfDF':
        dtype = 'd'
    if type in ['constant', 'c']:
        ret = data - np.mean(data, axis, keepdims=True)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.r_[0, bp, N]))
        if np.any(bp > N):
            raise ValueError("Breakpoints must be less than length "
                             "of data along given axis.")
        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdims = np.r_[axis, 0:axis, axis + 1:rnk]
        newdata = np.reshape(np.transpose(data, tuple(newdims)),
                             (N, _prod(dshape) // N))
        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)
        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2), dtype)
            A[:, 0] = np.cast[dtype](np.arange(1, Npts + 1) * 1.0 / Npts)
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - np.dot(A, coef)
        # Put data back in original shape.
        tdshape = np.take(dshape, newdims, 0)
        ret = np.reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = np.transpose(ret, tuple(olddims))
        return ret



"""標準化"""
def Z_ScoreNormalization(x,mu,sigma):
        x = (x - mu) / sigma;
        return x;



def fit_transform(X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        # if y is None:
        #     # fit method of arity 1 (unsupervised transformation)
        #     return fit(X, **fit_params).transform(X)
        # else:
        #     # fit method of arity 2 (supervised transformation)
        #     return fit(X, y, **fit_params).transform(X)

"""線性插值"""
def interpolated_signal(signal, time_data):

    listTemp_list = []
    L = len(signal) #訊號長度
    #  event_times = np.linspace(start, stop, num)
    event_times = np.linspace(time_data[0], time_data[-1], L)
    #  interpolated = np.interp(x_new, x_old, y_old)
    interpolated = np.interp(event_times, time_data, signal)
    # 漢明窗
    interpolated = np.hamming(L) * interpolated 

    # print("Interpolated Signal:", interpolated)
    # print("Event Times:", event_times)
    
    # 合併 interpolated 和 event_times 為一個 numpy 陣列
    result = np.column_stack((interpolated, event_times))  # 每行包含插值信號和時間
    # listTemp.append((interpolated,event_times))

    return result

"""歸一化"""
def normalization(listTemp):
    scaler = MinMaxScaler()
    # 使用MinMaxScaler對特徵進行擬合和轉換
    listTemp[:,0:1] = scaler.fit_transform(listTemp[:,0:1])

    return listTemp

"""計算R= (AC/DC)/(AC/DC)"""
def r_cal(listTemp):
    
    values, timestamps = zip(*listTemp)  
    values = np.array(values)
    timestamps = np.array(timestamps)
    
    dc = np.mean(values)
    ac = np.std(values)
    acdc = ac/dc
    acdc = round(acdc, 3)
    return acdc

"""(一階差分)找波峰"""
def finding_peaks(signal_data):
    signal_data = np.array(signal_data)
    values, times = signal_data[:, 0], signal_data[:, 1]
    
    # 計算一階差分
    diff_values = np.diff(values)
    
    # 找出波峰（正變負）和波谷（負變正）
    peaks = np.where((diff_values[:-1] > 0) & (diff_values[1:] <= 0))[0] + 1
    troughs = np.where((diff_values[:-1] < 0) & (diff_values[1:] >= 0))[0] + 1
    
    # # 繪製原始訊號
    # plt.figure(figsize=(10, 5))
    # plt.plot(times, values, label='Signal', color='blue')
    # plt.scatter(times[peaks], values[peaks], color='red', label='Peaks', marker='^')
    # plt.scatter(times[troughs], values[troughs], color='green', label='Troughs', marker='v')
    # # plt.scatter(times[peaks], values[peaks], 'ro',  label='Peaks')
    # # plt.scatter(times[troughs], values[troughs], 'bo', label='Troughs', )
    
    # # plt.axhline(y=threshold, color='purple', linestyle='--', label=f'Threshold = {threshold}')
    
    # plt.xlabel('Time (s)')
    # plt.ylabel('Signal Value')
    # plt.title('Signal with Peaks and Troughs')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    peaks_list.append(peaks)

    return peaks, peaks_list

"""計算ppi"""
def ppi_cal(peaks, sampling_rate = 12):
    ppi_values = []
    heart_rates = []

    # 計算 Ppi (相鄰波峰之間的時間差)
    for i in range(1, len(peaks)):
        # 將取樣點數轉換為毫秒
        ppi = (peaks[i] - peaks[i-1]) * (1000/sampling_rate)  # 轉換為毫秒
        ppi_values.append(ppi)

        # 根據 PPI 計算心率 (bpm)
        # hr = 60000 / ppi  # bpm = 60000 / PPI(ms)
        # heart_rates.append(hr)

    # 計算平均值
    avg_ppi = np.mean(ppi_values) if ppi_values else 0
    avg_hr = np.mean(heart_rates) if heart_rates else 0
    
    print("\n--共有", len(ppi_values), "個PPi")
    print("--平均PPi長度:", avg_ppi)
    print("--PPi值(當筆資料)：",ppi_values)
    # print("\n--平均心率 (HR): {:.2f} bpm".format(avg_hr))
    # print("-- HR 值 (當筆資料):", heart_rates)

    return ppi_values

"""計算SDNN、RMSSD"""
def sdnn_rmssd(ppi_values):    
    # 計算SDNN (標準差)
    sdnn = np.std(ppi_values)
    # 計算RMSSD (根均方差)
    rmssd = np.sqrt(np.mean(np.diff(ppi_values)**2))
    
    sdnn = round(sdnn, 3)
    rmssd = round(rmssd, 3)

    return sdnn, rmssd 

"""取所有peaks，算ptt"""
def ptt_t_cal(peaks_list, sampling_rate=15):
    """計算ptt(兩ROI波峰時間差)"""
    diff_list = []
    # ptt_t = []
    
    #判斷是否有兩個檔案
    if len(peaks_list) == 2:
        list_A = peaks_list[0]
        list_B = peaks_list[1]

    # 判斷哪一個比較長
        if len(list_A) > len(list_B):
            long_list = list_A
            short_list = list_B
        else:
            long_list = list_B
            short_list = list_A

        cost_matrix = np.zeros((len(long_list), len(short_list)))
        for i in range(len(long_list)):
            for j in range(len(short_list)):
                diff = abs(long_list[i] - short_list[j])
                cost_matrix[i][j] = diff if diff != 0 else np.inf  # 差值為0則設成無限大
    
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # print("\n最佳配對：")
        for i, j in zip(row_ind, col_ind):
            diff = abs(long_list[i] - short_list[j])
            # print(f"{long_list[i]} ↔ {short_list[j]}, 差值: {diff}")
            diff_list.append(diff)
        print("\n--所有PTT：\n", diff_list)

        # 計算 mean 和 std
        filtered_ptt_array = np.array(diff_list)
        mean_ptt = np.mean(filtered_ptt_array)
        std_ptt = np.std(filtered_ptt_array)

        # 保留落在 [mean - 2*std, mean + 2*std] 範圍內的值
        inlier_ptt = filtered_ptt_array[
            (filtered_ptt_array >= mean_ptt - 2 * std_ptt) &
            (filtered_ptt_array <= mean_ptt + 2 * std_ptt)
        ]

        print("過濾後的PTT（排除離群值）：\n", inlier_ptt)

        if len(inlier_ptt) > 0:
            avg_ptt_points = np.mean(inlier_ptt)
            avg_ptt_seconds = avg_ptt_points / sampling_rate
            ptt_t = round(avg_ptt_seconds,3)
            print(f"\n--平均波峰點數差：{ptt_t:.2f}")
            print(f"--平均PTT時間差（秒）: {ptt_t:.3f} 秒")
        else:
            print("\n所有資料都被視為離群值，無法計算平均")

        # unmatched_indices = set(range(len(long_list))) - set(row_ind)
        # unmatched_values = [long_list[i] for i in unmatched_indices]
        # print("\n應刪除的波峰索引（多餘的）:", unmatched_values)

    else:
        print("\n目前僅有一組波峰索引，尚無法進行配對比較，等待下一筆資料...")
    return ptt_t

"""找主/次波峰波谷"""
def finding_peaks_bp(signal_data, threshold = 0.1):
    signal_data = np.array(signal_data)
    values, times = signal_data[:, 0], signal_data[:, 1]
    
    # 計算一階差分
    diff_values = np.diff(values)
    
    # 找出波峰（正變負）和波谷（負變正）
    peaks = np.where((diff_values[:-1] > 0) & (diff_values[1:] <= 0))[0] + 1
    troughs = np.where((diff_values[:-1] < 0) & (diff_values[1:] >= 0))[0] + 1
    
    # # 繪製原始訊號
    # plt.figure(figsize=(10, 5))
    # plt.plot(times, values, label='Signal', color='blue')
    # plt.scatter(times[peaks], values[peaks], color='red', label='Peaks', marker='^')
    # plt.scatter(times[troughs], values[troughs], color='green', label='Troughs', marker='v')
    # # plt.scatter(times[peaks], values[peaks], 'ro',  label='Peaks')
    # # plt.scatter(times[troughs], values[troughs], 'bo', label='Troughs', )
    
    # plt.axhline(y=threshold, color='purple', linestyle='--', label=f'Threshold = {threshold}')
    
    # plt.xlabel('Time (s)')
    # plt.ylabel('Signal Value')
    # plt.title('Signal with Peaks and Troughs')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    # 篩選出 0.1 以下的波谷
    target_valleys = [v for v in troughs if values[v] < threshold]
    return values, times, peaks,troughs, target_valleys, signal_data


"""血壓特徵"""
def bp_features_cal(values, times, peaks,troughs, target_valleys, signal_data):
    
    result = []
    tr1, tr2 = [], []
    hr1, hr2 = [], []
    ar1, ar2 = [], []

    for i in range(len(target_valleys) - 1):
        start, end = target_valleys[i], target_valleys[i + 1]

        # 找出這個區間內的波峰與波谷
        peaks_in_range = [p for p in peaks if start < p < end]
        valleys_in_range = [v for v in troughs if start < v < end]

        """計算血壓特徵"""
        # 如果區間內有兩個波峰和一個波谷
        if len(peaks_in_range) == 2 and len(valleys_in_range) == 1:
            p1, p2 = peaks_in_range
            v1 = valleys_in_range[0]

            a,b,c,d,e = start, p1, v1, p2, end
            
            # 計算時間比
            time_a, time_c, time_e = times[a],times[c],times[e] 
            t = time_e - time_a
            t1 = time_c - time_a
            t2 = time_e - time_c

            if t == 0: continue  # 避免除以0
            timeRatio_1 = t1/t
            timeRatio_2 = t2/t

            # 計算振幅比
            amp_b, amp_c, amp_d = values[b], values[c], values[d]
            ampRatio_1 = amp_c/amp_b if amp_b != 0 else 0
            ampRatio_2 = amp_d/amp_b if amp_b != 0 else 0

            # 計算面積比
            area_abci = np.trapz(values[a:c+1],times[a:c+1])
            area_cdei = np.trapz(values[c:e+1],times[c:e+1])
            rect_afgi = 1*(times[c]-times[a])
            rect_ighe = 1*(times[e]-times[c])
            areaRatio_1 = area_abci/rect_afgi
            areaRatio_2 = area_cdei/rect_ighe

            # a_valleys.append(signal_data[start])
            # b_peaks.append(signal_data[p1])
            # c_valleys.append(signal_data[v1])
            # d_peaks.append(signal_data[p2])
            # e_valleys.append(signal_data[end])
            result.append((signal_data[start], signal_data[p1], signal_data[v1], signal_data[p2], signal_data[end]))# 篩選出 0.1 以下的波谷
            # timeRatio_all.append((timeRatio_1,timeRatio_2))
            # ampRatio_all.append((ampRatio_1, ampRatio_2))
            # areaRatio_all.append((areaRatio_1,areaRatio_2))
            tr1.append(timeRatio_1)
            tr2.append(timeRatio_2)
            hr1.append(ampRatio_1)
            hr2.append(ampRatio_2)
            ar1.append(areaRatio_1)
            ar2.append(areaRatio_2)
    # 如果沒有符合條件的特徵組合，避免回傳未定義變數
    if len(result) == 0:
        print("⚠️ 無法在目標波谷區間內找到完整的波形特徵 (2 peaks + 1 trough)")
        return result, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    tRatio_1 = round(np.mean(tr1), 3)
    tRatio_2 = round(np.mean(tr2), 3)
    hRatio_1 = round(np.mean(ampRatio_1), 3)
    hRatio_2 = round(np.mean(ampRatio_2), 3)
    aRatio_1 = round(np.mean(areaRatio_1), 3)
    aRatio_2 = round(np.mean(areaRatio_2), 3)
            
    
    return result, tRatio_1, tRatio_2, hRatio_1, hRatio_2, aRatio_1, aRatio_2


"""SampEn、ApEn計算"""
def embed_seq(x, m):
    """建立 m 維嵌入序列"""
    N = len(x)
    return np.array([x[i:i + m] for i in range(N - m + 1)])

def fast_sampen(x, m=2, r=None):
    """使用 KD-Tree 加速的 SampEn 計算"""
    x = np.array(x)
    if r is None:
        r = 0.2 * np.std(x)
    eps = 1e-10  # 防止 log(0)

    Xm = embed_seq(x, m)
    Xm1 = embed_seq(x, m + 1)

    def count_pairs(X):
        tree = KDTree(X, leafsize=16)
        count = 0
        for i, xi in enumerate(X):
            # 查找在距離 r 內的鄰居，排除自己
            neighbors = tree.query_ball_point(xi, r, p=np.inf)
            count += len(neighbors) - 1  # 不包含自己
        return count

    B = count_pairs(Xm)
    A = count_pairs(Xm1)

    if B == 0:
        return np.inf
    else:
        return -np.log((A + eps) / (B + eps))

def fast_apen(x, m=2, r=None):
    """使用 KD-Tree 加速的 ApEn 計算"""
    x = np.array(x)
    if r is None:
        r = 0.2 * np.std(x)
    eps = 1e-10

    def phi(X):
        N = len(X)
        tree = KDTree(X, leafsize=16)
        C = np.zeros(N)
        for i, xi in enumerate(X):
            neighbors = tree.query_ball_point(xi, r, p=np.inf)
            C[i] = (len(neighbors)) / N  # 包含自己
        C = np.where(C == 0, eps, C)
        return np.sum(np.log(C)) / N

    Xm = embed_seq(x, m)
    Xm1 = embed_seq(x, m + 1)

    return abs(phi(Xm) - phi(Xm1))
# # 計算距離：max norm（Chebyshev距離）
# def _max_dist(x_i, x_j):
#     return np.max(np.abs(x_i - x_j))

# # 建立 m 維向量序列
# def _embed_seq(x, m):
#     N = len(x)
#     return np.array([x[i:i + m] for i in range(N - m + 1)])

# # Sample Entropy（SampEn）
# def sampen_cal(x, m, r):
#     N = len(x)
#     x = np.array(x)
#     Xm = _embed_seq(x, m)
#     Xm1 = _embed_seq(x, m + 1)
#     eps = 1e-10 #epsilon(最小常數) = 1e-10，避免分母為0

#     def _count_matches(X):
#         N_temp = len(X)
#         count = 0
#         for i in range(N_temp):
#             for j in range(i + 1, N_temp): # 不與自己比
#                 if _max_dist(X[i], X[j]) <= r:
#                     count += 1
#         return count

#     B = _count_matches(Xm)
#     A = _count_matches(Xm1)

#     if B == 0 or A == 0:
#         return np.inf
#     else:
#         return -np.log( (A+ eps) /(B + eps))

# # Approximate Entropy（ApEn）
# def apen_cal(x, m, r):
#     N = len(x)
#     x = np.array(x)
#     Xm = _embed_seq(x, m)
#     Xm1 = _embed_seq(x, m + 1)
#     eps = 1e-10 #epsilon(最小常數) = 1e-10，避免 log(0)

#     def _phi(X):
#         N_temp = len(X)
#         C = np.zeros(N_temp)
#         for i in range(N_temp):
#             dist = np.array([_max_dist(X[i], X[j]) for j in range(N_temp)])
#             count = np.sum(dist <= r)
#             C[i] = count / N_temp if count > 0 else eps  # epsilon(最小常數) = 1e-10，避免 log(0)
#         return np.sum(np.log(C)) / N_temp

#     return _phi(Xm) - _phi(Xm1)


"""傅立葉+LF、HF計算"""
def fft_cal(listTemp,fps):

    values, timestamps = zip(*listTemp)  
    values = np.array(values)
    timestamps = np.array(timestamps)

    # 計算取樣率
    # duration = timestamps[-1] - timestamps[0]
    # fps = len(timestamps) / duration
    # print(f"\n Sampling rate = {fps:.2f} Hz")

    n = len(values)
    raw2 = np.fft.rfft(values)
    fft_freq = float(fps) / n * np.arange(n / 2 + 1)
    fft_power = np.abs(raw2)**2

    # # 畫圖
    # plt.figure(figsize=(10, 4))
    # plt.plot(fft_freq, fft_power, color='blue')
    # plt.title("Frequency Domain (FFT)-LF/HF")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.xlim(0, 1)  # 限制頻率範圍在 0~3 Hz
    # plt.tight_layout()
    # plt.show()

    return fft_freq, fft_power

"""LF、HF計算"""
def lfhf_cal(fft_freq, fft_power):
    # 定義各頻段範圍
    vlf_mask = (fft_freq >= 0.003) & (fft_freq < 0.04)
    lf_mask = (fft_freq >= 0.04) & (fft_freq < 0.15)
    hf_mask = (fft_freq >= 0.15) & (fft_freq < 0.4)
    # 總功率 TP（0.003~0.4 Hz）
    tp_mask = (fft_freq >= 0.003) & (fft_freq < 0.4)

    # 各頻段功率
    vlf_power = np.trapz(fft_power[vlf_mask], fft_freq[vlf_mask])
    lf_power = np.trapz(fft_power[lf_mask], fft_freq[lf_mask])
    hf_power = np.trapz(fft_power[hf_mask], fft_freq[hf_mask])
    # 總功率 TP（0.003~0.4 Hz）
    total_power = np.trapz(fft_power[tp_mask], fft_freq[tp_mask])
    
    # 標準化 nLF 與 nHF（避免除以零）
    # 分母
    denominator = total_power - vlf_power if total_power > vlf_power else 1e-8
    nlf = lf_power / denominator
    nhf = hf_power / denominator

    # LF/HF 比值
    lf_hf_ratio = nlf / nhf if nhf != 0 else np.inf

    nlf = round(nlf, 3)
    nhf = round(nhf, 3)
    lf_hf_ratio = round(lf_hf_ratio, 3)

    # 印出結果
    # print(f"VLF Power: {vlf_power:.3f}")
    # print(f"LF Power: {lf_power:.3f}")
    # print(f"HF Power: {hf_power:.3f}")
    # print(f"Total Power (TP): {total_power:.3f}")
    print(f"Normalized LF (nLF): {nlf:.3f}")
    print(f"Normalized HF (nHF): {nhf:.3f}")
    print(f"LF/HF Ratio: {lf_hf_ratio:.3f}")

    return nlf, nhf, lf_hf_ratio
    
    
"""傅立葉+HR計算"""
def hr_cal(fft_freq,fft_power):

    # values, timestamps = zip(*listTemp)  
    # values = np.array(values)
    # timestamps = np.array(timestamps)

    # # # 計算取樣率
    # # duration = timestamps[-1] - timestamps[0]
    # # fps = len(timestamps) / duration
    # # print(f"\n自動推算 fps: {fps:.2f} Hz")

    # n = len(values)
    # raw2 = np.fft.rfft(values*100) # 放大振幅，不影響頻率
    # # fft_freq = float(fps) / n * np.arange(n / 2 + 1) # 手動算頻率軸(x軸)
    # fft_freq = np.fft.rfftfreq(n, d=1/fps) # 正頻率對應的頻率軸
    # fft_power = np.abs(raw2)**2

    # # 畫圖
    # plt.figure(figsize=(10, 4))
    # plt.plot(fft_freq, fft_power, color='blue')
    # plt.title("Frequency Domain (FFT)-HR222222")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.xlim(0, 1)  # 可視範圍建議 0~1 Hz，若要分析 HR 可到 3~4 Hz
    # plt.tight_layout()
    # plt.show()

    hr_freqs = 60. * fft_freq
    hr_idx = np.where((hr_freqs > 60) & (hr_freqs < 120))
    
    pruned = fft_power[hr_idx]
    pfreq = hr_freqs[hr_idx]
    
    try:
        idx_hr = np.argmax(pruned)
        bpm = pfreq[idx_hr]
    except:
        bpm = -1  # 代表無法估算

    # 找主導頻率
    if np.any(hr_idx):
        peak_idx = np.argmax(fft_power[hr_idx])
        peak_freq = fft_freq[hr_idx][peak_idx]
    else:
        peak_freq = 0
        bpm = -1  # 無法估算

    print(f"Dominant Frequency: {peak_freq:.3f} Hz")

    if bpm > 0:
        print(f"心率: {bpm:.2f} BPM")
    else:
        print("心率: 無法估算（主頻為 0）")
    
    return bpm
    


    # heart rate(子月)############################################################
    # raw = np.fft.rfft(norm*100)                
    # freqs = float(fps) / L * np.arange(L / 2 + 1)  # freq(x軸)                                    
    # fft = np.abs(raw)**2   # power (y 軸)
    
    # hr_freqs = 60. * freqs    # 心率 freq bpm     
    # hr_idx = np.where((hr_freqs > 60) & (hr_freqs < 100))
    # #sidx2=str(idx2)+'\n'
    # #print(sidx2)
                
    # pruned = fft[hr_idx]               
    # pfreq = hr_freqs[hr_idx]
    
    # # freqs = pfreq 
    # # fft = pruned
            
    # try:
    #     idx_hr=np.argmax(pruned)                
    #     bpm = pfreq[idx_hr]
    # except:
    #     bpm = bpm
    # return  bpm 
    # heart rate (子月)#####################################################################
    


# """傅立葉"""
# def fft(signal, sampling_rate):
#     """
#     對時域訊號進行快速傅立葉轉換，並畫出頻域圖

#     參數:
#     - signal: 時域訊號（1D numpy array）
#     - sampling_rate: 取樣頻率（Hz）
#     - title: 圖表標題（可選）

#     回傳:
#     - freqs: 頻率軸
#     - magnitude: 對應頻率的振幅（magnitude spectrum）
#     """

#     # 計算FFT
#     n = len(signal)
#     fft_vals = np.fft.fft(signal)
#     fft_vals = np.abs(fft_vals) / n  # 正規化振幅
#     fft_vals = fft_vals[:n // 2]     # 取前半部（單邊頻譜）
    
#     # 頻率軸
#     freqs = np.fft.fftfreq(n, d=1/sampling_rate)
#     freqs = freqs[:n // 2]

    # # 找出主頻率
    # main_freq = freqs[np.argmax(fft_vals)]
    # heart_rate = main_freq * 60  # 轉為 bpm
    # print(f"主頻率：{main_freq:.2f} Hz，推估心率：{heart_rate:.1f} bpm")

    # # 畫圖
    # plt.figure(figsize=(10, 4))
    # plt.plot(freqs, fft_vals, color='blue')
    # plt.title("Frequency Domain (FFT)")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # return freqs, fft_vals


# def find_cycles(signal_data):
#     signal_data = np.array(signal_data)
#     values, times = signal_data[:, 0], signal_data[:, 1]
    
#     # 計算一階差分
#     diff_values = np.diff(values)
    
#     # 找出波峰（正變負）和波谷（負變正）
#     peaks = np.where((diff_values[:-1] > 0) & (diff_values[1:] <= 0))[0] + 1
#     troughs = np.where((diff_values[:-1] < 0) & (diff_values[1:] >= 0))[0] + 1
    
#     # 週期列表
#     cycles = []
    
#     # 遍歷所有相鄰的波谷來定義週期
#     for i in range(len(troughs) - 1):
#         start_trough = troughs[i]
#         end_trough = troughs[i + 1]
        
#         # 找出這個區間內的波峰與波谷
#         peaks_in_cycle = [p for p in peaks if start_trough < p < end_trough]
#         troughs_in_cycle = [t for t in troughs if start_trough < t < end_trough]
        
#         if not peaks_in_cycle or not troughs_in_cycle:
#             continue  # 沒有完整的週期，跳過
        
#         # 找主波峰（區間內最高的波峰）與次波峰
#         main_peak = max(peaks_in_cycle, key=lambda p: values[p])
#         secondary_peak = max((p for p in peaks_in_cycle if p != main_peak), 
#                              key=lambda p: values[p], default=None)
        
#         # 找主波谷（區間內最低的波谷）與次波谷
#         main_trough = min(troughs_in_cycle, key=lambda t: values[t])
#         secondary_trough = min((t for t in troughs_in_cycle if t != main_trough), 
#                                key=lambda t: values[t], default=None)
        
#         # 記錄這個週期
#         cycles.append({
#             "start_trough": start_trough,
#             "main_peak": main_peak,
#             "main_trough": main_trough,
#             "secondary_peak": secondary_peak,
#             "secondary_trough": secondary_trough,
#             "end_trough": end_trough
#         })
        
#     # 確認是否成功識別週期
#     if not cycles:
#         print("沒有識別到任何週期！請檢查信號資料。")

#     # 繪製訊號與週期標記
#     plt.figure(figsize=(12, 6))
#     plt.plot(times, values, label='Signal', color='blue', linewidth=1.5)

#     for cycle in cycles:
#         # 主波峰（紅色，較大）
#         if cycle["main_peak"] is not None:
#             plt.scatter(times[cycle["main_peak"]], values[cycle["main_peak"]], 
#                         color='darkred', marker='*', s=200, edgecolors='black', linewidths=1.5)

#         # 主波谷（深綠色，較大）
#         if cycle["main_trough"] is not None:
#             plt.scatter(times[cycle["main_trough"]], values[cycle["main_trough"]], 
#                         color='darkgreen', marker='*', s=200, edgecolors='black', linewidths=1.5)

#         # 次波峰（橙色）
#         if cycle["secondary_peak"] is not None:
#             plt.scatter(times[cycle["secondary_peak"]], values[cycle["secondary_peak"]], 
#                         color='orange', marker='^', s=150, edgecolors='black', linewidths=1.2)

#         # 次波谷（紫色）
#         if cycle["secondary_trough"] is not None:
#             plt.scatter(times[cycle["secondary_trough"]], values[cycle["secondary_trough"]], 
#                         color='purple', marker='v', s=150, edgecolors='black', linewidths=1.2)

#         # 週期開始 & 結束波谷（藍色圓圈）
#         if cycle["start_trough"] is not None:
#             plt.scatter(times[cycle["start_trough"]], values[cycle["start_trough"]], 
#                         color='blue', marker='o', s=120, edgecolors='black', linewidths=1.2)

#         if cycle["end_trough"] is not None:
#             plt.scatter(times[cycle["end_trough"]], values[cycle["end_trough"]], 
#                         color='blue', marker='o', s=120, edgecolors='black', linewidths=1.2)

#     plt.xlabel('Time (ms)')
#     plt.ylabel('Signal Value')
#     plt.title('Signal with Cycles Identified')

#     # 移除重複標籤
#     handles, labels = plt.gca().get_legend_handles_labels()
#     unique_labels = dict(zip(labels, handles))
#     plt.legend(unique_labels.values(), unique_labels.keys())

#     plt.grid()
#     plt.show()


# def find_turning_points(signal_data):
#     if signal_data is None or len(signal_data) == 0:
#         print("Find turning points fail: Input signal_data is empty or None.")
#         return None

#     # 轉換成 numpy array
#     signal_data = np.array(signal_data)

#     # 檢查數據是否至少有兩列（值, 時間）
#     if signal_data.ndim != 2 or signal_data.shape[1] < 2:
#         print("Find turning points fail: signal_data must have two columns [value, time].")
#         return None

#     values, times = signal_data[:, 0], signal_data[:, 1]

#     # 計算一階與二階差分
#     first_diff = np.diff(values)
#     second_diff = np.diff(first_diff)

#     # 確保時間軸對齊（差分後長度變短）
#     second_diff_times = times[2:]  

#     # 找二階差分的峰值（曲率變化最大的位置）
#     peaks, _ = find_peaks(np.abs(second_diff))

#     # **修正錯誤處理**
#     if peaks is None or len(peaks) == 0:
#         print("Find turning points fail: No turning points found.")
#         return None

#     # 取得轉折點的時間和數值
#     turning_times = second_diff_times[peaks]
#     turning_values = values[peaks + 1]  # 索引對應 values，需要加 1

#     # 畫出原始信號與轉折點
#     plt.figure(figsize=(10, 5))
#     plt.plot(times, values, label='Original Signal', color='blue')
#     plt.scatter(turning_times, turning_values, color='red', label='Turning Points', marker='o')
    
#     plt.xlabel('Time (s)')
#     plt.ylabel('Signal Value')
#     plt.title('Signal with Turning Points')
#     plt.legend()
#     plt.grid()
#     plt.show()

#     return list(zip(turning_times, turning_values))


# # 計算導數
# def compute_derivatives(y, x):
#     dy = np.gradient(y, x)  # 一階導數
#     ddy = np.gradient(dy, x)  # 二階導數
#     dddy = np.gradient(ddy, x)  # 三階導數
#     return dy, ddy, dddy

#     # 找尋波峰與波谷
# def detect_peaks_and_valleys(y, ddy):
#     peaks, _ = find_peaks(y)  # 找波峰
#     valleys, _ = find_peaks(-y)  # 找波谷

#     # 使用二階導數極值來找拐點 (次波峰與次波谷)
#     inflection_points, _ = find_peaks(np.abs(ddy))
#     return peaks, valleys, inflection_points
    
# def find_rppg_peaks_and_troughs2222(data):
#     # 解析數據
#     values = np.array([point[0] for point in data])
#     time = np.array([point[1] for point in data])
    
#     # 計算一階差分
#     diff_values = np.diff(values)
#     diff_time = time[:-1]  # 差分後時間少一個
    
#     # 找出一階差分的波峰
#     peaks, _ = find_peaks(diff_values)
    
#     plt.figure(figsize=(10, 4))
#     plt.plot(diff_time, diff_values, marker='o', label='First-order Difference')
#     plt.xlabel("Time")
#     plt.ylabel("Difference Value")
#     plt.title("First-order Difference of RPPG Signal")
#     plt.legend()
#     plt.show()

#     if len(peaks) == 0:
#         raise ValueError("未找到波峰")
    
#     # 找出最高的波峰 T
#     T = np.max(diff_values[peaks])
    
#     # 設定界線 0.6T
#     threshold = 0.6 * T
#     sub_peaks = peaks[diff_values[peaks] < threshold]
    
#     # 找出過零點
#     zero_crossings = np.where(np.diff(np.sign(diff_values)))[0]
#     pos_to_neg = zero_crossings[diff_values[zero_crossings] > 0]  # 正轉負（次波峰）
#     neg_to_pos = zero_crossings[diff_values[zero_crossings] < 0]  # 負轉正（次波谷）
    
#     # 繪圖
#     plt.figure(figsize=(10, 5))
#     plt.plot(diff_time, diff_values, label='First-order Difference', color='blue')
#     plt.scatter(diff_time[peaks], diff_values[peaks], color='red', label='Primary Peaks')
#     plt.scatter(diff_time[sub_peaks], diff_values[sub_peaks], color='orange', label='Sub Peaks')
#     plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold (0.6T)')
#     plt.scatter(diff_time[pos_to_neg], diff_values[pos_to_neg], color='green', label='Secondary Peaks')
#     plt.scatter(diff_time[neg_to_pos], diff_values[neg_to_pos], color='purple', label='Secondary Troughs')
#     plt.xlabel("Time")
#     plt.ylabel("Difference Value")
#     plt.legend()
#     plt.title("RPPG Signal Peak and Trough Detection")
#     plt.show()

#     # 結果
#     result = {
#         "primary_peaks": [(diff_time[p], diff_values[p]) for p in peaks],
#         "threshold": threshold,
#         "sub_peaks": [(diff_time[p], diff_values[p]) for p in sub_peaks],
#         "zero_crossings": diff_time[zero_crossings],
#         "secondary_peaks": diff_time[pos_to_neg],
#         "secondary_troughs": diff_time[neg_to_pos]
#     }
    
#     return result


# def find_peaks_in_second_derivative(listTemp, threshold=0.1):
#     # 解壓縮數據 (確保 listTemp 內容是 [(數值, 時間), (數值, 時間), ...])
#     values, timestamps = zip(*listTemp)

#     # 轉為 NumPy 陣列
#     values = np.array(values)
#     timestamps = np.array(timestamps)
    
#     second_diff = np.diff(values, n=2)
#     peaks, _ = find_peaks(second_diff, height=threshold)  # 找出二階差分的波峰
    
#     # 對應時間戳記（因為二階差分少了 2 個點，對應時間要從 index=1 開始）
#     peak_times = timestamps[peaks + 1]
    
 
#     origin= [(values,timestamps)]

#     # 畫圖
#     plt.figure(figsize=(12, 5))

#     plt.subplot(2, 1, 1)
#     plt.plot(timestamps, values, label="Original Signal")
#     plt.title("Original Signal")
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.grid(True)

#     plt.subplot(2, 1, 2)
#     plt.plot(timestamps[1:-1], second_diff, label="Second Derivative", color="g")
#     plt.plot(peak_times, second_diff[peaks], "ro", label="Peaks")  # 標記波峰
#     plt.title("Second Derivative and Peaks")
#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()
    
#     return origin, second_diff, peaks, peak_times

"""時域計算+前處理"""
def preProcessing_timeDomain(file_path_r:str):
# def preProcessing(timeWindow:int,第幾個timeWindow:int,file_path_r:str,區間_from:int,區間_to:int):
    
    tRatio_1 = tRatio_2 = hRatio_1 = hRatio_2 = aRatio_1 = aRatio_2 = None
    sdnn = rmssd = None

    
    listTemp =[]

    #設定timeWindow
    # timeWindow = 300
    # 第幾個timeWindow =  1

    """讀檔"""
    print("\n =======時域計算=====================================")
    listTemp = read_from_file(file_path_r)

    #將listTemp轉換為np.array
    listTemp = np.array(listTemp)
    #print(f"檢查：\n {listTemp[:5]}")

    """平滑化1"""
    # window_length 控制平滑範圍大小，數值越大，平滑效果越強。
    # polyorder 決定多項式的擬合複雜度，常用低階（如 2 或 3）。
    # mode 用於選擇邊界處理方式，默認 'interp'。

    # listTemp[:,0] = scipy.signal.savgol_filter(listTemp[:,0].astype(float), 時窗長度(奇數), 多項式階數)
    try:
        # listTemp[:,0] = scipy.signal.savgol_filter(listTemp[:,0].astype(float), timeWindow, 2)
        # listTemp[:,0] = scipy.signal.savgol_filter(listTemp[:,0].astype(float), 100 ,2)
        # listTemp[:,0] = smooth_signal(listTemp[:,0],100) ###要轉成float才能用

        ## smoothTriangle(data, degree)
        listTemp[:,0] = smoothTriangle(listTemp[:,0], 5)
        
        #plot_data(listTemp,"Smoothing 1 ")
    except Exception as e:
        print("Smoothing fail" + str(e))     

    """帶通濾波"""
    # try:
    #     # listTemp_處理前 = listTemp.copy()
        
    #     #原始程式使用的帶通濾波處理程式碼
    #     #帶通濾波器
    #     # processed_samples=listTemp[:,0]
    #     # # processed = butterworth_bandpass_filter(norm,0.1,3.5,self.fps,order=1)

    #     # processed = butterworth_bandpass_filter(norm,0.1,3.5,30,order=4)
    #     # # processed_samples = butterworth_bandpass_filter(processed_samples,0.8,3,self.fps,order=4)
    #     # processed_samples = butterworth_bandpass_filter(processed_samples,0.8,3,30,order=4)

    #     # processed=signal.detrend(processed)
    #     # processed_samples=signal.detrend(processed_samples)
    #     # listTemp[:,0]=signal.detrend(processed)
    #     # listTemp[:,0]=signal.detrend(processed_samples)

    #     #2024.12.01找到的使用bandpass Butterworth filter对信号数据进行滤波去噪
    #     # time = np.linspace(0, 0.02, 52)

    #     # bandPass_filter(signal, fs=12, lowcut=0.01, highcut=3, order=2):
    listTemp[:,0]= bandPass_filter(listTemp[:,0], 12, 0.03, 3 ,2)

    #     # 帶通濾波處理
    #     plot_data(listTemp,"Bandpass Filtering")
    #     # 兩圖合併(listTemp_處理前,listTemp,"Bandpass Filtering")
    # except Exception as e:
    #     print("Bandpass Filtering fail" + str(e))

    """線性插值法+漢明窗"""
    try:
        # 使用 listTemp[:, 0] (值) 和 listTemp[:, 1] (時間)
        listTemp = interpolated_signal(listTemp[:, 0], listTemp[:, 1])
        #plot_data(listTemp, "Interpolated")

        # # # 標準化插值結果(為何要兩個norm)
        # listTemp[:,0] = (listTemp[:,0] - np.mean(listTemp[:,0])) / np.std(listTemp[:,0])  
        # listTemp[:,0] = listTemp[:,0]/np.linalg.norm(listTemp[:,0])
        # 繪製插值後的信號
        # plot_data(listTemp, "norm")

    except Exception as e:
        print(f"Interpolated fail: {e}")

    #補點
    # even_times = np.linspace(listTemp[0,1], listTemp[len(listTemp)-1,1], 100)
    # listTemp[:,0] = np.interp(even_times, listTemp[:,1], listTemp[:,0])
    # listTemp[:,0]= np.hamming(100) * listTemp[:,0] 
    # listTemp[:,0] = (listTemp[:,0] - np.mean(listTemp[:,0]))/np.std(listTemp[:,0])
                            
    # listTemp[:,0] = listTemp[:,0]/np.linalg.norm(listTemp[:,0]) 

    #將listTemp轉換為float
    # listTemp =listTemp.astype(float)

    
    """平滑化2"""
    try:
        listTemp[:,0] = smoothTriangle(listTemp[:,0], 5)
        
        #將信號畫成圖型 
        #plot_data(listTemp,"Smoothing 2")
    except Exception as e:
        print("Smoothing 2 fail \n" + str(e))   


    """基線飄移(校準)"""
    ##基線校正 03 (有時窗)
    try:
        # listTemp = baseline_removal2(listTemp,100)
        listTemp = baseline_removal(listTemp,2,100)

        #plot_data(listTemp," Baseline correction")
    except Exception as e:
        print("Baseline correction fail" + str(e))

    
    """平滑化3"""
    try:
        listTemp[:,0] = smoothTriangle(listTemp[:,0], 5)
        
        #將信號畫成圖型 
        #plot_data(listTemp,"Smoothing 3")
    except Exception as e:
        print("Smoothing 3 fail \n" + str(e))


    """歸一化"""
    try:
        # avg = sum(listTemp[:,0])/len(listTemp[:,0])
        # listTemp[:,0] = Z_ScoreNormalization(listTemp[:,0],avg,statistics.stdev(listTemp[:,0]))
        # 創建MinMaxScaler實例，預設縮放到[0, 1]範圍
        # scaler = MinMaxScaler()

        # # 使用MinMaxScaler對特徵進行擬合和轉換
        # listTemp[:,0:1] = scaler.fit_transform(listTemp[:,0:1])
        listTemp = normalization(listTemp)

        # listTemp[:,1] = listTemp_處理前[:,1]
        # 歸一化
        # 兩圖合併(listTemp_處理前,listTemp,"Normalization")

        # listTemp[:,0:1] = normalize_with_time_window(listTemp[:,0:1],100)

        #plot_data(listTemp,"Normalization")
    except Exception as e:
            print("Normalization fail" + str(e))
            traceback.print_exc()


    """找波峰，計算時域特徵、PTT時域"""
    try:
        # 找波峰
        peaks, peaks_list = finding_peaks(listTemp)
        print("\n--共有", len(peaks),"個波峰")
        # print("---波峰索引值(最多2筆資料)：", peaks_list)
        
        # 計算ppi --> 計算sdnn. rmssd
        ppi_values = ppi_cal(peaks, 15)
        sdnn, rmssd = sdnn_rmssd(ppi_values)
        print("\n--SDNN：", sdnn)
        print("--RMSSD:", rmssd, "\n")

        # 計算血壓特徵(時間比、振福比、面積比)
        values, times, peaks,troughs, target_valleys, signal_data = finding_peaks_bp(listTemp)
        result, tRatio_1, tRatio_2, hRatio_1, hRatio_2, aRatio_1, aRatio_2 = bp_features_cal(values, times, peaks,troughs, target_valleys, signal_data)
        
        # print("\n--所有特徵點:\n",result)
        # print("\n--時間比特徵 1 :",tRatio_1)
        # print("--時間比特徵 2 :",tRatio_2)
        # print("\n--振幅比特徵 1 :",hRatio_1)
        # print("--振幅比特徵 2 :",hRatio_2)
        # print("\n--面積比特徵 1 :",aRatio_1)    
        # print("--面積比特徵 2 :", aRatio_2)
        

    except Exception as e:
        print("Features Calculation fail"+ str(e))

    """計算近似商、樣本商"""
    try:
        # # 設定參數
        m = 2
        r = 0.15 * np.std(listTemp[:,0])

        # # 計算 ApEn 和 SampEn(data, m, r)
        # #減少 m(維度)（通常用 m=2 是較穩定的起點）
        # #適當增大 r(兩者距離)（建議設為 0.1~0.25 * std(x)）
        
        apen = fast_apen(listTemp[:,0], m, r)
        sampen = fast_sampen(listTemp[:,0], m, r)

        # SampEn
        # x 是時間序列，order=m 為維度，r 是容差參數（通常為 std(x) 的 0.1 ~ 0.25 倍）
        # sampen = ant.sample_entropy(listTemp[:,0], order=m, r=r)

        # # ApEn
        # apen = ant.app_entropy(listTemp[:,0], order=m, r=r)

        apen = round(apen,3)
        sampen = round(apen,3)

        print("Approximate Entropy (ApEn):",apen)
        print("Sample Entropy (SampEn):",sampen)

    except Exception as e:
        print("Entrppy Calculation Fail"+ str(e))

    """特徵總和"""
    # 每次計算完一筆資料後
    hrvFeatures = [sdnn, rmssd]
    
    bpFeatures = [
        tRatio_1, tRatio_2,
        hRatio_1, hRatio_2,
        aRatio_1, aRatio_2,
    ]

    nonlinFeature = [sampen, apen]

    timeFeatures = hrvFeatures + bpFeatures + nonlinFeature
    
    timeFeatures_list = []
    timeFeatures_list.append(timeFeatures)

    # print("\n--時域特徵:",
    #       "\n tRatio_1:",tRatio_1,
    #       "\n tRatio_2:", tRatio_2,
    #       "\n hRatio_1:",hRatio_1,
    #       "\n hRatio_2:", hRatio_2,
    #       "\n aRatio_1:", aRatio_1,
    #       "\n aRatio_2:", aRatio_2,
    #       "\n sdnn:", sdnn,
    #       "\n rmssd:", rmssd,
    #       "\n SampEn:", sampen,
    #       "\n ApEn:", apen
    #       )

    return timeFeatures_list


"""頻域計算+前處理"""
def preProcessing_freqDomain(file_path_r:str):
    
    nlf = nhf = lf_hf_ratio = None
    heart_rate_bpm1 = heart_rate_bpm2 = heart_rate_bpm3 = None
    
    listTemp =[]

    #設定timeWindow
    # timeWindow = 300
    # 第幾個timeWindow =  1

    """讀檔"""
    print("=======頻域計算=====================================")
    listTemp = read_from_file(file_path_r)

    #將listTemp轉換為np.array
    listTemp = np.array(listTemp)
    #print(f"檢查：\n {listTemp[:5]}")

    """平滑化1"""
    # window_length 控制平滑範圍大小，數值越大，平滑效果越強。
    # polyorder 決定多項式的擬合複雜度，常用低階（如 2 或 3）。
    # mode 用於選擇邊界處理方式，默認 'interp'。
    # listTemp[:,0] = scipy.signal.savgol_filter(listTemp[:,0].astype(float), 時窗長度(奇數), 多項式階數)

    try:
        ## smoothTriangle(data, degree)
        listTemp[:,0] = smoothTriangle(listTemp[:,0], 5)
       #plot_data(listTemp,"Smoothing 1 ")

    except Exception as e:
        print("Smoothing fail" + str(e))     


    """線性插值法+漢明窗"""
    try:
        # 使用 listTemp[:, 0] (值) 和 listTemp[:, 1] (時間)
        listTemp = interpolated_signal(listTemp[:, 0], listTemp[:, 1])
        #plot_data(listTemp, "Interpolated")

    except Exception as e:
        print(f"Interpolated fail: {e}")

    #補點
    # even_times = np.linspace(listTemp[0,1], listTemp[len(listTemp)-1,1], 100)
    # listTemp[:,0] = np.interp(even_times, listTemp[:,1], listTemp[:,0])
    # listTemp[:,0]= np.hamming(100) * listTemp[:,0] 
    # listTemp[:,0] = (listTemp[:,0] - np.mean(listTemp[:,0]))/np.std(listTemp[:,0])
                            
    # listTemp[:,0] = listTemp[:,0]/np.linalg.norm(listTemp[:,0]) 

    #將listTemp轉換為float
    # listTemp =listTemp.astype(float)

    
    """平滑化2"""
    try:
        listTemp[:,0] = smoothTriangle(listTemp[:,0], 5)
        
        #將信號畫成圖型 
        #plot_data(listTemp,"Smoothing 2")
    except Exception as e:
        print("Smoothing 2 fail \n" + str(e))   


    """基線飄移(校準)"""
    # baseline_als(listTemp2[:,0].astype(float), 1e5, 0.1)
    # for test基線校準
    # input_array=[10,20,1.5,5,2,9,99,25,47]
    
    #基線校正(原)(全部一起基線校正)
    # try:
    #     # polynomial_degree=2 #only needed for Modpoly and IModPoly algorithm
    #     # #for test基線校準
    #     # baseObj=BaselineRemoval(listTemp[:,0])
    #     # #baseObj=BaselineRemoval(input_array)
    #     # #listTemp[:,0] = detrend(listTemp[:,0],100,'linear',500)
    #     # listTemp[:,0]=baseObj.ModPoly(polynomial_degree)
    #     # #baseObj=BaselineRemoval(listTemp[:,0])
    #     # # listTemp[:,0]=baseline_correction(listTemp[:,1],listTemp[:,0], 2)
    #     # #listTemp[:,0]=baseObj.IModPoly(polynomial_degree)
    #     # # listTemp[:,0]=baseObj.ZhangFit()
    #     # #將信號畫成圖型 

    #     listTemp[:,0] = baseline_correction(listTemp[:,0],150)
    
    #     plot_data(listTemp," Baseline correction")
    # except Exception as e:
    #     print("Baseline correction fail" + str(e))


    ##基線校正 03 (有時窗)
    try:
        # listTemp = baseline_removal2(listTemp,100)
        listTemp = baseline_removal(listTemp,2,100)

        #plot_data(listTemp," Baseline correction")
    except Exception as e:
        print("Baseline correction fail" + str(e))


    """平滑化3"""
    try:
        listTemp[:,0] = smoothTriangle(listTemp[:,0], 5)
        
        #將信號畫成圖型 
        #plot_data(listTemp,"Smoothing 3")
    except Exception as e:
        print("Smoothing 3 fail \n" + str(e))


    """歸一化"""
    try:
        listTemp = normalization(listTemp)
        #plot_data(listTemp,"Normalization")
    except Exception as e:
            print("Normalization fail" + str(e))

    """頻域特徵計算"""
    try:
        
        """計算LF、HF、LF/HF"""
        # 帶通(0.03-0.4)
        listTemp[:,0]= bandPass_filter(listTemp[:,0], 12, 0.03, 0.4 ,2)
        #plot_data(listTemp, "bandpass-LF+HF")
        # 傅立葉轉換
        fft_freq, fft_power = fft_cal(listTemp,12)
        # 計算LF、HF、LF/HF
        nlf, nhf, lf_hf_ratio = lfhf_cal(fft_freq, fft_power)

    except Exception as e:
        print("LF,HF, LF/HF Calculation fail"+ str(e))

    try:
        """計算HR"""
        # 帶通(0.6-3)
        # listTemp[:,0]= bandPass_filter(listTemp[:,0], 12, 0.03, 0.4 ,2)
        #plot_data(listTemp, "bandpass-hrrrrr")
        # 傅立葉轉換+ HR計算
        heart_rate_bpm = hr_cal(fft_freq,fft_power)
        heart_rate_bpm = round(heart_rate_bpm,3)
    
    except Exception as e:
        print("HR Calculation fail"+ str(e))

    # 每次計算完一筆資料後
    freqFeatures = [
        nlf, nhf, lf_hf_ratio, 
        heart_rate_bpm
    ]
    
    freqFeatures_list = []
    freqFeatures_list.append(freqFeatures)

    print("\n--頻域特徵:",
          "\n nLF:",nlf,
          "\n nHF:", nhf,
          "\n LF/HF:",lf_hf_ratio,
          "\n heart_rate:", heart_rate_bpm,
          )

    return freqFeatures_list

def main_all_features(listTemp):
    
    timeFeatures_list = preProcessing_timeDomain(listTemp)
    freqFeatures_list = preProcessing_freqDomain(listTemp)

    
    # 3. 取出各自的特徵向量（注意是 list 裡面的 list）
    time_features = timeFeatures_list[0]  # e.g. [sdnn, rmssd, ...]
    freq_features = freqFeatures_list[0]  # e.g. [nlf, nhf, ...]
    
    
    all_features_list = time_features + freq_features


    print("\n 所有特徵:",all_features_list)
    
    return all_features_list
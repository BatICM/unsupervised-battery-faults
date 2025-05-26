# -*- coding: utf-8 -*-
"""
Created on Tue May 27 02:07:23 2025

@author: lenovo
"""


from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import time
import datetime as dt
import cvxopt
from sklearn.cluster import KMeans
from multiprocessing import Pool
from functools import partial
import multiprocessing as mp
#%%
def preprocess_car(data):
    # data=pd.read_csv(r'C:\Users\lenovo\Desktop\github\data\car2.csv')
    
    data=data[data.BMSBatteryVoltage>100]
    if np.mean(data.BMSCellVoltageMin)>1000:
        data=data[data.BMSCellVoltageMin>1000]  
    else:
        data=data[data.BMSCellVoltageMin>1]
        data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)# 重排以获得正确时间序列
    # for i in range(len(data)):
    #     data.tboxTime[i]=dt.datetime.fromtimestamp(data.tboxTime[i])
    data['tboxTime'] = pd.to_datetime(data['tboxTime'])    
    data = data.sort_values(by='tboxTime')
    # save_path='C:\\Battery_Safety_Evaluation\\CAERI'
    cell_V_start=np.where(data.columns=='BMSCellVoltageM')[0][0]
    cell_T_start=np.where(data.columns=='BMSProbeTempM')[0][0]
    cell_volt=data.iloc[:,cell_V_start+1:cell_T_start]
    cell_Tem=data.iloc[:,cell_T_start+1:]
    # -------------------------------- 5.电压异常值处理
    # 删除大量空缺值：删除df中电压空值个数超过5的行
    rows_null = cell_volt.isnull().sum(axis=1)
    find_index = rows_null[rows_null > 5].index.tolist()
    cell_volt = cell_volt.drop(find_index)
    # 异常值替换：行平均值替代>4.5或<0.5的电压异常值
    vlot_ub = 4.5
    vlot_lb = 0.5
    volt_out_idx1 = cell_volt[(cell_volt > vlot_ub).any(axis=1)].index
    volt_out_idx2 = cell_volt[(cell_volt < vlot_lb).any(axis=1)].index
    volt_out_idx = list(set(volt_out_idx1).union(set(volt_out_idx2)))
    out_len = len(volt_out_idx)
    if out_len > 0:
        for idx in volt_out_idx:
            item = cell_volt.loc[idx]
            out_volt1 = item[item > vlot_ub].index.get_level_values(0)
            out_volt2 = item[item < vlot_lb].index.get_level_values(0)
            out_volt = out_volt1.union(out_volt2)
            in_volt = item.index.get_level_values(0).difference(out_volt)
            volt_ave = item.loc[in_volt].mean()
            mapping = dict(zip(item.loc[out_volt].values, [volt_ave] * len(out_volt)))
            cell_volt.loc[idx, :].replace(mapping, inplace=True)
    # 异常值替换：行平均值替代>4.5或<0.5的电压异常值
    rows_null_1 = cell_volt.isnull().sum(axis=1)
    find_index_1 = rows_null_1[rows_null > 0].index.tolist()
    cell_volt = cell_volt.drop(find_index_1)
    cell_volt = cell_volt.reset_index(drop=True)  # 预处理后的单体电压  
    # -------------------------------- 6.特征提取
    # 标准化的单体电压矩阵作为特征
    a = np.square(cell_volt).T
    sum_a = a.sum(axis=0)
    sqrt_sum_a = np.sqrt(sum_a)
    b = []
    for i in range(a.shape[1]):#列表生成式
        b1 = 1 / sqrt_sum_a[i]
        b.append(b1)
    nor_cell_volt = []
    for i in range(a.shape[1]):
        c1 = cell_volt.iloc[i, :] * b[i]
        nor_cell_volt.append(c1)
    nor_cell_volt = np.array(nor_cell_volt)
    feature_matrix = nor_cell_volt
    cell_volt_normized = cell_volt
    feature_matrix = pd.DataFrame(feature_matrix)
    return cell_volt_normized,feature_matrix
#%%
# 创建预处理函数映射字典
preprocessing_functions = {
    'car1': preprocess_car,
    'car2': preprocess_car,
    'car3': preprocess_car,
    'car4': preprocess_car
}
# 将file_paths改为字典格式
file_paths = {
    'car1': 'C:/Users/lenovo/Desktop/github/data_new/car1_processed.csv',
    'car2': 'C:/Users/lenovo/Desktop/github/data_new/car2_processed.csv',
    'car3': 'C:/Users/lenovo/Desktop/github/data_new/car3_processed.csv',
    'car4': 'C:/Users/lenovo/Desktop/github/data_new/car4_processed.csv'
}

# 修改process_selected_car_files函数
def process_selected_car_files(file_paths_dict, selected_files=None):
    results = {}
    
    # 如果未指定要处理的文件，则处理所有文件
    if selected_files is None:
        selected_files = list(file_paths_dict.keys())
    
    for car_type, file_path in file_paths_dict.items():
        # 如果当前车型不在选择列表中，则跳过
        if car_type not in selected_files:
            print(f"跳过: {car_type}")
            continue
            
        # 检查是否有对应的预处理函数
        if car_type in preprocessing_functions:
            # 读取CSV文件
            print(f"读取文件: {file_path}")
            df = pd.read_csv(file_path)
            
            # 应用对应的预处理函数
            processed_df, feature_mat = preprocessing_functions[car_type](df)
            
            # 存储处理后的数据框
            results[car_type] = [processed_df, feature_mat]
            print(f"成功处理: {car_type}")
        else:
            print(f"警告: 没有找到{car_type}的预处理函数")
    
    return results
car_type = 'car4'
# 调用函数
processed_data = process_selected_car_files(file_paths, selected_files=[car_type])

# 然后从processed_data中获取结果
if processed_data and car_type in processed_data:
    cell_volt_normized, feature_matrix = processed_data[car_type]
# In[提取香农熵特征]
def ShannonEn_K_Ensemble(x, n, K, step):
    N = len(x)
    num_windows = (N - K) // step + 1
    x_K = np.zeros((K, num_windows))
    H_x = np.zeros(num_windows)
    
    for i in range(num_windows):
        start_index = i * step
        end_index = start_index + K
        x_K[:, i] = x[start_index:end_index]
        H_x[i] = ShannonEn(x_K[:, i], n, x)
    
    return H_x

def ShannonEn(x_K, n, x):
    x_max = np.max(x_K)
    x_min = np.min(x_K)
    delta = abs(x_max - x_min) / n
    
    num = np.zeros(n)
    num[0] = np.sum(x_K < x_min + delta)
    
    for i in range(1, n-1):
        num[i] = np.sum((x_K >= x_min + (i-1)*delta) & (x_K < x_min + i*delta))
    
    num[n-1] = np.sum(x_K >= x_min + (n-1)*delta)
    
    p_num = num / np.sum(num)
    
    lnp_num = np.zeros(n)
    for i in range(n):
        if p_num[i] == 0:
            lnp_num[i] = 0
        else:
            lnp_num[i] = np.log(p_num[i])
    
    H_x = -np.sum(p_num * lnp_num)
    
    return H_x
# 顺序处理每个索引
veh_list = []
for index in range(cell_volt_normized.shape[1]):
    # 参数设置
    n = 30
    K = 100
    step = 1
    
    # 计算香农熵特征
    cell_volt_shan = ShannonEn_K_Ensemble(
        cell_volt_normized.values[:, index], 
        n, 
        K, 
        step
    )
    veh_list.append(cell_volt_shan)
# In[提取状态值特征]
# SRM算法
# -------------------------------- 1.确定时间窗口长度，确定标准状态矩阵G1
def SRM(feature_matrix):
    M = 1  # 时间窗口长度
    U1 = feature_matrix.iloc[0:M, :]
    U1 = np.array(U1)
    G1 = np.dot(U1.T, U1)
    # -------------------------------- 2.求标准状态C1和标准特征权重res(x)
    N = U1.shape[1]
    e = np.ones(N)
    e = mat(e)
    F = np.dot((np.eye(N) - 1 / N * e.T * e), G1)
    F = mat(F)
    # 目标函数min（||FX||^2）    X'F'FX
    H = F.T * F
    
    
    def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
        n_var = H.shape[1]
        P = cvxopt.matrix(H, tc='d')
        q = cvxopt.matrix(f, tc='d')
        if L is not None or k is not None:
            assert (k is not None and L is not None)
            if lb is not None:
                L = np.vstack([L, -np.eye(n_var)])
                k = np.vstack([k, -lb])
            if ub is not None:
                L = np.vstack([L, np.eye(n_var)])
                k = np.vstack([k, ub])
            L = cvxopt.matrix(L, tc='d')
            k = cvxopt.matrix(k, tc='d')
        if Aeq is not None or beq is not None:
            assert (Aeq is not None and beq is not None)
            Aeq = cvxopt.matrix(Aeq, tc='d')
            beq = cvxopt.matrix(beq, tc='d')
        sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq, lb)
        return np.array(sol['x'])
    
    H = F.T * F
    print(H)
    f = np.zeros((N, 1))
    L = np.zeros((1, N))
    k = np.zeros((1, 1))
    Aeq = np.ones((1, N))
    beq = 1
    lb = np.zeros((N, 1))
    res = quadprog(H, f, L, k, Aeq, beq, lb)
    print(res)
    C1 = np.dot(G1, res)  # C1为标准窗口下电池包的标准状态，res为电池包各单体的状态权重
    #3.计算各时间窗口下电池包各单体的状态C2
    win_nums = math.floor(feature_matrix.shape[0] / M)
    feature_matrix_need = win_nums * M
    
    index_1 = [i for i in range(M, feature_matrix_need, M)]
    U2 = [];
    G2 = [];
    C2 = [];
    
    for k in index_1:
        U_1 = feature_matrix.iloc[k:k + M, :]
        U_1 = np.array(U_1)
        G_1 = np.dot(U_1.T, U_1)
        C_1 = np.dot(G_1, res)
        U2.append(U_1)
        G2.append(G_1)
        C2.append(C_1)
    
    C2 = np.reshape(C2, (-1, feature_matrix.shape[1]))  # C2为每一个时间窗口下电池包各单体的状态矩阵
    C2 = pd.DataFrame(C2)  
    return C2 
C2= SRM(feature_matrix)
# In[提取均方根特征]
def calculate_window_mean(dataframe, Nwindow):
    # 按行求均值
    row_mean = dataframe.mean(axis=1)

    # 减去每行的均值
    dataframe = dataframe.sub(row_mean, axis=0)
    dataframe_squared = dataframe.apply(np.square)

    # 按照滑动窗口的方法取处理好后的数据的长度为Nwindow的均值
    window_mean = dataframe_squared.rolling(window=Nwindow, axis=0).mean().dropna()

    return window_mean
Nwindow=10
cell_volt_RMSE=calculate_window_mean(cell_volt_normized, Nwindow)
cell_volt_RMSE = cell_volt_RMSE.reset_index(drop=True)
# 定义一个字典将英文列名映射为数字列名
column_mapping = {old_name: i for i, old_name in enumerate(cell_volt_RMSE.columns)}

# 使用 rename 方法将列名替换为数字列名
cell_volt_RMSE = cell_volt_RMSE.rename(columns=column_mapping)
# In[统一特征长度]
data_shan=pd.DataFrame(veh_list).transpose()
data_C2=pd.DataFrame(C2)
data_cell=pd.DataFrame(cell_volt_RMSE)
# 确保两个数据框的行数相同
min_rows = min(data_shan.shape[0], data_C2.shape[0],data_cell.shape[0])
data_shan_interpolated = data_shan.tail(min_rows)
data_C2_interpolated = data_C2.tail(min_rows)
data_cell_interpolated=data_cell.tail(min_rows)
cell_volt_interpolated=cell_volt_normized.tail(min_rows)
data_shan_interpolated.reset_index(drop=True, inplace=True)
data_C2_interpolated.reset_index(drop=True, inplace=True)
data_cell_interpolated.reset_index(drop=True, inplace=True)
cell_volt_interpolated.reset_index(drop=True, inplace=True) 
# In[归一化]
def min_max_normalize_rows(df):
    normalized_df = df.sub(df.min(axis=1), axis=0).div(df.max(axis=1) - df.min(axis=1), axis=0)
    return normalized_df
data_shan_normalized = min_max_normalize_rows(data_shan_interpolated)
data_C2_normalized = min_max_normalize_rows(data_C2_interpolated)
data_cell_normalized=min_max_normalize_rows(data_cell_interpolated)
data_shan_normalized.fillna(0.5, inplace=True)
data_C2_normalized.fillna(0.5, inplace=True)
data_cell_normalized.fillna(0.5, inplace=True)
# In[DBSCAN聚类以及滑动窗口评分]
import numpy as np
from sklearn.cluster import dbscan
import matplotlib.pyplot as plt
# 创建空的累积结果列表
cumulative_list = []
# 循环处理多组数据
for i in range(data_shan_normalized.shape[0]):    
    data_shan_1 = data_shan_normalized.iloc[i].values
    data_C2_1 = data_C2_normalized.iloc[i].values
    data_cell_1 = data_cell_normalized.iloc[i].values
    points = np.column_stack((data_shan_1, data_C2_1, data_cell_1))
    core_samples, cluster_ids = dbscan(points, eps=0.6, min_samples=3)
    cluster_ids = np.where(cluster_ids < 0, 1, 0)
    
    # 转置聚类结果并添加到累积列表
    cumulative_list.append(np.transpose(cluster_ids))

# 将累积列表转换为一个新的数组
combined_array = np.vstack(cumulative_list)

# 将数组转换为 DataFrame
combined_df = pd.DataFrame(combined_array)

window_size = combined_df.shape[1]  # 获取列数
# 按照行进行滑动窗口累加
new_combined_df = combined_df.rolling(window=window_size).sum()
# 删除前499行的NaN值
new_combined_df = new_combined_df[window_size-1:]
new_combined_df = new_combined_df.div(window_size)
# 重置索引，保留原始索引的顺序
new_combined_df = new_combined_df.reset_index(drop=True)
new_combined_df = new_combined_df.to_numpy()
new_df = pd.DataFrame(new_combined_df)  
# In[绘制评分结果]
from matplotlib import rcParams
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
params = {
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.style': 'normal',
    'font.weight': 'normal',  # or 'blod'
    'font.size':12,  # or large, small
}
rcParams.update(params)
# 设置阈值范围和警告级别
threshold = 0.5
severity1 = 'yellow' # 严重警告


new_combined_df = pd.DataFrame(new_combined_df)
max_values = {col: new_combined_df[col].max() for col in new_combined_df.columns}
step = max(1, new_combined_df.shape[0] // 6)
# 创建一个新的 DataFrame 变量，列名为 'Max_Values'，数据为每列的最大值
max_df = pd.DataFrame(max_values, index=['Max_Values'])
# 创建一个 colormap，从蓝色到红色渐变
cmap = plt.cm.get_cmap('coolwarm')
norm = Normalize(vmin=0, vmax=1)

# 绘制 dQ_dV 图
plt.figure(figsize=(4.5, 4.5),dpi=600)
# 在图中绘制阈值线
plt.axhline(y=threshold, color='#4CAF50', linestyle='--', label=f'Severe Fault Threshold ({threshold})')
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # 设置一个空数组
cbar = plt.colorbar(sm, orientation='horizontal', pad=0.18, aspect=36)  # 调整orientation, pad和aspect参数以控制颜色条的位置和大小
for o in range (0, len(new_combined_df.columns)):
    color = cmap(norm(max_df[o]))
    plt.plot(new_combined_df[o] , linewidth=1, alpha=0.7, color=color)
    plt.xticks(np.arange(0, new_combined_df.shape[0], step=step),np.arange(0, new_combined_df.shape[0] , step=step))
    

plt.xlabel('Sample points(30s)')
plt.ylabel('Score')

plt.show()
# In[累积和]
def cusum_alarm_detection(new_combined_df, output_dir="C:/Users/lenovo/Desktop/github", my=0.5, lalarm=60):
    """
    对数据框的每一列进行CUSUM报警检测，并为触发报警的列生成图形。
    
    参数:
    new_combined_df (DataFrame): 输入的数据框
    output_dir (str): 保存图形的目录
    my (float): 信号均值
    lalarm (float): 警报阈值
    
    返回:
    dict: 包含诊断结果的字典
    """
    # 创建保存图形的目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置图表参数
    params = {'font.family': 'serif',
              'font.serif': 'Times New Roman',
              'font.style': 'normal',
              'font.weight': 'normal',
              'font.size': 12,
             }
    rcParams.update(params)
    
    # 初始化结果字典
    results = {
        "alarm_columns": [],
        "alarm_details": {},
        "total_columns": new_combined_df.shape[1],
        "alarm_count": 0
    }
    
    # 遍历每一列
    for col_idx in range(new_combined_df.shape[1]):
        # 提取当前列数据
        try:
            col_name = new_combined_df.columns[col_idx]
        except:
            col_name = f"Column_{col_idx}"
            
        new_df = new_combined_df.iloc[:, col_idx].to_frame()
        new_df_1 = new_df.values
        y = new_df_1
        
        # 初始化CUSUM参数
        minn = 0.5  # 累积和最小值初始化
        summ = 0    # 累积和初始化
        sumsave = np.zeros(len(y))
        alarm = np.zeros(len(y))
        
        # 计算CUSUM算法
        for i in range(len(y)):
            summ = summ + (y[i] - my)  # 累积和
            sumsave[i] = summ
            if sumsave[i] < minn:
                minn = sumsave[i]  # 寻找最小误差
            d = sumsave[i] - minn  # 计算其他累积和与最小累积和的差d
            if d >= lalarm or (i > 0 and alarm[i-1] == 1):
                alarm[i] = 1
            else:
                alarm[i] = 0
        
        # 检查是否存在报警
        if np.any(alarm == 1):
            # 记录报警信息
            results["alarm_columns"].append(col_idx)
            results["alarm_count"] += 1
            
            # 找出报警开始的点
            alarm_start_points = np.where(np.diff(np.append(0, alarm)) == 1)[0]
            
            # 保存详细信息
            results["alarm_details"][col_idx] = {
                "column_name": str(col_name),  # 确保列名是字符串
                "alarm_start_points": alarm_start_points.tolist(),
                "alarm_count": len(alarm_start_points)
            }
            
            # 创建图形
            plt.figure(figsize=(4.5, 4), dpi=600)
            plt.subplots_adjust(wspace=0.30, hspace=0.45)
            
            # 第一个子图：原始数据
            ax1 = plt.subplot(3, 1, 1)
            plt.plot(new_df, color='#0000FF')
            plt.ylabel('Score')
            # 确保列名是字符串
            column_title = str(col_name)
            plt.title(f'{column_title}')
            ax1.set_xticklabels([])
            
            # 第二个子图：累积和
            ax2 = plt.subplot(3, 1, 2)
            plt.plot(sumsave, color='#FF00FF')
            plt.ylabel('Cumulative sum')
            plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax2.set_xticklabels([])
            
            # 第三个子图：报警状态
            plt.subplot(3, 1, 3)
            plt.plot(alarm, '#FF0000')
            plt.xlabel('Sample points(30s)')
            plt.ylabel('Alarm')
            plt.xticks(np.arange(0, len(y), step=max(1, len(y)//5)), 
                      np.arange(0, len(y), step=max(1, len(y)//5)))
            
            # 保存图形 - 使用安全的文件名
            safe_column_title = str(column_title).replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = f"{output_dir}/alarm_col_{col_idx}_{safe_column_title}.png"
            plt.savefig(filename)
            plt.close()
    
    # 生成总结报告
    if results["alarm_count"] > 0:
        summary = f"检测到 {results['alarm_count']} 列中存在报警信号，占总列数的 {results['alarm_count']/results['total_columns']*100:.2f}%"
        results["summary"] = summary
        print(summary)
        
        # 生成详细报告
        print("\n详细报警信息:")
        for col_idx in results["alarm_columns"]:
            details = results["alarm_details"][col_idx]
            print(f"cell {col_idx+1}: 检测到 {details['alarm_count']} 次报警")
            print(f"  报警开始点: {details['alarm_start_points']}")
    else:
        summary = "所有列均未检测到报警信号"
        results["summary"] = summary
        print(summary)
    
    return results

# 使用示例:
results = cusum_alarm_detection(new_combined_df)
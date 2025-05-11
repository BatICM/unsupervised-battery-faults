# -*- coding: utf-8 -*-
"""
Created on Sun May 11 19:49:19 2025

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
# In[数据预处理]
def preprocess_car1(data):
    # # 读取文本文件
    # data=pd.read_csv(r'C:\Users\lenovo\Desktop\github\data\car1.csv')
    # 提取电池电压数据列，将其转换为 DataFrame
    data = data.sort_values(by='starttime')
    data = data.set_index('starttime').reset_index()
    # -------------------------------- 单体数据提取
    # save_path='C:\\Battery_Safety_Evaluation\\CAERI'
    battery_voltage_series = data['chan24_vehbmscellvolt']
    # 将每一行的数据转换为浮点数并存储在列表中
    X = []
    for row in battery_voltage_series:
        row = row.replace('[', '').replace(']', '')
        values = [float(value) for value in row.split(',')]
        X.append(values)
    # 将列表列表转换为 NumPy 数组
    X = np.array(X)
    # 创建包含电池电压数据的 DataFrame
    cell_volt = pd.DataFrame(X)
    
    
    # -------------------------------- 5.电压异常值处理
    # 删除大量空缺值：删除df中电压空值个数超过5的行
    rows_null = cell_volt.isnull().sum(axis=1)
    find_index = rows_null[rows_null > 5].index.tolist()
    cell_volt = cell_volt.drop(find_index)
    # 异常值替换：行平均值替代>4.5或<0.5的电压异常值
    vlot_ub = 4.5
    vlot_lb = 3.0
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
    feature_matrix = pd.DataFrame(feature_matrix)
    # cell_volt_normized = min_max_normalize_columns(cell_volt)
    cell_volt_normized = cell_volt
    return cell_volt_normized,feature_matrix
def preprocess_car2(data):
    # data=pd.read_csv(r'C:\Users\lenovo\Desktop\github\data\car2.csv')
    
    data=data[data.BMSBatteryVoltage>100]
    if np.mean(data.BMSCellVoltageMin)>1000:
        data=data[data.BMSCellVoltageMin>1000]  
    else:
        data=data[data.BMSCellVoltageMin>1]
        data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)# 重排以获得正确时间序列
    for i in range(len(data)):
        data.tboxTime[i]=dt.datetime.fromtimestamp(data.tboxTime[i])
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
def preprocess_car3(data):
    # data=pd.read_csv(r'C:\Users\lenovo\Desktop\github\data\car3.csv')
    data=data[data.BMSBatteryVoltage>100]
    if np.mean(data.BMSCellVoltageMin)>1000:
        data=data[data.BMSCellVoltageMin>1000]  
    else:
        data=data[data.BMSCellVoltageMin>1]
        data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)# 重排以获得正确时间序列
    for i in range(len(data)):
        data.tboxTime[i]=dt.datetime.fromtimestamp(data.tboxTime[i])
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
    cell_volt=cell_volt/1000
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
def preprocess_car4(data):
    # data=pd.read_csv(r'C:\Users\lenovo\Desktop\github\data\car4.csv')
    data=data[data.BMSBatteryVoltage>100]
    if np.mean(data.BMSCellVoltageMin)>1000:
        data=data[data.BMSCellVoltageMin>1000]  
    else:
        data=data[data.BMSCellVoltageMin>1]
        data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)# 重排以获得正确时间序列
    for i in range(len(data)):
        data.tboxTime[i]=dt.datetime.fromtimestamp(data.tboxTime[i])
    data = data.sort_values(by='tboxTime')
    # data = data.tail(60000)
    data.reset_index(drop=True, inplace=True)# 重排以获得正确时间序列
    # -------------------------------- 单体数据提取
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
    cell_volt = cell_volt / 1000
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
    feature_matrix = pd.DataFrame(feature_matrix)
    cell_volt_normized = cell_volt
    return cell_volt_normized,feature_matrix
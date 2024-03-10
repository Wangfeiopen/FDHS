import copy
import os.path
import random
from decimal import Decimal
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def FI(data, label):
    model = RandomForestClassifier()  
    model.fit(data, label)
    Feature_importance = model.feature_importances_
    return Feature_importance

def PCA_Va(data, label):
    pca = PCA()
    pca.fit(data)
    PCA_contribution = pca.explained_variance_ratio_
    return PCA_contribution


def Interval_threshold_calculation(feature_data, feature_importance):
    ITC = []
    for i in range(len(feature_data)):
        ITC.append(round(round(np.std(feature_data[i]), 4) * round(feature_importance[i], 4), 4))
    return ITC


def density_computing(data, label, Feature_importance, band_num, Pca_v):
    majority_data = []
    majority_label = []
    minority_data = []
    minority_label = []
    new_majority = []  
    for i in range(len(data)):  
        if label[i] == 0:
            majority_label.append(label[i])
            majority_data.append(data[i])
        else:
            minority_label.append(label[i])
            minority_data.append(data[i])
    feature_num = len(np.transpose(data))  
    Maj_count = len(majority_data)  
    Min_count = len(minority_data) 
    IR = Maj_count / Min_count  
    Under_Weight = np.zeros((Maj_count, feature_num))  
    undersampling_UW = []  
    undersampling_list = [] 
    oversampling_num = int((Maj_count - Min_count) / 2)
    undersampling_num = int((Maj_count - Min_count) / 2)
    density_Maj_list = []  
    density_Min_list = []
    oversampling_bins = [] 
    feature_pool = []  
    for i in range(len(data[0])):
        density_Maj_list.append(list())
        density_Min_list.append(list())
        oversampling_bins.append(list())
        feature_pool.append(list())
    min_transpose = np.transpose(minority_data)  
    maj_transpose = np.transpose(majority_data)
    Itc = Interval_threshold_calculation(np.transpose(data), Pca_v)
    for f_index in range(feature_num):  
        bins = []
        if Itc[f_index] != 0:
            temp_num = 0
            bins.append(temp_num)
            while (temp_num < 1):
                temp_num = Decimal(str(temp_num)) + Decimal(str(Itc[f_index]))
                bins.append(float(temp_num))
            bins[-1] = 1
        else:
            bins = np.linspace(0, 1, 100)  
        # print('bins:', bins)
        hist_Min, _ = np.histogram(min_transpose[f_index], bins) 
        hist_Maj, _ = np.histogram(maj_transpose[f_index], bins)
        for i in range(len(bins) - 1):
            temp_min = []  # 临时列表
            temp_maj = []
            temp_min.append(bins[i])
            temp_min.append(bins[i + 1])
            temp_maj.append(bins[i])
            temp_maj.append(bins[i + 1])
            temp_min.append(hist_Min[i])
            temp_maj.append(hist_Maj[i])
            density_Min_list[f_index].append(temp_min)
            density_Maj_list[f_index].append(temp_maj)
            if hist_Min[i] != 0 and hist_Maj[i] / hist_Min[i] <= IR:
                oversampling_bins[f_index].append(temp_min)
            for j in range(len(Under_Weight)): 
                if bins[i] <= majority_data[j][f_index] < bins[i + 1]:
                    Under_Weight[j][f_index] = (hist_Maj[i] / hist_Min[i]) * Feature_importance[f_index] if (
                                hist_Min[i] != 0) else 1
                elif majority_data[j][f_index] == bins[len(bins) - 2] and i == len(bins) - 2:
                    Under_Weight[j][f_index] = (hist_Maj[i] / hist_Min[i]) * Feature_importance[f_index] if (
                                hist_Min[i] != 0) else 1
        f_sum = 0 
        f_tail = []  
        for i in range(len(oversampling_bins[f_index])):
            f_sum = f_sum + oversampling_bins[f_index][i][2]
            left = (np.array(min_transpose[f_index]) < oversampling_bins[f_index][i][0]).sum()
            right = left + oversampling_bins[f_index][i][2]
            f_tail.append(0)
            for f in min_transpose[f_index][left:right]:  
                x = str(f)
                dot_position = x.find('.')
                f_tail[i] += len(x[dot_position + 1:])
            f_tail[i] = int(round(f_tail[i] / (oversampling_bins[f_index][i][2]), 0))
        # print(f_sum)
        temp_oversampling = []

        for i in range(len(oversampling_bins[f_index])):  
            rate = int(round((oversampling_bins[f_index][i][2] / f_sum) * oversampling_num + 1, 0))
            miu = np.mean([oversampling_bins[f_index][i][0], oversampling_bins[f_index][i][1]])  
            sigma = ((oversampling_bins[f_index][i][1] - miu) * (oversampling_bins[f_index][i][1] - miu))
            for j in range(rate):
                generate = np.random.normal(miu, sigma, 1)
                temp_oversampling.append(generate[0])
        feature_pool[f_index] += list(temp_oversampling)
    for maji in range(len(Under_Weight)):
        temp = []
        temp.append(np.mean(Under_Weight[maji]))
        temp.append(maji)
        undersampling_UW.append(tuple(temp))
    undersampling_UW = sorted(undersampling_UW, key=lambda x: x[0], reverse=False)
    for i in range(undersampling_num):
        undersampling_list.append(undersampling_UW[i][1])
    for index in range(len(majority_data)):
        if index not in undersampling_list:
            new_majority.append(majority_data[index])
    return feature_pool, new_majority


def fit(data, label):
    data_transpose = np.transpose(data)  
    Feature_num = len(data_transpose)  
    Feature_importance = FI(data, label)  
    Pca_v = PCA_Va(data, label)
    Oversampling_data = [] 
    majority_data = []
    majority_label = []
    minority_data = []
    minority_label = []
    for i in range(len(data)):  
        if label[i] == 0:
            majority_label.append(label[i])
            majority_data.append(data[i])
        else:
            minority_label.append(label[i])
            minority_data.append(data[i])
    Maj_count = len(majority_data) 
    Min_count = len(minority_data)
    Feature_pool, new_majority = density_computing(data, label, Feature_importance, 20, Pca_v) 
    for i in range(int((Maj_count - Min_count) / 2)):
        temp_list = list()
        for feature_index in range(Feature_num):
            max_index = len(Feature_pool[feature_index])
            temp_list.append(Feature_pool[feature_index].pop(random.randint(0, max_index - 1)))
        Oversampling_data.append(temp_list)
    new_minority_data = np.vstack((minority_data, np.array(Oversampling_data)))
    new_minority_label = (np.ones(len(new_minority_data), dtype=int))
    data = np.vstack((new_majority, new_minority_data))
    label = np.hstack(((np.zeros(len(new_majority), dtype=int)), new_minority_label))
    return data, label

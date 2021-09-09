# -*- coding : utf-8 -*-
# coding: utf-8

import pandas as pd
from logitboost import LogitBoost
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score

from pylab import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 求ROC曲线均值
def get_mean(fpr_arr, tpr_arr):
    n = len(fpr_arr)
    # 拼接fpr数组，并去除重复值
    all_fpr = np.unique(np.concatenate([fpr_arr[i] for i in range(n)]))
    # 再用这些点对ROC曲线进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += interp(all_fpr, fpr_arr[i], tpr_arr[i])
    # 最后求平均值
    mean_tpr /= n
    x = all_fpr
    y = mean_tpr
    return x, y


# 读取excel数据
label_all = pd.read_excel('label_new.xlsx', dtype=np.int)
label_list = label_all.columns

# label_select = ['死亡', '医疗相关事件', '血管再通', '心脏相关事件', '心源死亡+其他死亡+心绞痛+心梗']
label_select = ['死亡']  # 选择待预测的label
data_select = ['001/avg_死亡_001.xlsx']  # 数据集
data_name = ['ML']

# 产生两个1行10列，元素值全为0的数组
auc_array = np.zeros(10)
acc_array = np.zeros(10)

plt.figure()
lw = 2

label = label_all[label_select]
for ds, dn in zip(data_select, data_name):
    fpr_list = []
    tpr_list = []
    auc_list = []

    # 数据预处理，将“性别”属性二值化
    data = pd.read_excel(ds)
    if ds == 'CTA_data35.xlsx':
        class_mapping = {'男': 0, '女': 1}
        data['Sex (M, F)'] = data['Sex (M, F)'].map(class_mapping)
    data = data.fillna(0)

    # 循环运行100次，取平均值
    for times in range(100):
        # 搭建LogitBoost模型
        model = LogitBoost()
        fpr_list_temp = []
        tpr_list_temp = []

        '''
        # 划分数据集，简单交叉验证，测试集0.3，训练集0.7
        for i in range(10):
            # 模型训练与测试
            X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.3, stratify=label)
            model.fit(X_train, Y_train)
            y_pre = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
        '''

        # 划分数据集，3折交叉验证
        kf = KFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(data):
            # 模型训练与测试
            X_train, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
            y_train, y_test = label.iloc[train_index, :], label.iloc[test_index, :]
            model.fit(X_train, y_train)
            y_pre = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            # 输出预测结果
            print("===================")
            print(y_prob)
            print("===================")
            print(classification_report(y_test, y_pre))
            # 计算分类准确率
            acc = accuracy_score(y_test, y_pre)

            # 计算fpr、tpr用于绘制ROC曲线
            fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 0], pos_label=0)
            fpr_list_temp.append(list(fpr))
            tpr_list_temp.append(list(tpr))

        # 取平均值，保存一次循环中的计算结果
        fpr_mean_temp, tpr_mean_temp = get_mean(fpr_list_temp, tpr_list_temp)
        fpr_list.append(list(fpr_mean_temp))
        tpr_list.append(list(tpr_mean_temp))

        # 计算每一次循环后的ROC曲线的auc值
        auc_sorce_temp = auc(fpr_mean_temp, tpr_mean_temp)
        auc_list.append(auc_sorce_temp)

    # 计算100次循环后的AUC的平均值和方差
    if times == 99:
        fpr_mean, tpr_mean = get_mean(fpr_list, tpr_list)

        auc_score = auc(fpr_mean, tpr_mean)
        auc_arr_std = np.std(auc_list, ddof=1)

        plt.plot(fpr_mean, tpr_mean, lw=lw, label='%s auc=%0.3f std=%0.3f' % (dn, auc_score, auc_arr_std))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

# 绘制ROC图像
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(label_select[0])
plt.legend(loc="lower right")
plt.show()

result = np.zeros(shape=(10, 2))
result[:, 0] = acc_array
result[:, 1] = auc_array
result = pd.DataFrame(result, columns=['acc', 'auc'])

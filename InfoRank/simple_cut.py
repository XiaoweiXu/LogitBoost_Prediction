import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 计算信息熵
def calc_ent(x):
    """
        calculate entropy of x
    """
    x_count = pd.value_counts(x)
    p = x_count.values / x.shape[0]
    ent = -np.sum(p * np.log2(p))
    return ent


# 计算条件熵
def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
    return ent


# 计算信息增益
def calc_ent_grap(x, y):
    """
        calculate ent grap
    """
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


# 等距分箱
def equal_d_cut(x, q):
    x_max = np.max(x)
    x_min = np.min(x)
    d = (x_max - x_min) / q
    bins = x_min + np.arange(q + 1) * d
    res = pd.cut(x, bins)
    return res


if __name__ == "__main__":

    # 等频分箱和等距分箱
    # 读取excel数据
    data = pd.read_excel('CTA_data35.xlsx')
    label_all = pd.read_excel('label_new.xlsx', dtype=np.int)

    # 数据预处理，将“性别”属性二值化
    class_mapping = {'男': 0, '女': 1}
    data['Sex (M, F)'] = data['Sex (M, F)'].map(class_mapping)
    features = data.columns
    data = data.fillna(0)

    # 标签选择
    data['label'] = label_all['死亡'].values
    label = '死亡'
    data_change = data.copy()

    res = []

    # 筛选连续变量
    '''
    column_list = ['LVM (g)', 'LVM index (g/m)', 'EF (%)', 'LVED volume (mL)',
                   'LVES volume (mL)', 'Stroke volume (mL)', 'CCS (Agatston units)']
    '''

    '''
    column_list = ['冠脉周围脂肪体积', '冠脉周围脂肪密度', '冠脉周围脂肪密度标准差',
                   '心包脂肪体积', '心包脂肪密度', '心包脂肪密度标准差',
                   'LVM (g)', 'LVM index (g/m)', 'EF (%)', 'LVED volume (mL)',
                   'LVES volume (mL)', 'Stroke volume (mL)', 'CCS (Agatston units)']
    '''

    '''
    column_list = ['CCS (Agatston units)']
    '''

    column_list = ['冠脉周围脂肪体积', '冠脉周围脂肪密度', '冠脉周围脂肪密度标准差',
                   '心包脂肪体积', '心包脂肪密度', '心包脂肪密度标准差', 'CCS (Agatston units)']

    # 对连续变量进行等频分箱和等距分箱
    for ii, i in enumerate(features):
        for a in column_list:
            # 等频分箱
            name = "等频分箱"
            mid = pd.qcut(data[a].values, q=5, duplicates='drop')
            data_change[a] = mid.codes

            # 等距分箱
            '''name = "等距分箱"
            bins = equal_d_cut(data_change[a].values, q=5)
            data_change[a] = bins.codes'''

        # 计算信息增益值
        res.append(calc_ent_grap(data_change[i].values, data_change['label'].values))
    # 保存信息增益结果
    res = np.array(res)

    # 按信息增益由高到低排序
    index = np.argsort(-res)
    x_sort = res[index]

    # 绘制图像
    num_features = 35
    importances = res
    indices = index[0:35]
    plt.figure(figsize=(20, 10))
    plt.title(name)
    plt.bar(range(num_features), importances[indices], color="g", align="center")
    plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
    plt.xlim([-1, num_features])
    plt.savefig('results_35/' + name + "_" + label + '_result.png')
    plt.show()

    # 保存结果
    r_f, r_i = features[index], importances[index]
    record = pd.DataFrame()
    record['属性'] = r_f
    record['信息增益值'] = r_i
    record.to_csv('results_35/' + name + "_" + label + '.csv')

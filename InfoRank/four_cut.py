import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

from codes.discretization import chiMerge

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 决策树分箱边界值
def optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(nan).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=6,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续group_by操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]

    return boundary


'''
# 独热编码
def convert_one_hot(y, N):
    one_hot = np.zeros(shape=(y.shape[0], N), dtype=np.int32)
    for i in range(N):
        one_hot[y == i + 1, i] = 1
    return one_hot
'''


# 计算信息熵
def calculate_ent(x):
    x_counter = Counter(x)
    x_u = np.unique(x)
    p_x = []
    for i in range(x_u.shape[0]):
        p_x.append(x_counter[x_u[i]] / x.shape[0])
    p_x = np.array(p_x)
    return -np.sum(p_x * np.log2(p_x))


# 计算在x的条件下y出现的概率
def calculate_x_y(x, y):
    x_counter = Counter(x)
    x_u = np.unique(x)
    res = []
    for i in range(x_u.shape[0]):
        y_x = y[x == x_u[i]]
        y_x_c = Counter(y_x)
        y_u = np.unique(y_x)
        value = []
        for j in range(y_u.shape[0]):
            p_y = float(y_x_c[y_u[j]]) / x_counter[x_u[i]]
            value.append(p_y * np.log2(p_y))
        value = np.array(value)
        p_x = x_counter[x_u[i]] / x.shape[0]
        res.append(-p_x * np.sum(value))
    v = np.sum(np.array(res))
    return v


# 计算Gini系数（条件熵）
def calculate_gini(x, y):
    x1 = calculate_ent(y)
    x2 = calculate_x_y(x, y)
    return x1 - x2


if __name__ == "__main__":

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

    name_list = ['chiMerge', 'decision_tree']

    # 对连续变量进行决策树分箱和卡方分箱
    for name in name_list:
        # 读取excel数据
        data = pd.read_excel('CTA_data35.xlsx')
        label_all = pd.read_excel('label_new.xlsx', dtype=np.int)

        # 数据预处理，将“性别”属性二值化
        class_mapping = {'男': 0, '女': 1}
        data['Sex (M, F)'] = data['Sex (M, F)'].map(class_mapping)
        features = data.columns
        data = data.fillna(0)

        # 选择标签
        data['label'] = label_all['心脏相关事件'].values
        data_change = data.copy()
        label = '心脏相关事件'

        # 数据分箱
        res = []
        for a in column_list:
            group = []
            if name == 'decision_tree':
                title = '决策树分箱'
                group = optimal_binning_boundary(data[a], data['label'])
            elif name == 'chiMerge':
                title = '卡方分箱'
                model = chiMerge(max_interval=5, feature_type=0)
                group = model.dsct_pipeline(data, a, 'label')
            cats = pd.cut(data[a].values, group, right=True)
            data_change[a] = cats.codes

        # 计算信息增益，将结果保存为np数组
        for i in features:
            res.append(calculate_gini(data_change[i].values, data_change['label'].values))
        res = np.array(res)

        # 按信息增益由高到低排序
        index = np.argsort(-res)
        x_sort = res[index]

        # 绘制图像
        num_features = len(index)
        importances = res
        indices = index
        plt.figure(figsize=(20, 10))
        plt.title(title)
        plt.bar(range(num_features), importances[indices], color="g", align="center")
        plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
        plt.xlim([-1, num_features])
        plt.savefig('results_35/' + title + "_" + label + '_result.png')
        plt.show()

        # 保存实验结果
        r_f, r_i = features[index], importances[index]
        record = pd.DataFrame()
        record['属性'] = r_f
        record['信息增益值'] = r_i
        record.to_csv('results_35/' + title + "_" + label + '.csv')

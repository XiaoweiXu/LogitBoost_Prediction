import pandas as pd

# 计算四种分箱下的信息增益平均值
file = pd.read_excel('InfoRank.xlsx')
data = pd.DataFrame(file)

result = data.groupby('indents').mean().reset_index()

result = result.sort_values(ascending=False, by='values')
result.to_csv('MeanInfoRank.csv')

csv = pd.read_csv('MeanInfoRank.csv', encoding='utf-8')
csv.to_excel('MeanInfoRank.xlsx')

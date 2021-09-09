import pandas as pd
import openpyxl


def csv_to_xlsx():
    csv1 = pd.read_csv('results_35/等频分箱_死亡.csv', encoding='utf-8')
    csv1.to_excel('results_35/等频分箱_死亡.xlsx')


if __name__ == '__main__':
    csv_to_xlsx()

import pandas as pd
import numpy as np
import numbers
import csv

data = pd.read_csv("D:/学校/数据挖掘/mashroom dataset/data/data/data_correct/secondary_data_no_miss_correct.csv")
data_use = data.copy()
for column in data.columns:
    column_data = data[column]
    if isinstance(column_data[0], numbers.Number):
        continue
    dict = {}
    num = 0
    for i in range(len(column_data)):
        cell = column_data[i]
        if (cell not in dict.keys()):
            dict[cell] = num
            num = num + 1
        data_use.loc[i,column] = dict.get(cell)
    print(dict)
    dict.clear()
print(data_use)
csvFile = open("D:/学校/数据挖掘/mashroom dataset/data/data/data_correct/secondary_data_no_miss_correct_number_type.csv", "w")            #创建csv文件
writer = csv.writer(csvFile)
writer.writerow(data_use.columns)
for i in range(0,len(data_use)):
    writer.writerow(data_use.iloc[i])
csvFile.close()
import pandas as pd
import numpy as np
import numbers
import csv

secondary_data = pd.read_csv("D:/学校/数据挖掘/mashroom dataset/data/data/data_correct/secondary_data_no_miss_correct_same_feature_type.csv")
old_data = pd.read_csv("D:/学校/数据挖掘/mashroom dataset/data/data/data_correct/1987_data_no_miss_correct_same_feature_type.csv")
new_data_use = secondary_data.copy()
old_data_use = old_data.copy()
for column in secondary_data.columns:
    new_column_data = secondary_data[column]
    old_column_data = old_data[column]
    if isinstance(new_column_data[0], numbers.Number):
        continue
    dict = {}
    num = 0
    for i in range(len(new_column_data)):
        cell = new_column_data[i]
        if (cell not in dict.keys()):
            dict[cell] = num
            num = num + 1
        new_data_use.loc[i,column] = dict.get(cell)
    for i in range(len(old_column_data)):
        cell = old_column_data[i]
        if (cell not in dict.keys()):
            dict[cell] = num
            num = num + 1
        old_data_use.loc[i,column] = dict.get(cell)
    print(dict)
    dict.clear()
print(new_data_use)
csvFile = open("D:/学校/数据挖掘/mashroom dataset/data/data/data_correct/secondary_data_no_miss_correct_same_feature_number_type.csv", "w")            #创建csv文件
writer = csv.writer(csvFile)
writer.writerow(new_data_use.columns)
for i in range(0,len(new_data_use)):
    writer.writerow(new_data_use.iloc[i])
csvFile.close()
print(old_data_use)
csvFile = open("D:/学校/数据挖掘/mashroom dataset/data/data/data_correct/1987_data_no_miss_correct_same_feature_number_type.csv", "w")            #创建csv文件
writer = csv.writer(csvFile)
writer.writerow(old_data_use.columns)
for i in range(0,len(old_data_use)):
    writer.writerow(old_data_use.iloc[i])
csvFile.close()


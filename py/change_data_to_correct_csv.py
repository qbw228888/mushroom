import pandas as pd
import csv


data = pd.read_csv('D:/学校/数据挖掘/mashroom dataset/data/data/1987_data_no_miss.csv')
columns = data.columns[0]
csvFile = open("D:/学校/数据挖掘/mashroom dataset/data/data/data_correct/1987_data_no_miss_correct.csv", "w")            #创建csv文件
writer = csv.writer(csvFile)                  #创建写的对象
#先写入columns_name                             
writer.writerow(columns.split(";"))     #写入列的名称
#写入多行用writerows
for i in range(0, len(data)):
    print(data.iloc[i,0])
    writer.writerow(data.iloc[i,0].split(";"))
csvFile.close()
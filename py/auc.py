import numpy as np
from sklearn.metrics import roc_auc_score
def auc_calculate(labels,probs):
    N = 0           # 正样本数量
    P = 0           # 负样本数量
    neg_prob = []   # 负样本的预测值
    pos_prob = []   # 正样本的预测值
    for index, label in enumerate(labels):
        if label == 1:
            # 正样本数++
            P += 1
            # 把其对应的预测值加到“正样本预测值”列表中
            pos_prob.append(probs[index])
        else:
            # 负样本数++
            N += 1
            # 把其对应的预测值加到“负样本预测值”列表中
            neg_prob.append(probs[index])
    number = 0
    # 遍历正负样本间的两两组合
    for pos in pos_prob:
        for neg in neg_prob:
            # 如果正样本预测值>负样本预测值，正序对数+1
            if (pos > neg):
                number += 1
            # 如果正样本预测值==负样本预测值，算0.5个正序对
            elif (pos == neg):
                number += 0.5
    return number / (N * P)
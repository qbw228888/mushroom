#读数据
import pandas as pd
import numpy as np
train_data = pd.read_csv("D:\学校\数据挖掘\mashroom dataset\data\data\data_correct/1987_data_no_miss_correct_number_type.csv")
x = train_data.iloc[0:,1:].values
y = train_data.iloc[0:,0].tolist()


#分割数据集为测试机和训练集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

#进行标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# 进行多种方法进行分类
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from auc import auc_calculate
# from picture import plot_decision_regions
# import matplotlib.pyplot as plt
def train (classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    # x_combined_std = np.vstack((x_train, x_test))
    # y_combined = np.hstack((y_train, y_test))
    # plot_decision_regions(x_combined_std, y_combined, classifier=classifier)
    # plt.xlabel('petal length [standardized]')
    # plt.ylabel('petal width [standardized]')
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # plt.show()
    #计算准确率
    print('Accuracy: %.5f' % accuracy_score(y_test, y_pred))
    #绘制roc曲线
    try:
        fpr,tpr, thresholds = roc_curve(y_test,classifier.predict_proba(x_test)[:,1])
        plt.plot(fpr,tpr,label='ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        #计算auc
        print('AUC: %.5f' % auc_calculate(y_test,y_pred))
        plt.show()
    except BaseException:
        print("error in draw auc")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = MLPClassifier()
clf2 = DecisionTreeClassifier()
clf3 = KNeighborsClassifier()
#将上面三个基模型集成
eclf = VotingClassifier(
    estimators=[('net', clf1), ('tree', clf2), ('knn', clf3)],
    voting='hard')


train(SVC(kernel='linear'), x_train_std, y_train, x_test_std, y_test)


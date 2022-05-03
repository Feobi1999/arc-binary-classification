# import pandas as pd
'''前面的库要安装 例如 pip install csv 还不会的话可以百度一下'''
import os
import csv
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''获取有电弧的所有数据'''
listDir=os.listdir('yes')#获取当前目录下的所有内容
total_yes_data=[]
total_yes_label=[]
for x in listDir:
    data_cum2=[]
    #加载每条数据
    with open('yes/'+x, 'r') as f:
        reader = csv.reader(f)
        for index,i in enumerate(reader):
            #从第15行开始才有数据
            if index>15 and len(i)==3:
                data_cum2.append(float(i[1]))

    total_yes_data.append(data_cum2)
    #标签是1
    total_yes_label.append(1)

listDir=os.listdir('no')#获取当前目录下的所有内容
total_no_data =[]
total_no_label=[]
'''获取没有电弧的所有数据'''
for x in listDir:
    data_cum2=[]
    with open('no/'+x, 'r') as f:
        reader = csv.reader(f)
        for index,i in enumerate(reader):
            if index>15 and len(i)==3:
                data_cum2.append(float(i[1]))
    total_no_data.append(data_cum2)
    total_no_label.append(0)  #标签是0 其他的和上面一样

#80% 训练 20%测试  数据划分
pos_train, pos_X_test, pos_y_train, pos_y_test = total_yes_data[:80], total_yes_data[80:], total_yes_label[:80], total_yes_label[80:]
neg_train, neg_X_test, neg_y_train, neg_y_test = total_no_data[:80], total_no_data[80:], total_no_label[:80], total_no_label[80:]

#合并有电弧和没电弧的
x_train = pos_train + neg_train
y_train = pos_y_train + neg_y_train
x_test = pos_X_test + neg_X_test
y_test = pos_y_test + neg_y_test

#打乱数据顺序
shuffle_index = np.random.permutation(160)
shuffle_index_test = np.random.permutation(41)
x_train = np.array(x_train)[shuffle_index, :]
y_train = np.array(y_train)[shuffle_index]



x_test = np.array(x_test)[shuffle_index_test, :]
y_test = np.array(y_test)[shuffle_index_test]




print("==============SVM classifier===========================")

sgd_clf = SGDClassifier(tol=-np.infty, random_state=42) #构建模型
sgd_clf.fit(x_train, y_train) #训练
y_pred = sgd_clf.predict(x_test)
predictions_train = sgd_clf.predict(x_train)
#打印准确度
print("ACC of SVM train train",accuracy_score(y_train,predictions_train))
print("ACC of SVM test Classifier:",accuracy_score(y_test, y_pred))



print("==============logistic classifier===========================")

clf = LogisticRegression(penalty='l2',max_iter=1000)
clf.fit(x_train,y_train)
predictions = clf.predict(x_test)
predictions_train = clf.predict(x_train)

#保存模型的代码 已经保存了所以注释掉
# with open('save/logistic_model.pickle','wb') as f:
#     pickle.dump(clf,f)


print("ACC of logistic regression test",accuracy_score(y_test,predictions))
print("ACC of logistic regression train",accuracy_score(y_train,predictions_train))

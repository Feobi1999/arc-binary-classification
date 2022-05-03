
import pickle
import numpy as np
import csv


'''load 模型  模型名称可以选择'''
# model_name = 'svm'
model_name ='logistic'
with open('save/%s_model.pickle'%model_name,'rb') as f:
    model = pickle.load(f)

data_sample=[]

''' 改变文件名称,可以手动修改'''
with open('yes/T0105CH1.CSV','r') as f:
    reader = csv.reader(f)
    for index,i in enumerate(reader):
        if index>15 and len(i)==3:
            data_sample.append(float(i[1]))

# 进行测试
pred = model.predict(np.array([data_sample]))
if pred==0:
    print('no')
else:
    print('yes')

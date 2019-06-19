# -- coding:UTF-8 --
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import sys
from sklearn.ensemble import IsolationForest
from datetime import datetime

def get_grafana_data(filename):
	'''
	data processing
	filename: csv file's filename included '.csv'
	'''
	data = []
	with open(filename) as f:
	    reader = csv.reader(f,delimiter=';')
	    arr = np.array(list(reader))   # reader -> list -> np array
	    arr = np.delete(arr,0,axis=0)
	    #处理时间
	    time_list = arr[...,0]
	    time_list = np.delete(time_list,0)
	    temp = []
	    timeformat = '%Y-%m-%dT%H:%M:%S+08:00'
	    for t in time_list:
	    	date = datetime.strptime(t, timeformat)
	    	t = time.mktime(date.timetuple())
	    	temp.append(t)
	    temp = np.array(temp)
	    #处理数据
	    data_len = arr.shape[1]-1
	    for i in range(data_len):
		    data_mid = arr[...,i+1]   # array[len, 2] -> array[len, 1]
		    data_mid = np.delete(data_mid,0)   # delete data[0]
		    data_mid = np.array(data_mid)   # list -> np array
		    data = np.hstack((data, data_mid))
	    sum_arr = np.hstack((temp.reshape(temp.shape[0],1), data.reshape(data.shape[0]//data_len,data_len)))
	    i = 0
	    (x,y) = sum_arr.shape
	    for j in range(y):
	    	i=0
	    	for arr in sum_arr:
	    		if(sum_arr[i][j] == 'null'):
	    			sum_arr = np.delete(sum_arr,i,axis=0)
	    		else:
	    			i = i + 1
	return sum_arr.astype(float)   # array[len, ](str) -> array[len, 1](float)

def fit(data, rng):
	'''
	Isolation Forest predict and plot
	data: any dimentions
	rng: np.random.randomState type
	'''
	#predict
	clf = IsolationForest(random_state=rng, contamination = 0.03)
	clf.fit(data)
	return clf

def predict_plot(data, clf):
	#predict
	pred_train = clf.predict(data)
	print('pred_train:{}\n'.format(pred_train))

	#plot
	ymin = np.min(np.delete(data,0,axis=1))
	ymax = np.max(np.delete(data,0,axis=1))
	
	(x,y) = data.shape
	time = data[...,0]
	plt.title("IsolationForest")
	plt.xlabel("time")
	plt.ylabel("node data")
	#画出各个数据的折线
	for i in range(1,y):
		plt.plot(time, data[...,i])
	#画出异常所在的时间的竖线
	for i in range(x):
		if pred_train[i] == -1:
			plt.vlines(data[i][0],ymin,(5/4)*ymax,colors = "c",linestyles = "dashed")
	plt.savefig("Isolation Forest.jpg")
	plt.show()


if __name__ == "__main__":
	rng = np.random.RandomState(100)
	trainDataFile = sys.argv[1] if sys.argv[1] else ''
	testDataFile = sys.argv[2] if sys.argv[2] else ''

	if trainDataFile=='' or testDataFile=='':
		print('please input filename')
		exit()
	#读取文件
	trainData = get_grafana_data(trainDataFile)
	testData = get_grafana_data(testDataFile)

	if len(trainData) != 0 and len(testData) != 0:
		clf = fit(trainData, rng)   #训练
		predict_plot(testData, clf)   #预测并画图

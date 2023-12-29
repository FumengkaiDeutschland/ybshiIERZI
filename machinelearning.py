import streamlit as st
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pymysql as MySQLdb
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import pandas as pd
import xlwt
import math
from pylab import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pymysql
import base64
from tensorflow.keras import datasets, layers, models
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras
from livelossplot import PlotLossesKeras
from time import sleep
from PIL import Image
def get_database(hostName, local, inputsize):  # hostName:'localhost'
    if hostName == 'Excel':
        url = local
        df = pd.read_excel(url, header = None)
        input  = df.iloc[:,0:inputsize]
        output = df.iloc[:, -1]
        X, X1, y, y1 = train_test_split( input, output, test_size=0.2)
    if hostName == 'CSV':
        url = local
        df = pd.read_csv(url, header = None)
        input  = df.iloc[:,0:inputsize]
        output = df.iloc[:, -1]
        X, X1, y, y1 = train_test_split( input, output, test_size=0.2)
    return X, X1, y, y1

def add_parameter_ui(model_name, X, y):
    if model_name =='XGBoost':
        import xgboost as xgb
        max_depth = st.sidebar.slider('最大深度', 1, 100)
        n_estimators = st.sidebar.slider('迭代次数', 1, 10000)
        m = xgb.XGBRegressor(max_depth=max_depth, learning_rate=1e-2, n_estimators=n_estimators,
                                    reg_lambda=0.1, random_state=9,
                                    min_child_weight=1, gamma=2)
    if model_name =='LigbtGBM':
        from lightgbm import LGBMRegressor
        max_depth = st.sidebar.slider('最大深度', 1, 100)
        n_estimators = st.sidebar.slider('迭代次数', 1, 10000)
        random_state = st.sidebar.slider('随机数', 1, 100)
        num_leaves = st.sidebar.slider('叶子数', 1, 100)
        alpha = 0.010
        m = LGBMRegressor(n_estimators=n_estimators,
                                  learning_rate=0.0001,
                                  max_depth=max_depth,
                                  max_bin=100,
                                  min_child_samples=100,
                                  random_state=random_state,
                                  objective='regression',
                                  alpha=alpha,
                                  num_leaves=num_leaves,
                                  force_row_wise=True,
                                  boosting_type='dart'
                                  )
    if model_name =='CATBoost':
        from catboost import CatBoostRegressor
        depth = st.sidebar.slider('模型深度', 1, 100)
        iterations = st.sidebar.slider('迭代次数', 1, 10000)
        leaf_estimation_iterations = st.sidebar.slider('子叶迭代数', 1, 100)
        reg_lambda = st.sidebar.slider('Reg Lamda', 1, 200)
        m = CatBoostRegressor(objective='RMSE',
                                     logging_level='Silent',
                                     random_seed=42,
                                     iterations=iterations,
                                     learning_rate=0.01,
                                     depth=depth,
                                     subsample=0.6,
                                     colsample_bylevel=0.1,
                                     min_data_in_leaf=20,
                                     bagging_temperature=0.01,
                                     leaf_estimation_iterations=leaf_estimation_iterations,
                                     reg_lambda=reg_lambda)

    return m
def modelTrain(X, y, m):
    ModelTrained = m.fit(X, y)
    return ModelTrained
    print('模型训练完成')
def app():
    st.write("""
            # 机器学习
            """)
    hostName = st.sidebar.selectbox(
        "选择数据类型",
        ( "Excel", "CSV"))
    machinelearning_name = st.sidebar.selectbox(
        'Select classifier',
        ('XGBoost','LigbtGBM', 'CATBoost'))
    inputsize = st.text_input("数据尺寸", 2)
    inputsize = int(inputsize)
    local = st.text_input('数据位置', 'C://Users//Lannister//OneDrive//桌面//ABC.xlsx')
    X, X1, y, y1 = get_database(hostName, local, inputsize)
    on = st.toggle('开始训练')
    model = add_parameter_ui(machinelearning_name, X, y)
    if on:
        st.write("模型开始训练")
        modelTrained = modelTrain(X, y, model)
        pre_model = modelTrained.predict(X1)
        clf()  # 清图。
        cla()  # 清坐标轴。
        fig = plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn')
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')
        x = np.linspace(0, max(y), 100)
        plt.xlim(min(y), max(y))
        plt.ylim(min(y), max(y))
        y_1 = 0.9*x
        y_2 = 1.1*x
        plt.figure(num=3, figsize=(8, 8))
        plt.scatter(y1, pre_model)
        plt.plot(x, y_1, c='k', linestyle='--')
        plt.plot(x, y_2, c='k',linestyle='--')
        plt.ylabel('Predict Value', fontsize=20)
        plt.xlabel('Real Value', fontsize=20)
        plt.legend(['10% Erro Line'], loc='upper right', fontsize=15)
        plt.savefig('C:\\Users\\Lannister\\OneDrive\\桌面\\web杨彬\\WbFig\\训练模型.png', dpi=400, bbox_inches='tight')
        imagePump = Image.open('C:\\Users\\Lannister\\OneDrive\\桌面\\web杨彬\\WbFig\\训练模型.png')
        st.image(imagePump)

    else:
        st.write("等待本模型训练")

    two = st.toggle('保存模型')
    if two:
        modellocal = st.text_input('模型保存', 'C://Users//Lannister//OneDrive//桌面')
        st.write('模型已保存')
    else:
        st.write('模型未保存')








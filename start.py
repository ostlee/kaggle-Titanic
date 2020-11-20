'''
Titanic 

1.Goal
It is your job to predict if a passenger survived the sinking of the Titanic or not. 
For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.
2.Metric
Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.
3.Dataset
train.csv -- including features and labels (1/0 -- survived/not )
test.csv

Titanic Data Science Solutions(notebook)
Workflow stages for kaggle competitions 
seven steps:
1. Question or problem definition. -- 问题定义
2. Acquire training and testing data. -- 获取数据
3. Wrangle, prepare, cleanse the data. -- 数据预处理
4. Analyze, identify patterns, and explore the data. -- 初步分析数据
5. Model, predict and solve the problem. -- 建模解决问题
6. Visualize, report, and present the problem solving steps and final solution. -- 通过可视化等方法总结问题和模型
7. Supply or submit the results.  -- 提交结果
'''

import os

# data analysis and wrangling 
import pandas as pd 
import numpy as np 
import random as rnd

# visualization 
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

dir_path = os.getcwd()
train_df = pd.read_csv(os.path.join(dir_path, '/dataset/train.csv'))
test_df = pd.read_csv(os.path.join(dir_path, '/dataset/test.csv'))
combine = [train_df, test_df]
print(train_df.head)
print()
println("123")

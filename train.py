import pandas as pd                        
import numpy as np                         
import datetime as dt                      
import statsmodels.api as sm               
import seaborn as sns                      
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from matplotlib.pyplot import figure
import re
import pydotplus
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
from os import walk
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
import joblib
from utils import preprocesDataFrame
from utils import plotConfusionMatrix
from utils import createCSV
from utils import plotFeatureImportance
from utils import createDataFrame
from utils import filterByIP
from utils import plotCorrelation
from utils import dimensionalityReduction
from bashUtils import createArgusFilesOutput
import constants
import json

variables=json.load(open(f'variables.json',)) 

scannerIps = variables["scannerIps"]
targetIps = variables["targetIps"]
trainDataPath = variables["trainingData"]

def categorizeFlows(df):
    es_escaneo = []
    counter = 0
    for row in df.itertuples():
        scan = 0
        for i in range (0, len(scannerIps)):
            if (row[22].find(scannerIps[i])>=0 and row[24].find(targetIps[i])>=0):
                scan = 1
                break;
        counter +=  scan
        es_escaneo.append(scan)
    print("data categorization DONE", counter)
    return es_escaneo


#createArgusFilesOutput(trainDataPath)
#createCSV(trainDataPath)
df=createDataFrame(trainDataPath)
df[constants.ESCANEO] = categorizeFlows(df)
df=preprocesDataFrame(df)
df=dimensionalityReduction(df)
df=df.sample(frac=1)


#plotCorrelation(df)

#### TRAINING THE MODEL  #####


train, test = train_test_split(df, test_size=0.2)
Y_train = train[constants.ESCANEO]
X_train = train.drop(columns=[constants.ESCANEO]);
Y_test = test[constants.ESCANEO]
X_test = test.drop(columns=[constants.ESCANEO]);


model = BaggingClassifier(random_state=0, n_estimators=50, oob_score=True)
model = model.fit(X_train,Y_train)

joblib.dump(model, 'bag.pkl')


#####  DISPLAY TREE #####
df = df.drop(columns=[constants.ESCANEO]);
print("Y column dropped")
joblib.dump(df.columns.tolist(), 'columns.txt')
print ("out of bag score",model.oob_score_)
print(model.estimators_)
data = tree.export_graphviz(model.estimators_[0], out_file=None, feature_names=df.dtypes.keys(), precision = 10)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

print("now plotting")
img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)


plotConfusionMatrix(model, X_test, Y_test)
plotFeatureImportance(model, X_train)
plt.show(); 



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
from os import walk
from sklearn import tree, preprocessing
import joblib
from utils import *
from bashUtils import *
import constants
import io
import sys
import json


variables = json.load(open(f'variables.json',));
demoDirPath = variables["demoData"];


def addMissingEncodedColumns(df):
    treeColumns = joblib.load('columns.txt')
    df = df.drop(columns=list(filter(lambda c: not (c in treeColumns), df.columns.tolist())))
    for c in treeColumns:
        if not (c in df):
            df[c] = 0
            
    df = df[treeColumns]
    return df


process = createArgusDaemonOutput(demoDirPath)
clf = joblib.load('bag.pkl')
print("Real time netflow");
while True:
    header=True
    for line in io.TextIOWrapper(process.stdout, encoding="utf-8"):
        if not header:
            df = getFlowDataFrame(line)
            output = '{:^22}'.format(df[constants.FECHA][0]) + '{:^10}'.format(df[constants.PROTO][0]) + '{:^35}'.format(df[constants.SRCADDR][0])+ '{:^35}'.format(df[constants.DSTADDR][0])+ '{:^5}'.format(df[constants.STATE][0])+ '{:^10}'.format(df[constants.SUM][0])+"\n"
            df = preprocesDataFrame(df)
            df = addMissingEncodedColumns(df)

            if clf.predict(df) == 1:
                sys.stdout.shell.write(output, "COMMENT")
            else:
                sys.stdout.shell.write(output, "STRING")
            
        else:
            header=False
            
    if input() == 'STOP':
        process.kill()
        break;




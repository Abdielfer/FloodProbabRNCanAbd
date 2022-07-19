import pandas as pd
import myServices 
import randForReg as rfr

def run():
    datasetPath = './sample.csv'
    rfReg  = rfr.implementRandomForestRegressor(datasetPath,'percentage', 0.2)
    x_train,x_validation,y_train, y_validation = rfReg.getSplitedDataset()
    rfr.printDataBalace(x_train, x_validation, y_train, y_validation,'percentage')
    bestEstimator = rfReg.fitRFRegressor()
    print(bestEstimator)

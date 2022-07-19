import pandas as pd
import myServices 
import randForReg as rfr

def main():
    datasetPath = './sample.csv'
    rfReg = rfr.implementRandomForestRegressor(datasetPath,'percentage', 0.2)
    x_train,x_validation,y_train, y_validation = rfReg.getSplitedDataset()
    rfr.printDataBalace(x_train, x_validation, y_train, y_validation,'percentage')
    bestEstimator = rfReg.fitRFRegressor()
    rfr.reportErrors(bestEstimator, x_validation, y_validation)


if __name__ == "__main__":
    main()
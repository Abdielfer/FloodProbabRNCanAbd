'''
Aqui vamos a poner 
todo lo necesario para hacer fincionet RF a ppartir de competition 2
'''
from sqlite3 import Date
import time

from torchmetrics import Precision
import myServices
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.calibration import CalibrationDisplay  ## TODO ##

class implementRandomForestRegressor():
    '''
    Class implementing all necessary steps for a ranom Forest 
    regression with sklearnpare
    @imput:
      @ dataset: The full path to a *.csv file containing the dataSet.
      @ targetCol: The name of the column in the dataSet containig the target values.
      @ splitProportion: The proportion for the testing set creation.
    '''
    def __init__(self, dataSet, targetCol, splitProportion ):
        self.seedRF = 50
        self.paramGrid = createSearshGrid()
        X,Y = importDataSet(dataSet, targetCol)
        self.x_train,self.x_validation,self.y_train, self.y_validation = train_test_split(X,Y, test_size = splitProportion) 
        self.rfr_WithGridSearch = implementRandomForestRegressor.createModelRFRegressorWithGridSearsh(self)

    def createModelRFRegressorWithGridSearsh(self):
        estimator = RandomForestRegressor(random_state = self.seedRF)
        scoring = ['accuracy','balanced_accuracy','roc_auc','precision','f1']
        modelRFRegressorWithGridSearsh = GridSearchCV(estimator, 
                                                    param_grid = self.paramGrid,
                                                    n_jobs = -1, 
                                                    scoring = scoring,
                                                    refit="balanced_accuracy",
                                                    cv = 3, 
                                                    verbose = 1, 
                                                    return_train_score = True
                                                    )
        return modelRFRegressorWithGridSearsh

    def fitRFRegressor(self, saveTheModel = True):
        y_train = np.array(self.y_train).ravel()
        self.rfr_WithGridSearch.fit(self.x_train, y_train)
        print(self.rfr_WithGridSearch.best_params_, "\n")
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        if saveTheModel:
            saveModel(best_estimator)
        investigateFeatureImportance(best_estimator, self.x_train)
        return best_estimator

    def getSplitedDataset(self):
        '''
        just a comment to chech something
        '''
        return self.x_train,self.x_validation,self.y_train, self.y_validation
    
def importDataSet(dataSetName, targetCol: str):
    '''
    Import datasets and filling NaN values          
    @input: DataSetName => The dataset path. 
    @Output: Features(x) and tragets(y)    
    ''' 
    train = pd.read_csv(dataSetName, index_col = None)
    y = train[[targetCol]]
    y = y.fillna(0)
    x = train.drop(targetCol, axis=1)
    xMean = x.mean()
    x = x.fillna(xMean)
    return x, y

def split(x,y,TestPercent = 0.2):
    x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=TestPercent)
    return x_train, x_validation, y_train, y_validation

def printDataBalace(x_train, x_validation, y_train, y_validation, targetCol: str):
    ## Data shape exploration
    print("",np.shape(x_train),"  :",np.shape(x_validation) )
    print("Label balance on Training set: ", "\n", y_train[targetCol].value_counts())
    print("Label balance on Validation set: ", "\n", y_validation[targetCol].value_counts())

def predictOnFeaturesSet(model, featuresSet):
    y_hat = model.predict(featuresSet)
    return y_hat

def investigateFeatureImportance(classifier, x_train, printed = True):
    '''
    @input: feature matrix
            classifier trained
            printed option: Default = True
    @return: List of features ordered dessending by importance (pandas DF format). 
    '''
    features = x_train.columns
    clasifierName = type(classifier).__name__
    featuresInModel= pd.DataFrame({'feature': features,
                   'importance': classifier.feature_importances_}).\
                    sort_values('importance', ascending = False)
    clasifierNameExtended = clasifierName + "_featureImportance.csv"     
    featuresInModel.to_csv(clasifierNameExtended, index = None)
    if printed:
        with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
            print(featuresInModel)
    return featuresInModel

def createSearshGrid():
    param_grid = {
    'n_estimators': np.linspace(90, 120,2).astype(int), #default 100
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': list(np.arange(0.2, 1, 0.1)),
    'max_leaf_nodes': [None], # + list(np.linspace(10, 50, 500).astype(int)),
    # 'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
    }
    return param_grid   

def saveModel(best_estimator):
    date = time.strftime("%Y%M%D_%H%M%S")
    myServices.ensureDirectory('./models/rwReg')
    name = "rfwgs_"+ str(date) + ".pkl" 
    destiny = "./models//rwReg/" + name
    joblib.dump(best_estimator, destiny)



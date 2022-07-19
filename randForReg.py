'''
Aqui vamos a poner 
todo lo necesario para hacer fincionet RF a ppartir de competition 2
'''
import time
import myServices
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
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
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']
        modelRFRegressorWithGridSearsh = GridSearchCV(estimator, 
                                                    param_grid = self.paramGrid,
                                                    n_jobs = -1, 
                                                    scoring = scoring,
                                                    refit= "neg_mean_squared_error",
                                                    cv = 3, 
                                                    verbose = 1, 
                                                    return_train_score = True
                                                    )
        return modelRFRegressorWithGridSearsh

    def fitRFRegressor(self, saveTheModel = True, enhanceClassDiff = True):
        name = makeNameByTime()
        if enhanceClassDiff:
            implementRandomForestRegressor.enhanceClassDifferences(self, 10)    
        y_train= (np.array(self.y_train).astype('int')).ravel()
        self.rfr_WithGridSearch.fit(self.x_train, y_train)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        if saveTheModel:
            saveModel(best_estimator,name)
        investigateFeatureImportance(best_estimator, self.x_train)
        print(f"The best parameters: {self.rfr_WithGridSearch.best_params_}")
        reportErrors(best_estimator, self.x_validation, self.y_validation)
        return best_estimator
    
    def fitRFRegressorWeighted(self, dominantClassPenalty, saveTheModel = True, enhanceClassDiff = True):
        name = makeNameByTime()
        if enhanceClassDiff:
            implementRandomForestRegressor.enhanceClassDifferences(self, 10)    
        y_train= (np.array(self.y_train).astype('int')).ravel()
        weights = createWeightVector(y_train, dominantClassPenalty)
        self.rfr_WithGridSearch.fit(self.x_train, y_train,sample_weight = weights)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        if saveTheModel:
            saveModel(best_estimator,name)
        investigateFeatureImportance(best_estimator,name,self.x_train)
        print(f"The best parameters: {self.rfr_WithGridSearch.best_params_}")
        reportErrors(best_estimator, self.x_validation, self.y_validation)
        return best_estimator
    
    def enhanceClassDifferences(self, factor):
        self.y_train = self.y_train*factor
        self.y_validation*factor
    
    def getSplitedDataset(self):
        return self.x_train,self.x_validation,self.y_train, self.y_validation


def split(x,y,TestPercent = 0.2):
    x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=TestPercent)
    return x_train, x_validation, y_train, y_validation 

def importDataSet(dataSetName, targetCol: str):
    '''
    Import datasets and filling NaN values          
    @input: DataSetName => The dataset path. 
    @Output: Features(x) and tragets(y) 
    ''' 
    train = pd.read_csv(dataSetName, index_col = None)
    y = train[[targetCol]]
    y = y.fillna(0)
    train.drop([targetCol,'x_coord', 'y_coord'], axis=1, inplace = True)
    xMean = train.mean()
    x = train.fillna(xMean)
    return x, y

def printDataBalace(x_train, x_validation, y_train, y_validation, targetCol: str):
    ## Data shape exploration
    print("",np.shape(x_train),"  :",np.shape(x_validation) )
    print("Label balance on Training set: ", "\n", y_train[targetCol].value_counts())
    print("Label balance on Validation set: ", "\n", y_validation[targetCol].value_counts())

def predictOnFeaturesSet(model, featuresSet):
    y_hat = model.predict(featuresSet)
    return y_hat

def computeMainErrors(model, x_test, y_test):
    y_test  = (np.array(y_test).astype('int')).ravel()
    y_pred = model.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, y_pred) 
    mse= metrics.mean_squared_error(y_test, y_pred)
    r_mse = np.sqrt(metrics.mean_squared_error(y_test, y_pred)) # Root Mean Squared Error
    return mae, mse, r_mse

def reportErrors(model, x_test, y_test):
    mae, mse, r_mse = computeMainErrors(model, x_test, y_test)
    print('Mean Absolute Error: ',mae )  
    print('Mean Squared Error: ', mse)  
    print('Root Mean Squared Error: ',r_mse)

def investigateFeatureImportance(classifier, dateName, x_train, printed = True):
    '''
    @input: feature matrix
            classifier trained
            printed option: Default = True
    @return: List of features ordered dessending by importance (pandas DF format). 
    '''
    features = x_train.columns
    clasifierName = type(classifier).__name__
    clasifierName = clasifierName + dateName
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
    'n_estimators': np.linspace(20, 200,5).astype(int), #default 100
    'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
    'max_features': list(np.arange(0.2, 1, 0.1)),
    'max_leaf_nodes': [None] + list(np.linspace(2, 30, 1).astype(int)),
    # 'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
    }
    return param_grid   

def createWeightVector(y_vector, dominantClassPenalty):
    '''
    Create wight vector for sampling weighted training.
    The goal is to penalize the dominant class. 
    This is important is the flood study, where majority of points (usually more than 95%) 
    are not flooded areas. 
    '''
    weightVec = np.ones_like(y_vector).astype(float)
    weightVec = [dominantClassPenalty if y_vector[j] == 0 else 1 for j in range(len(y_vector))]
    return weightVec

def saveModel(best_estimator, id):
    myServices.ensureDirectory('./models/rwReg')
    name = "rfwgs_"+ id + ".pkl" 
    destiny = "./models/rwReg/" + name
    print(destiny)
    _ = joblib.dump(best_estimator, destiny, compress=9)

def makeNameByTime():
    name = time.strftime("%y%m%d%H%M")
    return name


def main():
    datasetPath = './sample.csv'
    rfReg = implementRandomForestRegressor(datasetPath,'percentage', 0.2)
    x_train,x_validation,y_train, y_validation = rfReg.getSplitedDataset()
    printDataBalace(x_train, x_validation, y_train, y_validation,'percentage')
    bestReg = rfReg.fitRFRegressorWeighted(0.1)
    return bestReg


if __name__ == "__main__":
    with myServices.timeit():
        main()
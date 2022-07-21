'''
Aqui vamos a poner 
todo lo necesario para hacer fincionet RF a ppartir de competition 2
'''
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV,  RandomizedSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from myServices import clipRasterWithPoligon

class implementRandomForestCalssifier():
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
        Y = pd.factorize(self.y_train)
        self.x_train,self.x_validation,self.y_train, self.y_validation = train_test_split(X,Y, test_size = splitProportion) 
        print(self.x_train.head())
        print("Train balance")
        printArrayBalance(self.y_train)
        print("Validation balance")
        printArrayBalance(self.y_validation)
        self.rfClassifier = implementRandomForestCalssifier.createModelRClassifier(self)
    
    def createModelRClassifier(self):
        estimator = RandomForestClassifier(random_state = self.seedRF)
        # Create the random search model
        rs = RandomizedSearchCV(estimator, 
                                critreion = 'entropy', 
                                param_grid = self.paramGrid, 
                                n_jobs = -1, 
                                scoring = 'roc_auc',
                                cv = 4,  # NOTE: in this configurtion StratifiedKfold is used by SckitLearn
                                n_iter = 10, 
                                verbose = 4, 
                                random_state=self.seedRF)           
        return rs
    
    def fitRFClassifierGSearch(self):
        name = "classifier" + makeNameByTime()
        self.rfClassifier.fit(self.x_train, self.y_train)
        best_estimator = self.rfClassifier.best_estimator_
        saveModel(best_estimator,name)
        investigateFeatureImportance(best_estimator, name, self.x_train)
        print(f"The best parameters are: {self.rfClassifier.best_params_}")
        implementRandomForestCalssifier.printClassificatinMetrics(self, best_estimator,self.x_validation,self.y_validation)
        

        return best_estimator

    def printClassificatinMetrics(model,x_validation,y_validation):



        return


    def getSplitedDataset(self):
        return self.x_train,self.x_validation,self.y_train, self.y_validation

class implementRandomForestRegressor():
    '''
    Class implementing all necessary steps for a ranom Forest 
    regression with sklearnpare
    @imput:
      @ dataset: The full path to a *.csv file containing the dataSet.
      @ targetCol: The name of the column in the dataSet containig the target values.
      @ splitProportion: The proportion for the testing set creation.
    '''
    def __init__(self, dataSet, targetCol, splitProportion, gridArgs):
        self.seedRF = 50
        self.paramGrid = createSearshGrid(gridArgs)
        X,Y = importDataSet(dataSet, targetCol)
        Y = quadraticRechapeLabes(Y, -0.125, 0.825)
        self.x_train,self.x_validation,self.y_train, self.y_validation = train_test_split(X,Y, test_size = splitProportion)
        print(self.x_train.head())
        print("Train balance")
        printArrayBalance(self.y_train)
        print("Validation balance")
        printArrayBalance(self.y_validation)
        self.rfr_WithGridSearch = implementRandomForestRegressor.createModelRFRegressorWithGridSearch(self)

    def createModelRFRegressorWithGridSearch(self):
        estimator = RandomForestRegressor(random_state = self.seedRF)
        # scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']
        modelRFRegressorWithGridSearch = GridSearchCV(estimator, 
                                                    param_grid = self.paramGrid,
                                                    n_jobs = -1, 
                                                    verbose = 2, 
                                                    return_train_score = True
                                                    )
        return modelRFRegressorWithGridSearch

    def fitRFRegressorGSearch(self):
        y_train= (np.array(self.y_train).astype('float')).ravel()
        self.rfr_WithGridSearch.fit(self.x_train, y_train)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        print(f"The best parameters: {self.rfr_WithGridSearch.best_params_}")
        r2_validation = validateWithR2(best_estimator, self.x_validation, self.y_validation, weighted = False)
        print("R2_score for validation set: ", r2_validation)
        return best_estimator, r2_validation
        
    def fitRFRegressorWeighted(self, dominantValeusPenalty):
        y_train= (np.array(self.y_train).astype('float')).ravel()
        weights = createWeightVector(y_train, 
                                    dominantValue = 0, dominantValuePenalty = dominantValeusPenalty)
        self.rfr_WithGridSearch.fit(self.x_train, y_train,sample_weight = weights)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        print(f"The best parameters: {self.rfr_WithGridSearch.best_params_}")
        r2_validation = validateWithR2(best_estimator,self.x_validation,self.y_validation,
                                      dominantValue = 0, dominantValuePenalty = dominantValeusPenalty)
        print("R2_score for validation set: ", r2_validation)
        return best_estimator, r2_validation
    
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
    train.drop([targetCol,'Unnamed: 0'], axis=1, inplace = True)
    return train, y

def printDataBalace(x_train, x_validation, y_train, y_validation, targetCol: str):
    ## Data shape exploration
    print("",np.shape(x_train),"  :",np.shape(x_validation) )
    print("Label balance on Training set: ", "\n", y_train[targetCol].value_counts())
    print("Label balance on Validation set: ", "\n", y_validation[targetCol].value_counts())

def printArrayBalance(array):
    unique, count = np.unique(array, return_counts=True)
    print('values,  counts')
    result = np.column_stack([unique, count]) 
    print(result)
    
def predictOnFeaturesSet(model, featuresSet):
    y_hat = model.predict(featuresSet)
    return y_hat

def validateWithR2(model, x_test, y_test, dominantValue = 0, dominantValuePenalty = 0.1, weighted = True):
    y_hate = model.predict(x_test)
    if weighted:
        weights = createWeightVector(y_test,dominantValue,dominantValuePenalty)
        r2 = metrics.r2_score(y_test, y_hate,sample_weight = weights)
    else: 
        r2 = metrics.r2_score(y_test, y_hate)
    return r2

def computeMainErrors(model, x_test, y_test ):
    y_test  = (np.array(y_test).astype('float')).ravel()
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

def investigateFeatureImportance(bestModel, dateName, x_train, printed = True):
    '''
    DEFAULT Path to save: "./models/rwReg/" 
    @input: feature matrix
            bestModel and dateName: created when fitting
            x_train: Dataset to extract features names
            printed option: Default = True
    @return: List of features ordered dessending by importance (pandas DF format). 
    '''
    features = x_train.columns
    clasifierName = type(bestModel).__name__
    clasifierName = clasifierName + dateName
    featuresInModel= pd.DataFrame({'feature': features,
                   'importance': bestModel.feature_importances_}).\
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

def createSearshGrid(arg):
    param_grid = {
    'n_estimators': eval(arg['n_estimators']), 
    'max_depth': eval(arg['max_depth']), 
    'max_features': eval(arg['max_features']),
    'max_leaf_nodes': eval(arg['max_leaf_nodes']),
    'pre_dispatch': '2*n_jobs',
    'bootstrap': eval(arg['bootstrap']),
    }
    return param_grid   

def createWeightVector(y_vector, dominantValue:float, dominantValuePenalty:float):
    '''
    Create wight vector for sampling weighted training.
    The goal is to penalize the dominant class. 
    This is important is the flood study, where majority of points (usually more than 95%) 
    are not flooded areas. 
    '''
    y_ravel  = (np.array(y_vector).astype('int')).ravel()
    weightVec = np.ones_like(y_ravel).astype(float)
    weightVec = [dominantValuePenalty if y_ravel[j] == dominantValue else 1 for j in range(len(y_ravel))]
    return weightVec

def saveModel(best_estimator, id):
    # myServices.ensureDirectory('./models/rwReg')
    name = id + ".pkl" 
    # destiny = "./models/rwReg/" + name
    # print(destiny)
    _ = joblib.dump(best_estimator, name, compress=9)

def makeNameByTime():
    name = time.strftime("%y%m%d%H%M")
    return name

def quadraticRechapeLabes(x, a, b):
    '''
    Apply quadratic function to a vector like : y = aX^2 + bX.
    You're responsable for providing a and b 
    '''
    x = np.array(x.copy())
    v = (a*x*x) + (b*x) 
    return v.ravel()




# def main():
    

# if __name__ == "__main__":
#     with myServices.timeit():
#         main()
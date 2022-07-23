'''
Aqui vamos a poner 
todo lo necesario para hacer fincionet RF a ppartir de competition 2
'''
from logging import critical
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV,  RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import myServices as ms

class implementRandomForestCalssifier():
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
        X,Y = ms.importDataSet(dataSet, targetCol)
        Y = np.array(Y).ravel()
        Y_factorized, self.labels = pd.factorize(Y) # See: https://pandas.pydata.org/docs/reference/api/pandas.factorize.html
        self.x_train,self.x_validation,self.y_train, self.y_validation = train_test_split(X,Y_factorized, test_size = splitProportion) 
        print(self.x_train.head())
        print("Train balance")
        printArrayBalance(self.y_train)
        print("Validation balance")
        printArrayBalance(self.y_validation)
        self.rfClassifier = implementRandomForestCalssifier.createModelRClassifier(self)
    
    def createModelRClassifier(self):
        estimator = RandomForestClassifier(criterion='entropy', random_state = self.seedRF)
        # Create the random search model
        rs = GridSearchCV(estimator, 
                        param_grid = self.paramGrid, 
                        n_jobs = -1, 
                        scoring = 'roc_auc',
                        cv = 4,  # NOTE: in this configurtion StratifiedKfold is used by SckitLearn
                        verbose = 5, 
                        )           
        return rs
    
    def fitRFClassifierGSearch(self):
        self.rfClassifier.fit(self.x_train, self.y_train)
        best_estimator = self.rfClassifier.best_estimator_
        best_params = self.rfClassifier.get_params()
        print(f"The best parameters are: {best_params}")
        return best_estimator,best_params

    def computeClassificationMetrics(self, model):
        '''
        Ref: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification/notebook
        '''
        y_hat = model.predict(self.x_validation)
        accScore = metrics.accuracy_score(self.y_validation, y_hat)
        macro_averaged_f1 = metrics.f1_score(self.y_validation, y_hat, average = 'macro') # Better for multiclass
        micro_averaged_f1 = metrics.f1_score(self.y_validation, y_hat, average = 'micro')
        ROC_AUC_multiClass = implementRandomForestCalssifier.roc_auc_score_multiclass(self,y_hat)
        print('Accuraci_score: ', accScore)  
        print('F1_macroAverage: ', macro_averaged_f1)  
        print('F1_microAverage: ', micro_averaged_f1)
        print('ROC_AUC one_vs_all: ', ROC_AUC_multiClass)
        return accScore, macro_averaged_f1, micro_averaged_f1, ROC_AUC_multiClass

    def roc_auc_score_multiclass(self, y_hat):
        '''
        Compute one-vs-all for every single class in the dataset
        From: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification/notebook
        '''
        #creating a set of all the unique classes using the actual class list
        unique_class = set(self.y_validation)
        roc_auc_dict = {}
        for per_class in unique_class:
            #creating a list of all the classes except the current class 
            other_class = [x for x in unique_class if x != per_class]
            #marking the current class as 1 and all other classes as 0
            new_y_validation = [0 if x in other_class else 1 for x in self.y_validation]
            new_y_hat = [0 if x in other_class else 1 for x in y_hat]
            #using the sklearn metrics method to calculate the roc_auc_score
            roc_auc = roc_auc_score(new_y_validation, new_y_hat, average = "macro")
            roc_auc_dict[per_class] = roc_auc
        return roc_auc_dict

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
        X,Y = ms.importDataSet(dataSet, targetCol)
        # Y = quadraticRechapeLabes(Y, -0.125, 0.825)
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
                                                    cv = 2,
                                                    n_jobs = -1, 
                                                    verbose = 5, 
                                                    return_train_score = True
                                                    )
        return modelRFRegressorWithGridSearch

    def fitRFRegressorGSearch(self):
        y_train= (np.array(self.y_train).astype('float')).ravel()
        self.rfr_WithGridSearch.fit(self.x_train, y_train)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        bestParameters = self.rfr_WithGridSearch.best_params_
        print(f"The best parameters: {bestParameters}")
        r2_validation = validateWithR2(best_estimator, self.x_validation, self.y_validation,0,0,weighted = False)
        print("R2_score for validation set: ", r2_validation)
        return best_estimator, bestParameters, r2_validation
        
    def fitRFRegressorWeighted(self, dominantValeusPenalty):
        y_train= (np.array(self.y_train).astype('float')).ravel()
        weights = createWeightVector(y_train, 0, dominantValeusPenalty)
        self.rfr_WithGridSearch.fit(self.x_train, y_train,sample_weight = weights)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        bestParameters = self.rfr_WithGridSearch.best_params_
        print(f"The best parameters: {bestParameters}")
        r2_validation = validateWithR2(best_estimator,self.x_validation,self.y_validation,0, dominantValeusPenalty)
        print("R2_score for validation set: ", r2_validation)
        return best_estimator, bestParameters, r2_validation
    
    def getSplitedDataset(self):
        return self.x_train,self.x_validation,self.y_train, self.y_validation


def split(x,y,TestPercent = 0.2):
    x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=TestPercent)
    return x_train, x_validation, y_train, y_validation 

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

def validateWithR2(model, x_test, y_test, dominantValue:float, dominantValuePenalty:float, weighted = True):
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
    # clasifierNameExtended = clasifierName + "_featureImportance.csv"     
    # featuresInModel.to_csv(clasifierNameExtended, index = None)
    if printed:
        with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
            print(featuresInModel)
    return clasifierName, featuresInModel

def createSearshGrid(arg):
    param_grid = {
    'n_estimators': eval(arg['n_estimators']), 
    'max_depth': eval(arg['max_depth']), 
    'max_features': eval(arg['max_features']),
    'max_leaf_nodes': eval(arg['max_leaf_nodes']),
    'bootstrap': eval(arg['bootstrap']),
    #'pre_dispatch': '2*n_jobs', # UNCOMMENT Only for classifier
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
    name = id + ".pkl" 
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

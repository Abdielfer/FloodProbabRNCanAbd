'''
Aqui vamos a poner 
todo lo necesario para hacer fincionet RF a ppartir de competition 2
'''
from re import L
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics 
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import myServices as ms

class implementRandomForestCalssifier():
    '''
    Class implementing all necessary steps for a ranom Forest 
    regression with sklear
    @imput:
      @ dataset: The full path to a *.csv file containing the dataSet.
      @ targetCol: The name of the column in the dataSet containig the target values.
      @ splitProportion: The proportion for the testing set creation.
    '''
    def __init__(self,trainDataSet, targetCol, gridArgs):
        self.seedRF = 50
        self.paramGrid = createSearshGrid(gridArgs)
        self.x_train,self.y_train= ms.importDataSet(trainDataSet, targetCol)
        self.y_train = np.array(self.y_train).ravel()
        print(self.x_train.head())
        print("Train balance")
        _,l = listClassCountPercent(self.y_train)
        print(l)
        self.rfClassifier = implementRandomForestCalssifier.createModelClassifier(self)
    
    def createModelClassifier(self):
        estimator = RandomForestClassifier(criterion='entropy', random_state = self.seedRF)
        # Create the random search model
        rs = GridSearchCV(estimator, 
                        param_grid = self.paramGrid, 
                        n_jobs = -1,
                        scoring = 'accuracy',
                        cv = 3,  # NOTE: in this configurtion StratifiedKfold is used by SckitLearn  
                        verbose = 5, 
                        )           
        return rs
    
    def fitRFClassifierGSearch(self):
        self.rfClassifier.fit(self.x_train, self.y_train)
        best_estimator = self.rfClassifier.best_estimator_
        best_params = self.rfClassifier.best_params_
        print(f"The best parameters are: {best_params}")
        return best_estimator, best_params  


class implementOneVsRestClassifier():
    '''
    Class implementing all necessary steps for a ranom Forest 
    regression with sklearn
    @imput:
      @ dataset: The full path to a *.csv file containing the dataSet.
      @ targetCol: The name of the column in the dataSet containig the target values.
      @ splitProportion: The proportion for the testing set creation.
    '''
    def __init__(self, dataSet, targetCol, gridArgs):
        self.seedRF = 50
        self.paramGrid = createSearshGridOneVsRest(gridArgs)
        self.x_train,Y = ms.importDataSet(dataSet, targetCol)
        Y = np.array(Y).ravel()
        self.y_train, self.labels = pd.factorize(Y) # See: https://pandas.pydata.org/docs/reference/api/pandas.factorize.html
        # self.x_train,self.x_validation,self.y_train, self.y_validation = train_test_split(X,Y_factorized, test_size = splitProportion) 
        print(self.x_train.head())
        print("Train balance")
        listClassCountPercent(self.y_train)
        # print("Validation balance")
        # printArrayBalance(self.y_validation)
        self.OneVsRestClassifier = implementOneVsRestClassifier.createModel(self)

    def createModel(self):
        estimator = RandomForestClassifier(criterion='entropy', random_state = self.seedRF, bootstrap = False) 
        model_to_set = OneVsRestClassifier(estimator)     
        # Create the random search model
        rs = GridSearchCV(model_to_set, 
                        param_grid = self.paramGrid, 
                        scoring = 'accuracy',
                        cv = 3,  # NOTE: in this configurtion StratifiedKfold is used by SckitLearn 
                        )           
        return rs
    
    def fitOneVsRestClassifierGSearch(self):
        self.OneVsRestClassifier.fit(self.x_train, self.y_train)
        best_params = self.OneVsRestClassifier.get_params
        model = self.OneVsRestClassifier.best_estimator_
        print(f"The best parameters are: {best_params}")
        return model, best_params  

   
    # def getSplitedDataset(self):
    #     return self.x_train,self.x_validation,self.y_train, self.y_validation


class implementRandomForestRegressor():
    '''
    Class implementing all necessary steps for a ranom Forest 
    regression with sklearnpare
    @imput:
      @ dataset: The full path to a *.csv file containing the dataSet.
      @ targetCol: The name of the column in the dataSet containig the target values.
      @ splitProportion: The proportion for the testing set creation.
    '''
    def __init__(self, dataSet, targetCol, gridArgs):
        self.seedRF = 50
        self.paramGrid = createSearshGrid(gridArgs)
        self.x_train, self.y_train= ms.importDataSet(dataSet, targetCol)
        print(self.x_train.head())
        print("Train balance")
        listClassCountPercent(self.y_train)
        self.rfr_WithGridSearch = implementRandomForestRegressor.createModelRFRegressorWithGridSearch(self)

    def createModelRFRegressorWithGridSearch(self):
        estimator = RandomForestRegressor(random_state = self.seedRF)
        # scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']
        modelRFRegressorWithGridSearch = GridSearchCV(estimator, 
                                                    param_grid = self.paramGrid,
                                                    cv = 3,
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
        return best_estimator, bestParameters, 
        
    def fitRFRegressorWeighted(self, dominantValeusPenalty):
        y_train= (np.array(self.y_train).astype('float')).ravel()
        weights = ms.createWeightVector(y_train, 0, dominantValeusPenalty)
        self.rfr_WithGridSearch.fit(self.x_train, y_train,sample_weight = weights)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        bestParameters = self.rfr_WithGridSearch.best_params_
        print(f"The best parameters: {bestParameters}")
        return best_estimator, bestParameters
     

class implementingMLPCalssifier():
    def __init__(self, dataSet, targetCol, args):
        self.seedRF = 50
        self.args = args
        self.logsDic = {}
        print(dataSet)
        print(targetCol)
        self.x_train, self.y_train= ms.importDataSet(dataSet, targetCol)
        self.mlpClassifier = implementingMLPCalssifier.createMLPClassifier(self)
        ### Report ###
        print(self.x_train.head())
        print("Train balance")
        print(listClassCountPercent(self.y_train))

    def createMLPClassifier(self):
        '''
        defoult parameters: MLPClassifier(hidden_layer_sizes=(100,), activation=['relu'/ 'logistic’], *, solver='adam', alpha=0.0001,
        batch_size='auto',learning_rate='constant'({‘constant’, ‘invscaling’, ‘adaptive’}, learning_rate_init=0.001, default=’constant’)
        power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)[source]
        '''
        params = {'random_state':50, 'hidden_layer_sizes':(200,150,100,50),
                'early_stopping':True,'verbose':True,
                'tol':0.00010,'validation_fraction':0.2,'warm_start':True}
        mlpClassifier = MLPClassifier(**params)
        return mlpClassifier

    def fitMLPClassifier(self):
        #  with warnings.catch_warnings():
        #     warnings.filterwarnings(
        #         "ignore", category=ConvergenceWarning, module="sklearn"
        #     )
        self.mlpClassifier.fit(self.x_train.values, self.y_train.values)
        implementingMLPCalssifier.logMLPClassifier(self)
        return 

    def getMLPClassifier(self):
        return self.mlpClassifier

    def plotLossBehaviour(self):
        lossList = self.mlpClassifier.loss_curve_
        iters = np.arange(1,self.mlpClassifier.n_iter_+1)
        plt.rcParams.update({'font.size': 14})
        plt.ylabel('Loss', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)
        plt.plot(iters,lossList)
    
    def logMLPClassifier(self):
        self.logsDic['optimizer'] = self.mlpClassifier._optimizer
        self.logsDic['activation'] = self.mlpClassifier.activation
        self.logsDic['hidden_layer_sizes'] = self.mlpClassifier.hidden_layer_sizes
        self.logsDic['n_iter'] = self.mlpClassifier.n_iter_
        self.logsDic['lossCurve'] = self.mlpClassifier.loss_curve_
    
    def get_logsDic(self):
        return self.logsDic


def split(x,y,TestPercent = 0.2):
    x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=TestPercent)
    return x_train, x_validation, y_train, y_validation 

def printDataBalace(x_train, x_validation, y_train, y_validation, targetCol: str):
    ## Data shape exploration
    print("",np.shape(x_train),"  :",np.shape(x_validation) )
    print("Label balance on Training set: ", "\n", y_train[targetCol].value_counts())
    print("Label balance on Validation set: ", "\n", y_validation[targetCol].value_counts())

def listClassCountPercent(array):
    '''
      Return the <total> amount of elements and the number of unique values (Classes) 
    in input <array>, with the corresponding count and percent respect to <total>. 
    '''
    unique, count = np.unique(array, return_counts=True)
    total = count.sum()
    result = np.empty_like(unique,dtype='f8')
    result = [(i/total) for i in count]
    listClassCountPercent = {}
    for i in range(len(unique)):
        listClassCountPercent[unique[i]] = str(f"Class_count: {count[i]}  for  %.4f  percent" %(result[i]))
    return total, listClassCountPercent
   
def predictOnFeaturesSet(model, featuresSet):
    y_hat = model.predict(featuresSet)
    return y_hat

def validateWithR2(model, x_test, y_test, dominantValue:float, dominantValuePenalty:float, weighted = True):
    y_hate = model.predict(x_test)
    if weighted:
        weights = ms.createWeightVector(y_test,dominantValue,dominantValuePenalty)
        r2 = metrics.r2_score(y_test, y_hate,sample_weight = weights)
    else: 
        r2 = metrics.r2_score(y_test, y_hate)
    return r2

def computeClassificationMetrics(model, x_validation, y_validation):
    '''
    Ref: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification/notebook
    '''
    y_hat = model.predict(x_validation)
    accScore = metrics.accuracy_score(y_validation, y_hat)
    macro_averaged_f1 = metrics.f1_score(y_validation, y_hat, average = 'macro') # Better for multiclass
    micro_averaged_f1 = metrics.f1_score(y_validation, y_hat, average = 'micro')
    ROC_AUC_multiClass = roc_auc_score_multiclass(y_validation,y_hat)
    print('Accuraci_score: ', accScore)  
    print('F1_macroAverage: ', macro_averaged_f1)  
    print('F1_microAverage: ', micro_averaged_f1)
    print('ROC_AUC one_vs_all: ', ROC_AUC_multiClass)
    return accScore, macro_averaged_f1, micro_averaged_f1, ROC_AUC_multiClass


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
    

def investigateFeatureImportance(bestModel, x_train, printed = True):
    '''
    DEFAULT Path to save: "./models/rwReg/" 
    @input: feature matrix
            bestModel and dateName: created when fitting
            x_train: Dataset to extract features names
            printed option: Default = True
    @return: List of features ordered dessending by importance (pandas DF format). 
    '''
    features = x_train.columns
    featuresInModel= pd.DataFrame({'feature': features,
                   'importance': bestModel.feature_importances_}).\
                    sort_values('importance', ascending = False)
    # clasifierNameExtended = clasifierName + "_featureImportance.csv"     
    # featuresInModel.to_csv(clasifierNameExtended, index = None)
    if printed:
        with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 4,
                       ):
            print(featuresInModel)

    return featuresInModel

def roc_auc_score_multiclass(y_validation, y_hat):
        '''
        Compute one-vs-all for every single class in the dataset
        From: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification/notebook
        '''
        unique_class = y_validation.unique()
        print("UNIQUE CLASSES: ", unique_class)
        roc_auc_dict = {}
        for per_class in unique_class:
            other_class = [x for x in unique_class if x != per_class]
            new_y_validation = [0 if x in other_class else 1 for x in y_validation]
            new_y_hat = [0 if x in other_class else 1 for x in y_hat]
            roc_auc = roc_auc_score(new_y_validation, new_y_hat, average = "macro")
            roc_auc_dict[per_class] = roc_auc
        return roc_auc_dict

def plot_ROC_AUC_OneVsRest(classifier, x_test, y_test):
    '''
    Allows to plot multiclass classification ROC_AUC by computing tpr and fpr of each calss by One vs Rest. 
    '''
    unique_class = y_test.unique()
    print("UNIQUE CLASSES: ", unique_class)
    y_hat = classifier.predict(x_test)
    print("Test Set balance:" )
    listClassCountPercent(y_test)
    print("Prediction balance:")
    listClassCountPercent(y_hat)
    y_prob = classifier.predict_proba(x_test)   
    fig, axs = plt.subplots(1,figsize=(13,4), sharey=True)
    plt.rcParams.update({'font.size': 14})
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.figure(0).clf()
    roc_auc_dict = {}
    i = 0
    for per_class in unique_class:
        y_testInLoop = y_test.copy()
        other_class = [x for x in unique_class if x != per_class]
        print(f"actual class: {per_class} vs rest {other_class}")
        new_y_validation = [0 if x in other_class else 1 for x in y_testInLoop]
        new_y_hat = [0 if x in other_class else 1 for x in y_hat]
        print(f"Class {per_class} balance vs rest")
        listClassCountPercent(new_y_validation)
        roc_auc = roc_auc_score(new_y_validation, new_y_hat, average = "macro")
        roc_auc_dict[per_class] = roc_auc
        fpr, tpr, _ = metrics.roc_curve(new_y_validation, y_prob[:,i])  ### TODO Results doen't match roc_auc_score..
        axs.plot(fpr,tpr,label = "Class "+ str(per_class) + " AUC : " + format(roc_auc,".4f")) 
        axs.legend()
        i+=1
    return roc_auc_dict

def plotLostCurve(lossList):

    pass


def createSearshGrid(arg):
    param_grid = {
    'n_estimators': eval(arg['n_estimators']), 
    'max_depth': eval(arg['max_depth']), 
    'max_features': eval(arg['max_features']),
    'max_leaf_nodes': eval(arg['max_leaf_nodes']),
    'bootstrap': eval(arg['bootstrap']),
    }
    return param_grid   

def createSearshGridOneVsRest(arg):
    param_grid = {
    'estimator__n_estimators': eval(arg['estimator__n_estimators']), 
    'estimator__max_depth': eval(arg['estimator__max_depth']), 
    'estimator__max_features': eval(arg['estimator__max_features']),
    'estimator__max_leaf_nodes': eval(arg['estimator__max_leaf_nodes']),
    'n_jobs': eval(arg['n_jobs']), 
    'verbose': eval(arg['verbose']),
    }
    return param_grid   

def quadraticRechapeLabes(x, a, b):
    '''
    Apply quadratic function to a vector like : y = aX^2 + bX.
    You're responsable for providing a and b 
    '''
    x = np.array(x.copy())
    v = (a*x*x) + (b*x) 
    return v.ravel()

'''
Aqui vamos a poner 
todo lo necesario para hacer fincionet RF a ppartir de competition 2
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics 
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
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
    Class implementing all necessary steps for a ranom multiclass classification with RF Classifier
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
     
class implementingMLPCalssifierOneHLayer():
    def __init__(self, trainingSet, targetCol, params):
        '''
        @MLPClassifier >> default parameters: MLPClassifier(hidden_layer_sizes=(100,), activation=['relu'/ 'logistic’], *, solver='adam', alpha=0.0001,
        batch_size='auto',learning_rate='constant'({‘constant’, ‘invscaling’, ‘adaptive’}, learning_rate_init=0.001, default=’constant’)
        power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)[source]
        '''
        self.params = params
        self.logsDic = {}
        self.x_train, self.y_train= ms.importDataSet(trainingSet, targetCol)
        self.mlpClassifier = MLPClassifier(**self.params)
        ### Report ###
        self.scoreRecord = self.newScoreRecordDF(self)
        print(self.x_train.head())
        print("Train balance")
        print(listClassCountPercent(self.y_train))

    def fitMLPClassifier(self):
        #  with warnings.catch_warnings():
        #     warnings.filterwarnings(
        #         "ignore", category=ConvergenceWarning, module="sklearn"
        #     )
        y_train = self.y_train.copy()
        x_trains = self.x_train.copy()
        self.mlpClassifier.fit(x_trains.values,y_train.values)
        self.logMLPClassifier(self)
        return 
 
    def explore4BestHLSize(self,X_val,Y_val,firstInterval,classOfInterest,loops):
        '''
        This function implements a cyclic exploration to find the best hidden layer size   
        ''' 
        print('firstInterval_____:', firstInterval)
        def helperTrain(i):
            self.params['hidden_layer_sizes']=int(i)
            self.restartMLPCalssifier(self,self.params)
            self.fitMLPClassifier(self)
            y_hat = self.mlpClassifier.predict(X_val.values)
            ROC_AUC_calculation = roc_auc_score_calculation(Y_val,y_hat)
            scoreList = []
            keysROC = list(ROC_AUC_calculation.keys())
            keysROC.sort()
            for k in keysROC:
                scoreList.append(ROC_AUC_calculation[k])
            scoreList.append(i)
            self.scoreRecord.loc[len(self.scoreRecord)]= scoreList
            hypParam, bestScore = self.getHyperParamOfBestClassScoreRecorded(self,classOfInterest)
            print('ROC_AUC___ HiperParam: ', scoreList)
            # print(scoreList)
            print("current center__: ",hypParam, ' corrent best score__:', bestScore)
            return hypParam
        
        def iters(loops,firstInterval):
            '''
            function@: A function that returns as last element an integer, 
            corresponding to the center of next intervale to explore or the optimal value from the last iteration.  
            '''
            print("Loops to go >>>",loops)
            if loops>1:
                loops-=1
                for i in firstInterval:
                    center = helperTrain(i)
                interval = buildInterval(loops,center)
                print("loop____",loops ,' newInterval____: ', interval)
                iters(loops,interval)
            else:
                for j in firstInterval:
                    center = helperTrain(j)
            print('Final loop')
            return center 
        bestSize:int
        bestSize = iters(loops,firstInterval)
        print(self.scoreRecord)
        return bestSize 

    def logMLPClassifier(self,newFeatureDic = {}):
        '''
        @newFeatureDic : is a dictionary of features to add
        '''
        self.logsDic['optimizer'] = self.mlpClassifier._optimizer
        self.logsDic['activation'] = self.mlpClassifier.activation
        self.logsDic['hidden_layer_sizes'] = self.mlpClassifier.hidden_layer_sizes
        self.logsDic['n_iter'] = self.mlpClassifier.n_iter_
        self.logsDic['lossCurve'] = self.mlpClassifier.loss_curve_
        for k in newFeatureDic.keys():
            self.logsDic[k] = newFeatureDic[k]
     
    def plotLossBehaviour(self):
        lossList = self.mlpClassifier.loss_curve_
        epochs = np.arange(1,self.mlpClassifier.n_iter_+1)
        plt.rcParams.update({'font.size': 14})
        plt.ylabel('Loss', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)
        plt.plot(epochs,lossList)
    
    def newScoreRecordDF(self):
        recordDF = pd.DataFrame()
        classList = (self.y_train.unique()).tolist()
        classList.sort()
        if len(classList)<=2:
            recordDF['Metric'] = pd.Series(dtype=float)
        else: 
            for i in classList:
                className = 'class_'+str(i)
                recordDF[className] = pd.Series(dtype=float)
        recordDF['hyperParam'] = pd.Series(dtype='int16')
        print('New Score recorder ready: ', recordDF)
        return recordDF
    
    def getHyperParamOfBestClassScoreRecorded(self,classOfInterest):
        bestScore = self.scoreRecord[classOfInterest].max()
        newCenter = self.scoreRecord.loc[self.scoreRecord[classOfInterest].idxmax(),'hyperParam']
        return newCenter, bestScore

    def restartMLPCalssifier(self, params):
        self.mlpClassifier = MLPClassifier(**params)

    def getMLPClassifier(self):
        return self.mlpClassifier

    def get_logsDic(self):
        return self.logsDic

class implementingMLPCalssifier():
    def __init__(self, trainingSet, targetCol, params):
        '''
        @MLPClassifier >> default parameters: MLPClassifier(hidden_layer_sizes=(100,), activation=['relu'/ 'logistic’], *, solver='adam', alpha=0.0001,
        batch_size='auto',learning_rate='constant'({‘constant’, ‘invscaling’, ‘adaptive’}, learning_rate_init=0.001, default=’constant’)
        power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)[source]
        '''
        self.params = params
        self.logsDic = {}
        self.x_train, self.y_train= ms.importDataSet(trainingSet, targetCol)
        self.mlpClassifier = MLPClassifier(**self.params)
        ### Report ###
        self.scoreRecord = self.newScoreRecordDF(self)
        print(self.x_train.head())
        print("Train balance")
        print(listClassCountPercent(self.y_train))

    def fitMLPClassifier(self):
        #  with warnings.catch_warnings():
        #     warnings.filterwarnings(
        #         "ignore", category=ConvergenceWarning, module="sklearn"
        #     )
        y_train = self.y_train.copy()
        x_trains = self.x_train.copy()
        self.mlpClassifier.fit(x_trains.values,y_train.values)
        self.logMLPClassifier(self)
        return 
 
    def logMLPClassifier(self):
        self.logsDic['optimizer'] = self.mlpClassifier._optimizer
        self.logsDic['activation'] = self.mlpClassifier.activation
        self.logsDic['hidden_layer_sizes'] = self.mlpClassifier.hidden_layer_sizes
        self.logsDic['n_iter'] = self.mlpClassifier.n_iter_
        self.logsDic['lossCurve'] = self.mlpClassifier.loss_curve_
        
     
    def plotLossBehaviour(self):
        lossList = self.mlpClassifier.loss_curve_
        epochs = np.arange(1,self.mlpClassifier.n_iter_+1)
        plt.rcParams.update({'font.size': 14})
        plt.ylabel('Loss', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)
        plt.plot(epochs,lossList)
    
    def newScoreRecordDF(self):
        recordDF = pd.DataFrame()
        classList = (self.y_train.unique()).tolist()
        classList.sort()
        if len(classList)<=2:
            recordDF['Metric'] = pd.Series(dtype=float)
        else: 
            for i in classList:
                className = 'class_'+str(i)
                recordDF[className] = pd.Series(dtype=float)
        recordDF['hyperParam'] = pd.Series(dtype='int16')
        print('New Score recorder ready: ', recordDF)
        return recordDF
    
    def getHyperParamOfBestClassScoreRecorded(self,classOfInterest):
        bestScore = self.scoreRecord[classOfInterest].max()
        newCenter = self.scoreRecord.loc[self.scoreRecord[classOfInterest].idxmax(),'hyperParam']
        return newCenter, bestScore

    def restartMLPCalssifier(self, params):
        self.mlpClassifier = MLPClassifier(**params)

    def getMLPClassifier(self):
        return self.mlpClassifier

    def get_logsDic(self):
        return self.logsDic

class MLPModel:
    def __init__(self, dataset_path,trainingParams):
        self.dataset_path = dataset_path
        self.model = None
        self.logs = {}

        self.params = trainingParams
        self.epochs = self.params['epochs']
        self.lr = self.params['lr']
        self.splitProportion = self.params['testSize']
        self.batchSize = self.params['batchSize']

    def load_data(self):
        # Load the dataset
        data = pd.read_csv(self.dataset_path)
        
        # Assuming the last column is the target variable
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        listClassCountPercent(y)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= self.splitProportion, random_state=42)
        printDataBalace(X_train, X_test, y_train, y_test)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float), torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

    def train(self):
        # Load the data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Get the input size
        input_size = X_train.shape[1]
        
        # Define the hidden layer sizes
        hidden_layer_sizes = [int(1.5*input_size)]*3 + [input_size] + [int(input_size/2)]+[1]
        
        # Initialize the MLPClassifier
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[3], hidden_layer_sizes[4]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[4], len(torch.unique(y_train))),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[5], len(torch.unique(y_train))),
            nn.Sigmoid(),
        )
        
        # Define the loss function and the optimizer
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert the training set into torch tensors
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=self.batchSize, shuffle=True)
        
        # Train the model
        for e in range(self.epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                log_probs = self.model(inputs)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        # Print the accuracy on the test set
        with torch.no_grad():
            output = self.model(X_test)
            threshold = 0.5
            y_hat = (output > threshold).float()
            correct = (y_hat == y_test).sum().item()
            print("Test set accuracy: ", correct / len(y_test))
            roc_auc_score_calculation(y_test,y_hat)

        investigateFeatureImportance(self.model,X_test)
        self.logMLP()
        
        return self.model, self.logs 

    def getModel(self):
        return self.model
    
    def logMLP(self):
        '''
        @newFeatureDic : is a dictionary of features to add
        '''
        self.logs['optimizer'] = self.model._optimizer
        self.logs['activation'] = self.model.activation
        self.logs['hidden_layer_sizes'] = self.model.hidden_layer_sizes
        self.logs['n_iter'] = self.model.n_iter_
        self.logs['lossCurve'] = self.model.loss_curve_
       



### Helper functions
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
        listClassCountPercent[unique[i]] = str(f"Class_count: {count[i]} (%.4f percent)" %(result[i]))
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
    ROC_AUC_multiClass = roc_auc_score_calculation(y_validation,y_hat)
    # print('Accuraci_score: ', accScore)  
    # print('F1_macroAverage: ', macro_averaged_f1)  
    # print('F1_microAverage: ', micro_averaged_f1)
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

def roc_auc_score_calculation(y_validation, y_hat):
        '''
        Compute one-vs-all for every single class in the dataset
        From: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification/notebook
        '''
        unique_class = y_validation.unique()
        roc_auc_dict = {}
        if len(unique_class)<=2:
            rocBinary = roc_auc_score(y_validation, y_hat, average = "macro")
            roc_auc_dict['ROC'] = rocBinary
            return roc_auc_dict   
        for per_class in unique_class:
            other_class = [x for x in unique_class if x != per_class]
            new_y_validation = [0 if x in other_class else 1 for x in y_validation]
            new_y_hat = [0 if x in other_class else 1 for x in y_hat]
            roc_auc = roc_auc_score(new_y_validation, new_y_hat, average = "macro")
            roc_auc_dict[per_class] = roc_auc
        return roc_auc_dict

def plot_ROC_AUC(classifier, x_test, y_test):
    '''
    Allows to plot multiclass classification ROC_AUC (One vs Rest), by computing tpr and fpr of each calss respect the rest. 
    The simplest application is the binary classification.
    '''
    unique_class = y_test.unique()
    unique_class.sort()
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
    if len(unique_class)<=2:  ## Binary case
            fpr,tpr,thresholds = metrics.roc_curve(y_test, y_prob[:,1], drop_intermediate=False)  ### TODO Results doen't match roc_auc_score..
            print(thresholds)
            roc_auc = roc_auc_score(y_test, y_hat, average = "macro")
            axs.plot(fpr,tpr,label = "Class "+ str(classifier.classes_[1]) + " AUC : " + format(roc_auc,".4f")) 
            axs.legend()
    else: 
        for per_class in unique_class:  # Multiclass
            y_testInLoop = y_test.copy()
            other_class = [x for x in unique_class if x != per_class]
            print(f"actual class: {per_class} vs rest {other_class}")
            new_y_validation = [0 if x in other_class else 1 for x in y_testInLoop]
            new_y_hat = [0 if x in other_class else 1 for x in y_hat]
            print(f"Class {per_class} balance vs rest")
            listClassCountPercent(new_y_validation)
            roc_auc = roc_auc_score(new_y_validation, new_y_hat, average = "macro")
            roc_auc_dict[per_class] = roc_auc
            fpr, tpr, _ = metrics.roc_curve(new_y_validation, y_prob[:,i],drop_intermediate=False)  ### TODO Results doen't match roc_auc_score..
            axs.plot(fpr,tpr,label = "Class "+ str(per_class) + " AUC : " + format(roc_auc,".4f")) 
            axs.legend()
            i+=1
    return roc_auc_dict

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

def buildInterval(loops,center):
    '''
    Helper function for cyclical searching. The function creates intervals arround some <center>, periodically reducing 
    the step size. 
    @start and @end can be customized into the function. 
    n = number of times the function has been called. 
    center = center of interval
    '''
    start:np.int16
    end:np.int16
    if loops > 1:
        if center >=100:
                start = center-40
                end = center+50
                stepSize = 10  
        else:
            start = center-10
            end = center+11
            stepSize = 2
            if center <= 10 : start = 1
            return np.arange(start,end,stepSize).astype(int)
    else:
            start = center-10
            end = center+11
            stepSize = 2
            if center <= 10 : start = 1  
    return np.arange(start,end,stepSize).astype(int)  
    

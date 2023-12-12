'''
Aqui vamos a poner 
todo lo necesario para hacer fincionet RF a ppartir de competition 2
'''
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics 
from sklearn.metrics import roc_auc_score,accuracy_score
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

class MLPModelTrainCycle:
    def __init__(self,
                 model,
                 modelName,
                 loss_fn,
                 optimizer,
                 labels,
                 pathTrainingDSet,
                 trainingParams, 
                 pathValidationDSet = None,
                 scheduler = None, 
                 initWeightfunc= None, 
                 initWeightParams= None, 
                 removeCoordinates = True, 
                 logger:ms.logg_Manager = None):
        
        self.train_dataset_path = pathTrainingDSet
        self.validation_dataset_path = pathValidationDSet
        self.model = model
        self.modelName = modelName
        self.logger = logger
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f' The device : {self.device}')
        self.NoCoordinates = removeCoordinates
        self.colsToDrop = trainingParams['colsToDrop']
        
        ### Load Dataset
        # Read Labels names
        self.logger.update_logs({'Data Set': self.train_dataset_path})
        self.labels = labels
        ## Load Training Dataset
        self.X, self.Y = self.load_Dataset(self.labels, self.train_dataset_path)
        
        ## Setting Parameters
        self.epochs = trainingParams['epochs']
        self.logger.update_logs({'Epochs': self.epochs})

        self.splitProportion = trainingParams['testSize']
        self.batchSize = trainingParams['batchSize']
        self.saveModelFolder = trainingParams['modelsFolder']
        self.criterion = loss_fn()
        self.initWeightParams = initWeightParams
        self.initWeightFunc = initWeightfunc  
        self.optimizer = optimizer
        self.optimiserInitialState = optimizer
        if scheduler is not None:
            self.scheduler = scheduler
            self.schedulerInitialState = scheduler
        ## Define training values
        self.presetting_Training(self.X)

    def splitData_asTensor(self,X,Y):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size= self.splitProportion, random_state=42)
        printDataBalace(X_train, X_test, y_train, y_test)
        return torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float), torch.tensor(y_train, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)

    def load_Dataset(self,labels,dataset):
        if self.NoCoordinates:
            X, Y = ms.importDataSet(dataset, labels, self.colsToDrop)
        else: 
            X, Y = ms.importDataSet(dataset, labels)
        return X, Y

    def read_split_as_tensor(self, X_train, y_train, X_test,y_test):
        return torch.tensor(X_train.values, dtype=torch.float), torch.tensor(X_test.values, dtype=torch.float), torch.tensor(y_train.values, dtype=torch.float), torch.tensor(y_test.values, dtype=torch.float)
        
    def presetting_Training(self,X_train):
        '''
        Place the necesary information together to set the training.
        '''
        input_size = X_train.shape[1]
        # Define Models's hidden layer sizes
        self.model = self.model(input_size)
        ### Define Optimizer and scheduler
        self.optimizer  = self.optimizer(self.model.parameters())
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)
            self.Sched = True    
        ### Init model parameters.
        if self.initWeightFunc  is not None:
                init_params(self.model, self.initWeightFunc, *self.initWeightParams) 
        pass

    def resetTraining(self):
        #Reset Model params to run in loop the K-fold validation.
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        ## Reset optimizer
        self.optimizer  = self.optimiserInitialState
        self.optimizer = self.optimizer(self.model.parameters())
        if self.Sched:
            self.scheduler = self.schedulerInitialState
            self.scheduler = self.scheduler(self.optimizer)

    def modelTrainer(self, kFold:bool = False, nSplits:int = 1):
        if kFold:
            self.logger.update_logs({'training Mode': 'KFold'})
            self.logger.update_logs({'N Folding': nSplits})
            return self.train_KFold(nSplits)
        else:
            self.logger.update_logs({'training Mode': 'Single train'})
            if self.validation_dataset_path is None:
                X_train, X_test, y_train, y_test = self.splitData_asTensor(self.X,self.Y)
            else:
                X_val, y_val = self.load_Dataset(self.labels,self.validation_dataset_path)
                X_train,X_test,y_train,y_test = self.read_split_as_tensor(self.X,self.Y,X_val, y_val)
                self.logger.update_logs({'Testing on Validation Dataset': self.validation_dataset_path})
          
          # Convert the training set into torch tensors
            self.model, trainMetrics =  self.train(X_train,X_test,y_train, y_test)
            modelName = str(self.modelName +'.pkl')
            model_Path = self.saveModelFolder + modelName
            self.logger.update_logs({'model Name': modelName})
            self.logger.update_logs({'metric': trainMetrics})
            ms.saveModel(self.model,model_Path)
            return self.model, trainMetrics

    def train(self, X_train,X_test, y_train, y_test)->[nn.Sequential,dict]:
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=self.batchSize, shuffle=True)
        # Train the model
        startTime = datetime.now()
        self.model.to(self.device)
        self.trainLosses = []
        self.valLosses = []
        # print('Model deivice',  next(self.model.parameters()).device)
        actual_lr = self.optimizer.param_groups[0]["lr"]
        print(f"Initial Lr : {actual_lr}")
        for e in range(1,self.epochs):
            loopLoss = []
            valLoss = []
            self.model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(inputs)
                loss = self.criterion(log_probs,labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                loopLoss.append(loss.item())
            self.trainLosses.append(np.mean(loopLoss))
            threshold = 0.5
            ## Evaluate in the test Set:
            with torch.no_grad():
                self.model.eval()
                inputs, labels = X_test.to(self.device), y_test.to(self.device)
                output = self.model(inputs)
                valLoss = self.criterion(output,labels.unsqueeze(1))
                self.valLosses.append(valLoss.item())
                y_hat = (output > threshold).float()
                accScore, macro_averaged_f1, micro_averaged_f1  = computeClassificationMetrics(y_test.cpu(),y_hat.cpu())
            
            if e%10 ==0:
                elapsed_time = datetime.now() - startTime
                avg_time_per_epoch = elapsed_time.total_seconds() / e
                remaining_epochs = self.epochs - e
                estimated_time = remaining_epochs * avg_time_per_epoch
                print(f"Elapsed Time after {e} epochs : {elapsed_time} ->> Estimated Time to End: {ms.seconds_to_datetime(estimated_time)}",' ->actual loss %.4f' %(self.trainLosses[-1]), '-> actual lr = %.6f' %(actual_lr))
                print('            Train metrics: accScore: %.4f '%(accScore), 'macro_averaged_f1 : %.4f'%(macro_averaged_f1), 'micro_averaged_f1: %.4f' %(micro_averaged_f1))
            if self.Sched:   
                self.scheduler.step(loss.item())
                actual_lr = self.optimizer.param_groups[0]["lr"]      
        ###  Return the final metrics
        metrics = {'accScore':accScore, 'macro_averaged_f1' : macro_averaged_f1, 
                'micro_averaged_f1' : micro_averaged_f1}
        print('Final train metrics: accScore: %.4f '%(accScore), 'macro_averaged_f1 : %.4f'%(macro_averaged_f1), 'micro_averaged_f1: %.4f' %(micro_averaged_f1))
        # self.logMLP()
        self.logger.update_logs({'Train metric': metrics})
        return self.model, metrics 

    def train_KFold(self,nSplits):
        # Define the KFold cross-validator
        kf = KFold(n_splits=nSplits)
        kf_iter = 1
        # Perform cross-validation
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X.loc[train_index], self.X.loc[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            X_train, y_train, X_test, y_test = self.read_split_as_tensor(X_train, y_train,X_test,y_test) 
            self.model, metrics = self.train(X_train, y_train, X_test, y_test)
            modelName = str(self.modelName + '_' + str(kf_iter) + '.pkl')
            model_Path = self.saveModelFolder + modelName
            self.logger.update_logs({'model Name': modelName})
            self.logger.update_logs({'Fold number': kf_iter})
            self.logger.update_logs({'Train metric': metrics})
            ms.saveModel(self.model,model_Path)
            self.resetTraining() 
            kf_iter +=1   
        return self.model, metrics



    def avaluateModel(self, model = None):
        threshold = 0.5
        X_val,Y_val = self.load_Dataset(self.labels,self.validation_dataset_path)
        if model is None:
            valModel = self.model
        else: 
            valModel = model
        with torch.no_grad():
            valModel.eval()
            inputs, labels = X_val.to(self.device), Y_val.to(self.device)
            output = valModel(inputs)
            valLoss = self.criterion(output,labels.unsqueeze(1))
            self.valLosses.append(valLoss.item())
            y_hat = (output > threshold).float()
            accScore, macro_averaged_f1, micro_averaged_f1  = computeClassificationMetrics(Y_val.cpu(),y_hat.cpu())
            metrics = {'accScore':accScore, 'macro_averaged_f1' : macro_averaged_f1, 
                'micro_averaged_f1' : micro_averaged_f1}
            self.logger.update_logs({'Validation metric': metrics})


    def plotLosses(self):
        epochs = range(len(self.trainLosses))
        plt.figure(figsize=(10,5))
        plt.plot(epochs, self.trainLosses, 'r-', label='Training loss')
        plt.plot(epochs, self.valLosses, 'b-', label='Validation loss')
        plt.title('Training loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        figName = str(self.modelName +'_losses.png')
        model_Path = self.saveModelFolder + figName
        plt.savefig(model_Path)
        plt.show()


    def getModel(self):
        return self.model
    
    def getLosses(self):
        return self.trainLosses


#### MODELS List
class MLP_1(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.ReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )

    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        return self.model(x)

class MLP_2(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )

    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        return self.model(x)

### Helper functions
def split(x,y,TestPercent = 0.2):
    x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=TestPercent)
    return x_train, x_validation, y_train, y_validation 

def printDataBalace(x_train, x_validation, y_train, y_validation):
    ## Data shape exploration
    print("",np.shape(x_train),"  :",np.shape(x_validation) )
    unique, counts = np.unique(y_train, return_counts=True)
    print("Label balance on Training set: ", "\n",unique,"\n",counts)
    unique, counts = np.unique(y_validation, return_counts=True)
    print("Label balance on Validation set: ","\n",unique,"\n",counts)

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

def computeClassificationMetrics(y_validation,y_hat):
    '''
    Ref: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification/notebook
    '''
    accScore = metrics.accuracy_score(y_validation, y_hat)
    macro_averaged_f1 = metrics.f1_score(y_validation, y_hat, average = 'macro') # Better for multiclass
    micro_averaged_f1 = metrics.f1_score(y_validation, y_hat, average = 'micro')
    # ROC_AUC_multiClass = roc_auc_score_calculation(y_validation,y_hat)
    # print('Accuraci_score: ', accScore)  
    # print('F1_macroAverage: ', macro_averaged_f1)  
    # print('F1_microAverage: ', micro_averaged_f1)
    # print('ROC_AUC one_vs_all: ', ROC_AUC_multiClass)
    return accScore, macro_averaged_f1, micro_averaged_f1 #, ROC_AUC_multiClass

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

def plot_ROC_AUC(y_test, y_hat, y_prob):
    '''
    Allows to plot multiclass classification ROC_AUC (One vs Rest), by computing tpr and fpr of each calss respect the rest. 
    The simplest application is the binary classification.
    '''
    unique_class = y_test.unique()
    unique_class.sort()
    print("UNIQUE CLASSES: ", unique_class)
    print("Test Set balance:" )
    listClassCountPercent(y_test)
    print("Prediction balance:")
    listClassCountPercent(y_hat)  
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
            axs.plot(fpr,tpr,label = "Class "+ unique_class[1] + " AUC : " + format(roc_auc,".4f")) 
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

def feature_importance_torch(model, dataloader, device):
    model.eval()
    feature_importances = []

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        acc = accuracy_score(labels.cpu(), preds.cpu())

        for j in range(inputs.shape[1]):  # iterate over each feature
            inputs_clone = inputs.clone()
            inputs_clone[:, j] = torch.rand_like(inputs_clone[:, j])  # permute feature values
            outputs = model(inputs_clone)
            _, preds = torch.max(outputs, 1)
            acc_permuted = accuracy_score(labels.cpu(), preds.cpu())
            feature_importances.append(acc - acc_permuted)  # importance is decrease in accuracy

    return feature_importances

def init_params(model, init_func, *params, **kwargs):
        '''
        Methode to initialize the model's parameters, according a function <init_func>,
            and the necessary arguments <*params, **kwargs>. 
        Change the type of <p> in <if type(p) == torch.nn.Conv2d:> for a different behavior. 
        '''
        for p in model.parameters():
            # if type(p) == (torch.nn.Conv2d or torch.nn.Conv1d) :   # Add a condition as needed. 
            init_func(p, *params)

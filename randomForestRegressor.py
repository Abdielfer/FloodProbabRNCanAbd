'''
Aqui vamos a poner 
todo lo necesario para hacer fincionet RF a ppartir de competition 2
'''
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer,accuracy_score,roc_curve, auc, roc_auc_score, f1_score
from sklearn.calibration import CalibrationDisplay

def importDataSet(dataSetName):
    '''
    Import datasets and filling NaN values          
    @input: DataSetName => The dataset path. 
    @Output: Features(x) and tragets(y)    
    ''' 
    train = pd.read_csv(dataSetName, index_col = None)
    y = train[['LABELS']]
    y = y.fillna(0)
    x = train.drop('LABELS', axis=1)
    xMean = x.mean()
    x = x.fillna(xMean)
    return x, y

def split(x,y,TestPercent = 0.2):
    x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=TestPercent)
    return x_train, x_validation, y_train, y_validation

def printDataBalace(x_train, x_validation, y_train, y_validation):
    ## Data shape exploration
    print("",np.shape(x_train),"  :",np.shape(x_validation) )
    print("Label balance on Training set: ", "\n", y_train['LABELS'].value_counts())
    print("Label balance on Validation set: ", "\n", y_validation['LABELS'].value_counts())

def predictOnFeaturesSet(model, featuresSet):
    y_hat = model.predict(featuresSet)
    return y_hat

def investigateFeatureImportance(x_train, classifier, printed = True):
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

def createsearshGreed():
    param_grid = {
    'n_estimators': np.linspace(90, 120,2).astype(int), #default 100
    'max_depth': [None],  # + list(np.linspace(3, 20).astype(int)),
    'max_features': ['auto', 'sqrt', None], # + list(np.arange(0.5, 1, 0.1)),
    'max_leaf_nodes': [None], # + list(np.linspace(10, 50, 500).astype(int)),
    # 'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
    }
    return param_grid


def executeRandomForestRegression():
    def __init__(self, dataSet, saveModelPath, splitProportion = 0.2):
        self.paramGrid = createsearshGreed()
        self.dataSet = dataSet
        self.seedRF = 50
        

        rfr_WithGridSearch = createModelRFRegressorWithGridSearsh()

        bestRegressor = fitModelRFRegressor(rfr_WithGridSearch)
        return bestRegressor
    
    def createModelRFRegressorWithGridSearsh(self):
        estimator = RandomForestRegressor(random_state = self.seedRF)
        scoring = {"AUC": "roc_auc", "Accuracy": accuracy_score, 
                    "roc_auc_Score":roc_auc_score, "F1":f1_score}
        rs = GridSearchCV(estimator, 
                        param_grid = self.paramGrid,
                        n_jobs = -1, 
                        scoring = scoring,
                        refit="AUC",
                        cv = 3, 
                        n_iter = 10,
                        verbose = 1, 
                        random_state=self.seedRF,
                        return_train_score = True
                        )
        return rs
  
    def fitModelRFRegressor(self, rfr_WithGridSearch, x_train, y_train, saveModel = True):
        y_train = np.array(y_train)
        rfr_WithGridSearch.fit(x_train, y_train).ravel()
        print(rfr_WithGridSearch.best_params_, "\n")
        ### Working with best estimator from RandomizedSearch 
        best_model = rs.best_estimator_
        if saveModel:
            joblib.dump(best_model, "rf_RandomSearch.pkl")
        ## Evaluating ROC Curve and extracting features priority
        fi_model = evaluate_model(x_train, y_train, x_validation, y_validation, test_nolabels_prediction)
        
        return bestEstimator

     


## Trash ...
def metric_RocAuc(y_probability, y_validation, estimator_name):
    '''
    Calculate and plt ROC metric
    @argument: y_probability : the probability class=1.
    @argument: y_validation: True labels.
    fpr, tpr = false_positive, true_positive.
    Return: "false_positive" and "true_positive", ROC_auc metric.
    '''
    fpr, tpr, _ = roc_curve(y_validation, y_probability) 
    roc_auc = auc(fpr, tpr)
    fig, axes = plt.subplots(constrained_layout=True,figsize=(5,3), dpi=150)
    fig.suptitle(estimator_name)
    axes.plot([0, 1], [0, 1], color= 'k',linestyle="--") # perfect fit 
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                       estimator_name=estimator_name)
    display.plot(ax=axes)
    return fpr, tpr, roc_auc



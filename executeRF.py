import pandas as pd
import myServices as ms
import models as m
import hydra
from omegaconf import DictConfig
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier


def executeRFRegressor(cfg: DictConfig):
    log = {}
    name = m.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    # penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    rForestReg = m.implementRandomForestRegressor(pathDataset,'percentage', percentOfValidation, arg)
    best_estimator, bestParameters, r2_validation= rForestReg.fitRFRegressorGSearch()
    m.saveModel(best_estimator,name)
    _,x_validation,_,_ = rForestReg.getSplitedDataset()
    regressorName, featureImportance = m.investigateFeatureImportance(best_estimator,name,x_validation)
    print(r2_validation)
    # Fill Log
    log['dataset'] = cfg['pathDataset']
    log['model_Id'] = name
    log['regressor_Name'] = regressorName
    log['best_param'] = bestParameters
    log['features_Importance'] = pd.DataFrame(featureImportance).to_string(justify= 'left')
    log['r2_score'] = r2_validation 
    return best_estimator,name,log


def executeRFRegressorWeighted(cfg: DictConfig):
    log = {}
    name = m.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    rForestReg = m.implementRandomForestRegressor(pathDataset,'percentage', percentOfValidation, arg)
    best_estimator, bestParameters, r2_validation= rForestReg.fitRFRegressorWeighted(penalty)
    m.saveModel(best_estimator,name)
    _,x_validation,_,_ = rForestReg.getSplitedDataset()
    regressorName, featureImportance = m.investigateFeatureImportance(best_estimator,name,x_validation)
    print(r2_validation)
    # Fill Log
    log['dataset'] = cfg['pathDataset']
    log['model_Id'] = name
    log['regressor_Name'] = regressorName
    log['best_param'] = bestParameters
    log['features_Importance'] = pd.DataFrame(featureImportance).to_string(justify= 'left')
    log['r2_score'] = r2_validation 
    return best_estimator,name,log

def executeRFCalssifier(cfg: DictConfig):
    log = {}
    name = m.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    arg = cfg.parameters
    rfClassifier = m.implementRandomForestCalssifier(pathDataset,'percentage', percentOfValidation, arg)
    _,x_validation,_,_ = rfClassifier.getSplitedDataset()
    best_estimator, bestParameters = rfClassifier.fitRFClassifierGSearch()
    classifierName, featureImportance = m.investigateFeatureImportance(best_estimator,name,x_validation)
    accScore, macro_averaged_f1, micro_averaged_f1, ROC_AUC_multiClass = rfClassifier.computeClassificationMetrics(best_estimator)
    log['dataset'] = cfg['pathDataset']
    log['model_Id'] = name
    log['model_Name'] = classifierName
    log['best_param'] = bestParameters
    log['features_Importance'] = featureImportance.to_dict('tight')
    log['Accuraci_score'] = accScore
    log['macro_averaged_f1'] = macro_averaged_f1
    log['micro_averaged_f1'] = micro_averaged_f1
    log['ROC_AUC_multiClass'] = ROC_AUC_multiClass
    return best_estimator,name,log

def testOneVsAll(cfg: DictConfig):
    log = {}
    name = m.makeNameByTime()
    local = cfg.local
    pathTrainDataset = local + cfg['pathTrainingDataset']
    pathValidationDataset = local + cfg['pathValidationDataset']
    estimator = RandomForestClassifier(criterion='entropy', random_state = 50)
    x_train,y_train = ms.importDataSet(pathTrainDataset, 'percentage')
    classifier = OneVsRestClassifier(estimator).fit(x_train,y_train)
    x_validation,y_validation = ms.importDataSet(pathValidationDataset, 'percentage')
    y_hat = classifier.predict(x_validation)
    accScore = metrics.accuracy_score(y_validation, y_hat)
    macro_averaged_f1 = metrics.f1_score(y_validation, y_hat, average = 'macro') # Better for multiclass
    micro_averaged_f1 = metrics.f1_score(y_validation, y_hat, average = 'micro')
    ROC_AUC_multiClass = m.roc_auc_score_multiclass(y_validation,y_hat)
    log['pathTrainDataset'] = cfg['pathTrainDataset']
    log['pathValidationDataset'] = cfg['pathValidationDataset']
    log['model_Id'] = name
    log['model_Name'] = "classifier"
    log['best_param'] = classifier.get_params
    log['Accuraci_score'] = accScore
    log['macro_averaged_f1'] = macro_averaged_f1
    log['micro_averaged_f1'] = micro_averaged_f1
    log['ROC_AUC_multiClass'] = ROC_AUC_multiClass
    return classifier,name,log

@hydra.main(config_path=f"config", config_name="configClassOnevsAll.yaml")
def main(cfg: DictConfig):
    best_estimator,name, log = testOneVsAll(cfg)
    m.saveModel(best_estimator, name)
    logToSave = pd.DataFrame.from_dict(log, orient='index')
    logToSave.to_csv(name +'.csv',index = True, header=True) 

if __name__ == "__main__":
    with ms.timeit():
        main()
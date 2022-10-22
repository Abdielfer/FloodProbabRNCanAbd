import numpy as np
import pandas as pd
import myServices as ms
import models as m
import hydra
from omegaconf import DictConfig


def executeRFRegressor(cfg: DictConfig):
    log = {}
    name = ms.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    # penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    rForestReg = m.implementRandomForestRegressor(pathDataset,cfg['targetColName'], percentOfValidation, arg)
    best_estimator, bestParameters, r2_validation= rForestReg.fitRFRegressorGSearch()
    m.saveModel(best_estimator,name)
    _, featureImportance = m.investigateFeatureImportance(best_estimator,name,x_validation)
    print(r2_validation)
    # Fill Log
    log['dataset'] = cfg['pathTrainingDataset']
    log['model_Id'] = name
    log[''] = pd.DataFrame(featureImportance).to_string(justify= 'left')
    log['r2_score'] = r2_validation 
    return best_estimator,name,log

def executeRFRegressorWeighted(cfg: DictConfig):
    name = ms.makeNameByTime()
    local = cfg.local
    pathTrainingDataset = local + cfg.pathTrainingDataset
    pathTestDataset = local + cfg.pathTestDataset
    penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    rForestReg = m.implementRandomForestRegressor(pathTrainingDataset, cfg['targetColName'], arg)
    x_validation, y_validation = ms.importDataSet(pathTestDataset, cfg['targetColName'])
    x_validation_Clean = x_validation.copy()
    x_validation_Clean.drop(['x_coord','y_coord'], axis=1, inplace = True)
    print("Test set  balance")
    total, classCountPercent = m.listClassCountPercent(y_validation)
    print(classCountPercent)
    best_estimator, _ = rForestReg.fitRFRegressorWeighted(penalty)
    r2_validation = m.validateWithR2(best_estimator, x_validation_Clean, y_validation,
                                     cfg['weightPenalty'], cfg['dominantClass'], weighted = cfg['weighted'])
    print("R2_score for validation set: ", r2_validation)
    featureImportance = m.investigateFeatureImportance(best_estimator,x_validation_Clean)
    print(r2_validation)
    prediction = ms.makePredictionToImportAsSHP(best_estimator, x_validation, y_validation, cfg['targetColName'])
    # Fill Log
    log = {}
    log['dataset'] = cfg['pathTrainingDataset']
    log['Testing_samples'] = total
    log['Class_Count_percent'] = classCountPercent
    log['model'] = best_estimator
    log['model_Id'] = name
    log[''] = pd.DataFrame(featureImportance).to_string(justify= 'left')
    log['r2_score'] = r2_validation 
    return best_estimator,name,log, prediction

def executeRFCalssifier(cfg: DictConfig):
    name = ms.makeNameByTime()
    local = cfg.local
    pathTrainingDataset = local + cfg['pathTrainingDataset']
    pathTestDataset = local + cfg['pathTestDataset']
    arg = cfg.parameters
    rfClassifier = m.implementRandomForestCalssifier(pathTrainingDataset,cfg['targetColName'], arg)
    x_validation, y_validation = ms.importDataSet(pathTestDataset, cfg['targetColName'])
    x_validation_Clean = x_validation.copy()
    x_validation_Clean.drop(['x_coord','y_coord'], axis=1, inplace = True)
    print("Test set  balance")
    total, classCountPercent = m.listClassCountPercent(y_validation)
    print(classCountPercent)
    best_estimator, _ = rfClassifier.fitRFClassifierGSearch()
    featureImportance = m.investigateFeatureImportance(best_estimator, x_validation_Clean)
    accScore, macro_averaged_f1, micro_averaged_f1, ROC_AUC_multiClass = m.computeClassificationMetrics(best_estimator,x_validation_Clean,y_validation)
    ## Make Prediction
    prediction = ms.makePredictionToImportAsSHP(best_estimator, x_validation, y_validation, cfg['targetColName'])
    # Log
    log = {}
    log['model_Id'] = name
    log['model'] = best_estimator
    log['dataset'] = cfg['pathTrainingDataset']
    log['Training samples'] = total
    log['Class_Count_percent'] = classCountPercent
    log[''] = pd.DataFrame(featureImportance).to_string(justify= 'left')
    log['Accuraci_score'] = accScore
    log['macro_averaged_f1'] = macro_averaged_f1
    log['micro_averaged_f1'] = micro_averaged_f1
    log['ROC_AUC_multiClass'] = ROC_AUC_multiClass
    return best_estimator,name,log, prediction

def executeOneVsAll(cfg: DictConfig):
    name = ms.makeNameByTime()
    local = cfg.local
    pathTrainingDataset = local + cfg['pathTrainingDataset']
    pathValidationDataset = local + cfg['pathValidationDataset']
    arg = cfg.parameters  
    oneVsAllClassifier = m.implementOneVsRestClassifier(pathTrainingDataset,'percentage', arg)
    print("Fitting ")
    model,best_params = oneVsAllClassifier.fitOneVsRestClassifierGSearch()
    x_validation,y_validation = ms.importDataSet(pathValidationDataset, 'percentage')
    print("Computing metrics ")
    accScore, macro_averaged_f1, micro_averaged_f1, ROC_AUC_multiClass = m.computeClassificationMetrics(model,x_validation,y_validation)
    # print(f"ROC_AUC_multiClass: {ROC_AUC_multiClass}")
    log = {}
    log['pathTrainingDataset'] = cfg['pathTrainingDataset']
    log['pathValidationDataset'] = cfg['pathValidationDataset']
    log['model_Id'] = name
    log['model'] = model
    log['Accuraci_score'] = accScore
    log['macro_averaged_f1'] = macro_averaged_f1
    log['micro_averaged_f1'] = micro_averaged_f1
    log['ROC_AUC_multiClass'] = ROC_AUC_multiClass
    return oneVsAllClassifier,name,log


def excecuteMLPClassifier(cfg: DictConfig):
    name = ms.makeNameByTime()
    modelParams = cfg.parameters 
    pathTrainingDataset = cfg.local + cfg['pathTrainingDataset']
    pathValidationDataset = cfg.local + cfg['pathTestDataset']
    mlpc = m.implementingMLPCalssifier(pathTrainingDataset,cfg['targetColName'], modelParams)
    print("Exploring best Hyper parameter >>>>>>>>>>> ")
    firstInterval =  eval(cfg['firstInterval'])
    x_val,Y_val = ms.importDataSet(pathValidationDataset, cfg['targetColName'])
    X = x_val.copy()
    X.drop(['x_coord','y_coord'], axis=1, inplace=True)
    betsHLS = int(mlpc.explore4BestHLSize(X,Y_val,firstInterval,cfg['clasOfInterest'],2))
    print('bestHLT ---- : ', betsHLS)
    ## Evaluating best parametere..
    modelParams['verbose'] = True
    modelParams['hidden_layer_sizes'] = betsHLS
    print(modelParams)
    mlpc.restartMLPCalssifier(modelParams)
    print("Training with best Hyper parameter >>>>>>>>>>> ")
    mlpc.fitMLPClassifier()
    bestMLPC = mlpc.getMLPClassifier()
    y_hat = bestMLPC.predict(X.values)
    ROC_AUC_multiClass = m.roc_auc_score_multiclass(Y_val,y_hat)
    mlpc.logMLPClassifier({'ROC_AUC_multiClass': ROC_AUC_multiClass})
    logs = mlpc.get_logsDic()
    print(logs)
    prediction = ms.makePredictionToImportAsSHP(bestMLPC, x_val,Y_val, cfg['targetColName'])
    return bestMLPC, name, prediction,logs


@hydra.main(config_path=f"config", config_name="configMLPClassifier.yaml")
def main(cfg: DictConfig):
    best_estimator, name, prediction, logs = excecuteMLPClassifier(cfg)
    ms.saveModel(best_estimator, name)
    predictionName = name + "_prediction_" + cfg['pathTrainingDataset']
    prediction.to_csv(predictionName, index = True, header=True)  
    logToSave = pd.DataFrame.from_dict(logs, orient='index')
    logToSave.to_csv(name +'.csv',index = True, header=True) 

if __name__ == "__main__":
    with ms.timeit():
        main()
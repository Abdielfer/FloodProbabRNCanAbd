import pandas as pd
import myServices 
import randForest as r
import hydra
from omegaconf import DictConfig

def executeRFRegressor(cfg: DictConfig):
    log = pd.DataFrame()
    name = r.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    # penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    frRegGS = r.implementRandomForestRegressor(pathDataset,'percentage', percentOfValidation, arg)
    best_estimator, bestParameters, r2_validation= frRegGS.fitRFRegressorGSearch()
    r.saveModel(best_estimator,name)
    _,x_validation,_,_ = frRegGS.getSplitedDataset()
    regressorName, featureImportance = r.investigateFeatureImportance(best_estimator,name,x_validation)
    print(r2_validation)
    # Fill Log
    log['dataset'] = cfg['pathDataset']
    log['model_Id'] = name
    log['regressor_Name'] = regressorName,
    log['best_param'] = bestParameters
    log['features_Importance'] = featureImportance
    log['r2_score'] = r2_validation 
    return best_estimator, log

def executeRFCalssifier(cfg: DictConfig):
    log = pd.DataFrame()
    name = r.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    # penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    frRegGS = r.implementRandomForestCalssifier(pathDataset,'percentage', percentOfValidation, arg)
    _,x_validation,_,y_validation = frRegGS.getSplitedDataset()
    best_estimator, bestParameters, r2_validation= frRegGS.fitRFClassifierGSearch()
    classifierName, featureImportance = r.investigateFeatureImportance(best_estimator,name,x_validation)
    accScore, macro_averaged_f1, micro_averaged_f1, ROC_AUC_multiClass = frRegGS.computeClassificationMetrics(best_estimator,x_validation,y_validation)
    log['dataset'] = cfg['pathDataset']
    log['model_Id'] = name
    log['regressor_Name'] = classifierName
    log['best_param'] = bestParameters
    log['features_Importance'] = featureImportance
    log['Accuraci_score'] = accScore
    log['macro_averaged_f1'] = macro_averaged_f1
    log['micro_averaged_f1'] = micro_averaged_f1
    log['ROC_AUC_multiClass'] = ROC_AUC_multiClass

@hydra.main(config_path=f"config", config_name="config.yaml")
def main(cfg: DictConfig):
    best_estimator, log = executeRFRegressor(cfg)
    r.saveModel(best_estimator)
    log.to_csv({best_estimator.__name__}+'.csv')

if __name__ == "__main__":
    with myServices.timeit():
        main()
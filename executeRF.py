import pandas as pd
import myServices 
import randForest as r
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=f"config", config_name="config.yaml")
def main(cfg: DictConfig):
    log = pd.DataFrame()
    name = r.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    # penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    frRegGS = r.implementRandomForestRegressor(pathDataset,'percentage', percentOfValidation, arg)
    best_estimator, r2 = frRegGS.fitRFRegressorGSearch()
    r.saveModel(best_estimator,name)
    _,x_validation,_,_ = frRegGS.getSplitedDataset()
    casiffierName, featureImportance = r.investigateFeatureImportance(best_estimator,name,x_validation)
    print(r2)

    # Fill Log
    log['dataset'] = cfg['pathDataset']
    log['model+Id'] = name
    log['classifierName'] = casiffierName, featureImportance
    log['best_param'] = best_estimator
    log['r2_score'] = r2  # UNCOMENT For Regression Only

if __name__ == "__main__":
    with myServices.timeit():
        main()
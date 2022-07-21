import myServices 
import randForReg as r
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=f"config", config_name="config.yaml")
def main(cfg: DictConfig):
    name = r.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    arg = cfg.parameters
    frRegGS = r.implementRandomForestRegressor(pathDataset,'percentage', percentOfValidation, arg)
    best_estimator, r2 = frRegGS.fitRFRegressorGSearch()
    print(r2)
    r.saveModel(best_estimator,name)

if __name__ == "__main__":
    with myServices.timeit():
        main()
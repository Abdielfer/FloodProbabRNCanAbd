from re import M
import pandas as pd
import myServices as ms
import randForest as r
from sklearn import metrics
import hydra
from omegaconf import DictConfig

class getEstimatorMetric():
    def __init__(self,cfg:DictConfig):
        local = cfg.local
        self.pathDataset = local + cfg['pathDataset']
        self.model = cfg.model
        self.targetColName = cfg.targetColName
        self.x_validation, self.y_validation = ms.importDataSet(self.pathDataset, self.targetColName)
        pass

    def classifierMetric(self):
        classifier = r.loadModel(self.model)
        y_hat = classifier.predict(self.x_validation)
        accScore = metrics.accuracy_score(self.y_validation, y_hat)
        macro_averaged_f1 = metrics.f1_score(self.y_validation, y_hat, average = 'macro') # Better for multiclass
        micro_averaged_f1 = metrics.f1_score(self.y_validation, y_hat, average = 'micro')
        ROC_AUC_multiClass = r.roc_auc_score_multiclass(self.y_validation,y_hat)
        accScore, macro_averaged_f1, micro_averaged_f1, ROC_AUC_multiClass = classifier.computeClassificationMetrics(best_estimator)
        metric = {}
        metric['accScore'] = accScore
        metric['macro_averaged_f1'] = macro_averaged_f1
        metric['micro_averaged_f1'] = micro_averaged_f1
        metric['ROC_AUC_multiClass'] = ROC_AUC_multiClass
        return metric

    def regressorMetric(self):
        regressor = r.loadModel(self.model)
        r2_validation = r.validateWithR2(regressor,self.x_validation,self.y_validation,0,0.1, weighted = False)
        return r2_validation
            
    # Defining a function to decide which function to call
    def metric(self, modelType: str):
        method = switch(self,modelType)
        if method != False:
            return getattr(self, method)()
        else: 
            return print("Invalide model type. Vefify configurations")

def switch(self,model):
    dict={
        'regressor': getEstimatorMetric.regressorMetric(),
        'classifier': getEstimatorMetric.classifierMetric(),
    }
    return dict.get(model, False)

# Calling the switch case method
def computeMetric(cfg: DictConfig):
    model = cfg.model
    estimatorMetric = getEstimatorMetric(cfg)
    metricMethod = estimatorMetric.metric(cfg.modelType)
    metric = metricMethod(model)
    return metric


@hydra.main(config_path=f"config", config_name="configEval.yaml")
def main(cfg: DictConfig):
    metric = computeMetric(cfg)
    print(metric)
    # metricToSave = pd.DataFrame.from_dict(metric, orient='index')
    # metricToSave.to_csv('metric.csv',index = True, header=True)

if __name__ == "__main__":
    with ms.timeit():
        main()



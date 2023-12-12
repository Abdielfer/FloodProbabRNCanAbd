
import myServices as ms
import models as m

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig,OmegaConf


def excecuteMLPClassifier(cfg: DictConfig,logManager:ms.logg_Manager):
    ''' 
    model, modelName, loss_fn,optimizer, labels, pathTrainingDSet,trainingParams, pathValidationDSet = None,scheduler = None, initWeightfunc= None, initWeightParams= None, removeCoordinates = True,logger:ms.logg_Manager = None)
    '''
    # MOdel
    modelChoice = OmegaConf.create(cfg.parameters['model'])
    model = instantiate(modelChoice)
    modelName = ms.makeNameByTime()
    logManager.update_logs({'model name': modelName})    
    
    # Loss
    loss = OmegaConf.create(cfg.parameters['criterion'])
    loss_fn = instantiate(loss)
    logManager.update_logs({'loss function': loss})   
    
    # Optimizer & Scheduler
    optimizerOptions = OmegaConf.create(cfg.parameters['optimizer'])
    optimizer = instantiate(optimizerOptions)
    logManager.update_logs({'optimizer': optimizerOptions})   
    
    schedulOptions = OmegaConf.create(cfg.parameters['scheduler'])
    scheduler = instantiate(schedulOptions)
    logManager.update_logs({'schedule': schedulOptions})   
    
    # Labels
    Labels = cfg['targetColName']

    # Datasets
    pathTrainingDataset = cfg.datasetFolder  + cfg['trainingDataset']
    pathValidationDataset = cfg.datasetFolder  + cfg['validationDataset']
    
    # Parameter initialization
    initWeightFunc = instantiate(OmegaConf.create(cfg.parameters['init_weight']))
    initWeightParams = cfg.parameters['initWeightParams']
    trainingParams = cfg['parameters']  
    
    # Modeling tool instanciate
    mlpc = m.MLPModelTrainCycle(model,modelName,loss_fn,optimizer,Labels,pathTrainingDataset,trainingParams,pathValidationDataset,scheduler=scheduler,initWeightfunc=initWeightFunc, initWeightParams= initWeightParams,logger=logManager)
    
    ## Excecute Training 
    kFold = cfg.parameters['kFold']
    if kFold:
        nSplits = cfg.parameters['n_folds']
        model,metrics = mlpc.modelTrainer(kFold=True,nSplits=nSplits)
    else:
        model,metrics = mlpc.modelTrainer()
    
    # Plot Losses
    mlpc.plotLosses()
    #  logs = mlpc.get_logsDic()
    # print(logs)
    # prediction = ms.makePredictionToImportAsSHP(bestMLPC, x_val,Y_val, cfg['targetColName'])
    return model, metrics #, logs

@hydra.main(version_base=None,config_path=f"config", config_name="configMLPClassifier.yaml")
def main(cfg: DictConfig):
    #### Set Logging
    logManager = ms.logg_Manager()
    ## Training 
    model,_ = excecuteMLPClassifier(cfg,logManager)

    # # predictionName = name + "_prediction_" + cfg['pathTrainingDataset']
    # prediction.to_csv(predictionName, index = True, header=True)  
    # logToSave = pd.DataFrame.from_dict(logs, orient='index')
    # logToSave.to_csv(name +'.csv',index = True, header=True) 
    

if __name__ == "__main__":
    with ms.timeit():
        main()
        
    #### Extract Dataframes for Class 1 and 5
    # csv = r'C:\Users\abfernan\CrossCanFloodMapping\FloodProbabRNCanAbd\datasets\AL_Lethbridge_FullBasin_Cilp_FullDataset.csv'
    # ms.extractFloodClassForMLP(csv)

   #######    Preprocessing For MLP
    # ### Concat all datasets with all classes
    # #####  Concat Datasets
    # csvList = ms.createListFromCSV(cfg.datasetsList)
    # out = ms.addSubstringToName(cfg.datasetsList,'_Concat')
    # ms.concatDatasets(csvList,out)

    # ### Clean Datasets from elevation > 0
    # outClean = ms.addSubstringToName(out,'_Clean')
    # DF = pd.read_csv(out,index_col=None)
    # DF = DF[DF.Cilp>=0]
    # print(DF.describe())
    # DF.to_csv(outClean, index=None)456+

    # ### Divide dataset by classes
    # csvClass1, csvClass5 = ms.extractFloodClassForMLP(outClean)
    
    # ### Copmute balance undersampling for Class 1 
    # DF = pd.read_csv(csvClass1,index_col=None)
    # Y = DF['Labels']
    # DF.drop(['Labels'], axis=1, inplace = True)
    # x_underSamp, y_undersamp = ms.randomUndersampling(DF,Y)
    # print(len(x_underSamp), '---', len(y_undersamp))
    # x_underSamp['Labels']= y_undersamp
    # print(x_underSamp.head)
    # out = ms.addSubstringToName(csvClass1,'_Balance')
    # x_underSamp.to_csv(out, index=None)
    
    # ### Copmute balance undersampling for Class 5 
    # DF = pd.read_csv(csvClass5,index_col=None)
    # Y = DF['Labels']
    # DF.drop(['Labels'], axis=1, inplace = True)
    # x_underSamp, y_undersamp = ms.randomUndersampling(DF,Y)
    # print(len(x_underSamp), '---', len(y_undersamp))
    # x_underSamp['Labels']= y_undersamp
    # print(x_underSamp.head)
    # out = ms.addSubstringToName(csvClass5,'_Balance')
    # x_underSamp.to_csv(out, index=None)
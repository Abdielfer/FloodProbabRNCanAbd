
import myServices as ms
import models as m
import sys 

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig,OmegaConf

from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context

def evaluateExternalModels(cfg:DictConfig,logManager:ms.logg_Manager):
    ''' 
    
    '''
    ## Current wdr
    orig_cwd = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Origin wDir -> {orig_cwd}")

    # Model
    model = cfg.modelsFolder + cfg.externalModel
    modelChoice =  ms.loadModel(model) 
    modelName = ms.makeNameByTime()
    logManager.update_logs({'model': modelName})       
    
    # Loss
    loss = OmegaConf.create(cfg.parameters['criterion'])
    loss_fn = instantiate(loss)
    
    # Optimizer & Scheduler
    optimizerOptions = OmegaConf.create(cfg.parameters['optimizer'])
    optimizer = instantiate(optimizerOptions)
    
    schedulOptions = OmegaConf.create(cfg.parameters['scheduler'])
    scheduler = instantiate(schedulOptions)
    
    # Labels
    Labels = cfg['targetColName']

    # Datasets
    pathValidationDataset = cfg.datasetFolder  + cfg['validationDataset']
    
    # Parameter initialization
    initWeightFunc = instantiate(OmegaConf.create(cfg.parameters['init_weight']))
    initWeightParams = cfg.parameters['initWeightParams']
    trainingParams = cfg['parameters']  
    
    # Modeling tool instanciate
    mlpc = m.MLPModelTrainCycle(model,modelName,loss_fn,optimizer,Labels,trainingParams,pathValidationDataset,scheduler=scheduler,initWeightfunc=initWeightFunc, initWeightParams= initWeightParams,logger=logManager, nWorkers = cfg.nWorkers)
    
    ### Evaluate the model.
    Y_val,y_hat, metrics = mlpc.evaluateModel(model)
    mlpc.plot_ConfussionMatrix(Y_val,y_hat,showIt=True, saveTo = orig_cwd)

    # Plot Losses
    # mlpc.plotLosses(showIt= False, saveTo = orig_cwd)
    # mlpc.plot_ConfussionMatrix(showIt= False, saveTo = orig_cwd)
    # prediction = ms.makePredictionToImportAsSHP(bestMLPC, x_val,Y_val, cfg['targetColName'])
    return model,metrics 

    pass

def excecuteMLPClassifier(cfg:DictConfig,logManager:ms.logg_Manager,train:bool = True, evaluate:bool= False, externalModel = None):
    ''' 
    model, modelName, loss_fn,optimizer, labels, pathTrainingDSet,trainingParams, pathValidationDSet = None,scheduler = None, initWeightfunc= None, initWeightParams= None, removeCoordinates = True,logger:ms.logg_Manager = None)
    '''
    ## Current wdr
    orig_cwd = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Origin wDir -> {orig_cwd}")

    # Model
    modelChoice = OmegaConf.create(cfg.parameters['model'])
    model = instantiate(modelChoice)
    modelName = ms.makeNameByTime()
    logManager.update_logs({'model name': modelName})    
    logManager.update_logs({'model': modelChoice})    
    
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
    mlpc = m.MLPModelTrainCycle(model,modelName,loss_fn,optimizer,Labels,pathTrainingDataset,trainingParams,pathValidationDataset,scheduler=scheduler,initWeightfunc=initWeightFunc, initWeightParams= initWeightParams,logger=logManager, nWorkers = cfg.nWorkers)
    
    ## Excecute Training 
    model,metrics = mlpc.modelTrainer()
    
    ### Evaluate the model.
    y_val,y_hat,_ = mlpc.evaluateModel()
    mlpc.plot_ConfussionMatrix(y_val,y_hat,saveTo = orig_cwd)

    # Plot Losses
    mlpc.plotLosses(showIt= False, saveTo = orig_cwd)
    # mlpc.plot_ConfussionMatrix(showIt= False, saveTo = orig_cwd)
    # prediction = ms.makePredictionToImportAsSHP(bestMLPC, x_val,Y_val, cfg['targetColName'])
    return model,metrics #, logs

@hydra.main(version_base=None,config_path=f"config", config_name="configMLPClassifier.yaml")
def main(cfg: DictConfig):
    print("-------------------   NEW Training -----------------------")
    #### Set Logging
    logManager = ms.logg_Manager() 
    _,_ = excecuteMLPClassifier(cfg,logManager, evaluate=True)

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

    # ### Clean Datasets from elevation < 0
    # outClean = ms.addSubstringToName(out,'_Clean')
    # DF = pd.read_csv(out,index_col=None)
    # DF = DF[DF.Cilp>=0]
    # print(DF.describe())
    # DF.to_csv(outClean, index=None)456+

    # ### Divide dataset by classes
    # csvClass1, csvClass5 = ms.extractFloodClassForMLP(outClean)
    
    # ### Compute balance undersampling for Class 1 
    # DF = pd.read_csv(csvClass1,index_col=None)
    # Y = DF['Labels']
    # DF.drop(['Labels'], axis=1, inplace = True)
    # x_underSamp, y_undersamp = ms.randomUndersampling(DF,Y)
    # print(len(x_underSamp), '---', len(y_undersamp))
    # x_underSamp['Labels']= y_undersamp
    # print(x_underSamp.head)
    # out = ms.addSubstringToName(csvClass1,'_Balance')
    # x_underSamp.to_csv(out, index=None)
    
    # ### Compute balance undersampling for Class 5 
    # DF = pd.read_csv(csvClass5,index_col=None)
    # Y = DF['Labels']
    # DF.drop(['Labels'], axis=1, inplace = True)
    # x_underSamp, y_undersamp = ms.randomUndersampling(DF,Y)
    # print(len(x_underSamp), '---', len(y_undersamp))
    # x_underSamp['Labels']= y_undersamp
    # print(x_underSamp.head)
    # out = ms.addSubstringToName(csvClass5,'_Balance')
    # x_underSamp.to_csv(out, index=None)
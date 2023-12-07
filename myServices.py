import os
import glob
import pathlib
import shutil
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler 
import logging
import hydra
from hydra.core.hydra_config import HydraConfig

### General applications ##
class timeit(): 
    '''
    to compute execution time do:
    with timeit():
         your code, e.g., 
    '''
    def __enter__(self):
        self.tic = datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(datetime.now() - self.tic))

def seconds_to_datetime(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}:{minutes}:{seconds}'

def makeNameByTime():
    name = time.strftime("%y%m%d%H%M")
    return name

def get_parenPath_name_ext(filePath):
    '''
    Ex: user/folther/file.txt
    parentPath = pathlib.PurePath('/src/goo/scripts/main.py').parent 
    parentPath => '/src/goo/scripts/'
    parentPath: can be instantiated.
         ex: parentPath[0] => '/src/goo/scripts/'; parentPath[1] => '/src/goo/', etc...
    '''
    parentPath = pathlib.PurePath(filePath).parent
    fpath = pathlib.Path(filePath)
    ext = fpath.suffix
    name = fpath.stem
    return parentPath, name, ext
  
def addSubstringToName(path, subStr: str, destinyPath = None) -> os.path:
    '''
    @path: Path to the raster to read. 
    @subStr:  String o add at the end of the origial name
    @destinyPath (default = None)
    '''
    parentPath,name,ext= get_parenPath_name_ext(path)
    if destinyPath != None: 
        return os.path.join(destinyPath,(name+subStr+ext))
    else: 
        return os.path.join(parentPath,(name+subStr+ ext))

def createListFromCSV(csv_file_location: os.path, delim:str =',')->list:  
    '''
    @return: list from a <csv_file_location>.
    Argument:
    @csv_file_location: full path file location and name.
    '''       
    df = pd.read_csv(csv_file_location, index_col= None, header=None, delimiter=delim)
    out = []
    for i in range(0,df.shape[0]):
        out.append(df.iloc[i][0])
    return out

## modeling manipulation
def saveModel(estimator, name):
    _ = joblib.dump(estimator, name, compress=9)

def loadModel(modelName):
    return joblib.load(modelName)

def logTransformation(x):
    '''
    Logarithmic transformation to redistribute values between 0 and 1. 
    '''
    x_nonZeros = np.where(x <= 0.0000001, 0.0001, x)
    return np.max(np.log(x_nonZeros)**2) - np.log(x_nonZeros)**2

def createWeightVector(y_vector, dominantValue:float, dominantValuePenalty:float):
    '''
    Create wight vector for sampling weighted training.
    The goal is to penalize the dominant class. 
    This is important is the flood study, where majority of points (usually more than 95%) 
    are not flooded areas. 
    '''
    y_ravel  = (np.array(y_vector).astype('int')).ravel()
    weightVec = np.ones_like(y_ravel).astype(float)
    weightVec = [dominantValuePenalty if y_ravel[j] == dominantValue else 1 for j in range(len(y_ravel))]
    return weightVec

####  Sampling manipulation
def stratifiedSampling(dataSetName, targetColName):
    '''
    Performe a sampling that preserve classses proportions on both, train and test sets.
    '''
    X,Y = importDataSet(dataSetName, targetColName)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=50)
    for train_index, test_index in sss.split(X, Y):
        print("TRAIN:", train_index.size, "TEST:", test_index)
        X_train = X.iloc[train_index]
        y_train = Y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = Y.iloc[test_index]
    
    return X_train, y_train, X_test, y_test

def randomUndersampling(x_DataSet, y_DataSet, sampling_strategy = 'auto'):
    sm = RandomUnderSampler(random_state=50, sampling_strategy=sampling_strategy)
    x_res, y_res = sm.fit_resample(x_DataSet, y_DataSet)
    print('Resampled dataset shape %s' % Counter(y_res))
    return x_res, y_res

def removeCoordinatesFromDataSet(dataSet):
    '''
    Remove colums of coordinates if they exist in dataset
    @input:
      @dataSet: pandas dataSet
    '''
    DSNoCoord = dataSet
    if 'x_coord' in DSNoCoord.keys(): 
      DSNoCoord.drop(['x_coord','y_coord'], axis=1, inplace = True)
    else: 
      print("DataSet has no coordinates to remove")
    return DSNoCoord

def transformDatasets(cfg):
    '''
    This fucntion takes a list of dataset as *csv paths and apply a transformation in loop. You can custom your transformation to cover your needs. 
    @cfg:DictConfig: The input must be a Hydra configuration lile, containing the key = datasetsList to a csv with the addres of all *.cvs file to process.  
    '''
    DatasetsPath = cfg.datasetsList
    listOfDatsets = createListFromCSV(DatasetsPath)
    for file in listOfDatsets:
        X,Y = importDataSet(file, targetCol='Labels')
        x_dsUndSamp, Y_dsUndSamp = randomUndersampling(X,Y)
        x_dsUndSamp['Labels'] = Y_dsUndSamp.values
        NewFile = addSubstringToName(file,'_balanced')
        x_dsUndSamp.to_csv(NewFile, index=None)
    pass

def DFOperation_removeNegative(DF:pd.DataFrame,colName):
    '''
    NOTE: OPPERATIONS ARE MADE IN PLACE!!!
    Remove all row index in the collumn <colName> if the value is negative
    '''
    DF = DF[DF.colName>=0]
    return DF

    
### Modifying class domain
def pseudoClassCreation(dataset, conditionVariable, threshold, pseudoClass, targetColumnName):
    '''
    Replace <targetClass> by  <pseudoClass> where <conditionVariable >= threshold>. 
    Return:
      dataset with new classes group. 
    '''
    datasetReclassified = dataset.copy()
    actualTarget = (np.array(dataset[targetColumnName])).ravel()
    conditionVar = (np.array(dataset[conditionVariable])).ravel()
    datasetReclassified[targetColumnName] = [ pseudoClass if conditionVar[j] >= threshold 
                                           else actualTarget[j]
                                           for j in range(len(actualTarget))]
    print(Counter(datasetReclassified[targetColumnName]))
    return  datasetReclassified

def revertPseudoClassCreation(dataset, originalClass, pseudoClass, targetColumnName):
    '''
    Restablich  <targetClass> with <originalClass> where <targetColumnName == pseudoClass>. 
    Return:
      dataset with original classes group. 
    '''
    datasetReclassified = dataset.copy()
    actualTarget = (np.array(dataset[targetColumnName])).ravel()
    datasetReclassified[targetColumnName] = [ originalClass if actualTarget[j] == pseudoClass
                                           else actualTarget[j]
                                           for j in range(len(actualTarget))]
    print(Counter(datasetReclassified[targetColumnName]))
    return  datasetReclassified

def makeBinary(dataset,targetColumn,classToKeep:int, replacerClassName:int):
    '''
    makeBinary(dataset,targetColumn,classToKeep, replacerClassName)
    @classToKeep @input: Class name to conserv. All different classes will be repleced by <replacerClassName>
    '''
    repalcer  = dataset[targetColumn].to_numpy()
    dataset[targetColumn] = [replacerClassName if repalcer[j] != classToKeep else repalcer[j] for j in range(len(repalcer))]  
    return dataset

### Pretreatment. 
def extractFloodClassForMLP(csvPath):
    '''
    THis function asume the last colum of the dataset are the lables
    
    The goal is to create separated Datasets with classes 1 and 5 from the input csv. 
    The considered rule is: 
        Class_5: all class 5.
        Class_1: All classes, since they are inclusive. All class 5 are also class 5. 
    '''
    df = pd.read_csv(csvPath,index_col = None)
    print(df.head())
    labelsName = df.columns[-1]
    labels= df.iloc[:,-1]
    uniqueClasses = pd.unique(labels)
    if 1 in uniqueClasses:
        class1_Col = [1 if i!= 0 else 0 for i in labels]
        df[labelsName] = class1_Col
        dfOutputC1 = addSubstringToName(csvPath,'_Class1')
        df.to_csv(dfOutputC1,index=None)

    if 5 in uniqueClasses:
        class5_Col = [1 if i == 5 else 0 for i in labels]
        df[labelsName] = class5_Col
        dfOutputC5 = addSubstringToName(csvPath,'_Class5')
        df.to_csv(dfOutputC5,index=None)
    
    return dfOutputC1,dfOutputC5

### Configurations And file management
def importConfig():
    with open('./config.txt') as f:
        content = f.readlines()
    print(content)    
    return content

def getLocalPath():
    return os.getcwd()

def makePath(str1,str2):
    return os.path.join(str1,str2)

def ensureDirectory(pathToCheck):
    if os.path.isdir(pathToCheck): 
        return pathToCheck
    else:
        os.mkdir(pathToCheck)
        print(f"Confirmed directory at: {pathToCheck} ")
        return pathToCheck

def relocateFile(inputFilePath, outputFilePath):
    '''
    NOTE: @outputFilePath ust contain the complete filename
    Sintax:
     @shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    '''
    shutil.move(inputFilePath, outputFilePath)
    return True

def createTransitFolder(parent_dir_path):
    path = os.path.join(parent_dir_path, 'TransitDir')
    ensureDirectory(path)
    return path

def clearTransitFolderContent(path, filetype = '/*'):
    '''
    NOTE: This well clear dir without removing the parent dir itself. 
    We can replace '*' for an specific condition ei. '.tif' for specific fileType deletion if needed. 
    @Arguments:
    @path: Parent directory path
    @filetype: file type toi delete. @default ='/*' delete all files. 
    '''
    files = glob.glob(path + filetype)
    for f in files:
        os.remove(f)
    return True

def listFreeFilesInDirByExt(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    for (root, dirs, file) in os.walk(cwd):
        file_list = [i for i in file if ext in i]
        return file_list

def listALLFilesInDirByExt(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    FILE_LIST = []
    for (root, dirs, file) in os.walk(cwd):
        for i in file:
            if ext in i:
                FILE_LIST.append(i)
    return FILE_LIST

def createListFromCSVColumn(csv_file_path, col_id:str):  
    '''
    @return: list from <col_id> in <csv_file_location>.
    Argument:
    @csv_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.read_csv(csv_file_path, index_col = None)
    for i in df[col_id]:
        x.append(i)
    return x

def createListFromExelColumn(excell_file_location,Sheet_id:str, col_id:str):  
    '''
    @return: list from <col_id> in <excell_file_location>.
    Argument:
    @excell_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.ExcelFile(excell_file_location).parse(Sheet_id)
    for i in df[col_id]:
        x.append(i)
    return x

def splitFilenameAndExtention(file_path):
    '''
    pathlib.Path Options: 
    '''
    fpath = pathlib.Path(file_path)
    extention = fpath.suffix
    name = fpath.stem
    return name, extention 

def importDataSet(csvPath, targetCol: str, colsToDrop:list=None):
    '''
    Import datasets and return         
    @csvPath: DataSetName => The dataset path as *csv file. 
    @Output: Features(x) and tragets(y) 
    ''' 
    x  = pd.read_csv(csvPath, index_col = None)
    # print(x.columns)
    y = x[targetCol]
    x.drop([targetCol], axis=1, inplace = True)
    if colsToDrop is not None:
        # print(x.columns)
        x.drop(colsToDrop, axis=1, inplace = True)
        # print(x.columns)
    return x, y

def concatDatasets(datsetList_csv, outPath):
    outDataSet = pd.DataFrame()
    for file in datsetList_csv:
        df = pd.read_csv(file,index_col = None)
        outDataSet = pd.concat([outDataSet,df], ignore_index=True)
    outDataSet.to_csv(outPath, index=None)
    pass


### Metrics ####  
def accuracyFromConfusionMatrix(confusion_matrix):
    '''
    Only for binary
    '''
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def pritnAccuracy(y_predic, y_val):
    '''
    Only for binary
    '''
    cm = confusion_matrix(y_predic, y_val) 
    print("Accuracy of MLPClassifier : ", accuracyFromConfusionMatrix(cm)) 

            ###########            
            ### GIS ###
            ###########

def makePredictionToImportAsSHP(model, x_test, y_test, targetColName):
    '''
    We asume here that x_test contain coordinates as <x_coord> and <y_coord>.
    Return:
         The full dataset including a prediction column.  
    '''
    x_testCopy = x_test 
    sampleCoordinates = pd.DataFrame()
    sampleCoordinates['x_coord'] = x_testCopy['x_coord']
    sampleCoordinates['y_coord'] = x_testCopy['y_coord']
    x_testCopy.drop(['x_coord','y_coord'], axis=1, inplace = True)
    y_hay = model.predict(x_testCopy)
    ds_toSHP = x_testCopy
    ds_toSHP[targetColName] = y_test
    ds_toSHP['x_coord'] = sampleCoordinates['x_coord']
    ds_toSHP['y_coord'] = sampleCoordinates['y_coord']
    ds_toSHP['prediction'] = y_hay
    return ds_toSHP



##### Logging

class logg_Manager:
    '''
    This class creates a logger object that writes logs to both a file and the console. 
    @log_name: lLog_name. Logged at the info level by default.
    @log_dict: Dictionary, Set the <attributes> with <values> in the dictionary. 
    The logger can be customized by modifying the logger.setLevel and formatter attributes.

    The update_logs method takes a dictionary as input and updates the attributes of the class to the values in the dictionary. The method also takes an optional level argument that determines the severity level of the log message. 
    '''
    def __init__(self,log_dict:dict=None):# log_name, 
        self.HydraOutputDir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']  
        self.logger = logging.getLogger(HydraConfig.get().job.name)
        logerName = self.logger.name
        # Log some messages
        # logpath = os.path.join(self.HydraOutputDir,logerName)
        # self.logger.setLevel(logging.INFO)  ## Default Level
        # self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # self.file_handler = logging.FileHandler(logpath)
        # self.file_handler.setLevel(logging.DEBUG)
        # self.file_handler.setFormatter(self.formatter)
        # self.stream_handler = logging.StreamHandler()
        # self.stream_handler.setLevel(logging.ERROR)
        # self.stream_handler.setFormatter(self.formatter)
        # self.logger.addHandler(self.file_handler)
        # self.logger.addHandler(self.stream_handler)
        
        ### Fill the logger at creation with a dictionary.
        if log_dict is not None:
            for key, value in log_dict.items():
                setattr(self, key, value)
                self.logger.info(f'{key} - {value}')
        
    def update_logs(self, log_dict, level=None):
        for key, value in log_dict.items():
            setattr(self, key, value)
            if level == logging.DEBUG:
                self.logger.debug(f'{key} - {value}')
            elif level == logging.WARNING:
                self.logger.warning(f'{key} - {value}')
            elif level == logging.ERROR:
                self.logger.error(f'{key} - {value}')
            else:
                self.logger.info(f'{key} - {value}')
    
    # def saveLogsAsTxt(self):
    #     '''
    #     Save the hydra default logger as txt file. 
    #     '''
    #     # Create a file handler
    #     handler = logging.FileHandler(self.HydraOutputDir + self.logger.name + '.txt')
    #     # Set the logging format
    #     formatter = logging.Formatter('%(name)s - %(message)s')
    #     handler.setFormatter(formatter)
    #     # Add the file handler to the logger
    #     self.logger.addHandler(handler)


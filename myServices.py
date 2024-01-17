import os,ntpath
import random
import glob
import pathlib
import shutil
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler 
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from osgeo import gdal,ogr,osr


## Multiprocessing 
import multiprocessing
import concurrent.futures

### Colorlogs. Not necessary, just fun !!!
import coloredlogs
coloredlogs.DEFAULT_FIELD_STYLES = {'asctime': {'color': 28},'levelname': {'bold': True, 'color': 'blue'}, 'name': {'color': 'blue'}, 'programname': {'color': 'cyan'}}
coloredlogs.DEFAULT_LEVEL_STYLES = {'critical': {'bold': True, 'color': 'red'}, 'debug': {'color': 'green'}, 'error': {'color': 'red'}, 'info': {}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'bold': True, 'color': 'green'}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}
coloredlogs.COLOREDLOGS_LEVEL_STYLES='spam=22;debug=28;verbose=34;notice=220;warning=202;success=118,bold;error=124;critical=background=red'


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
def importDataSet(csvPath, targetCol: str, colsToDrop:list=None)->pd.DataFrame:
    '''
    Import datasets and return         
    @csvPath: DataSetName => The dataset path as *csv file. 
    @Output: Features(x) and tragets(y) 
    ''' 
    x  = pd.read_csv(csvPath, index_col = None)
    y = x[targetCol]
    x.drop([targetCol], axis=1, inplace = True)
    if colsToDrop is not None:
        x.drop(colsToDrop, axis=1, inplace = True)
        print(f'Features for training {x.columns}')
    return x, y

def concatDatasets(datsetList_csv, outPath):
    outDataSet = pd.DataFrame()
    for file in datsetList_csv:
        df = pd.read_csv(file,index_col = None)
        outDataSet = pd.concat([outDataSet,df], ignore_index=True)
    outDataSet.to_csv(outPath, index=None)
    pass

def stratifiedSplit(dataSetName, targetColName):
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

### Pretreatment
def standardizeDataSetCol(dataSetPath, colName):
    '''
    Perform satandardizartion on a column of a DataFrame
    '''
    dataSet = pd.read_csv(dataSetPath, index_col=None)
    mean = dataSet[colName].mean()
    std = dataSet[colName].std()
    column_estandar = (dataSet[colName] - mean) / std
    return column_estandar,mean,std
 
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

def listFreeFilesInDirByExt(cwd:str, ext = '.tif'):
    '''
    @ext = *.tif by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    cwd = os.path.abspath(cwd)
    # print(f"Current working directory: {cwd}")
    file_list = []
    for (root, dirs, file) in os.walk(cwd):
        for f in file:
            # print(f"File: {f}")
            _,_,extent = get_parenPath_name_ext(f)
            if extent == ext:
                file_list.append(f)
    return file_list

def listFreeFilesInDirByExt_fullPath(cwd:str, ext = '.csv') -> list:
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    cwd = os.path.abspath(cwd)
    # print(f"Current working directory: {cwd}")
    file_list = []
    for (root,_, file) in os.walk(cwd, followlinks=True):
        for f in file:
            # print(f"Current f: {f}")
            _,extent = splitFilenameAndExtention(f)
            # print(f"Current extent: {extent}")
            if ext == extent:
                file_list.append(os.path.join(root,f))
    return file_list

def listFreeFilesInDirBySubstring_fullPath(cwd:str, substring = '') -> list:
    '''
    @substring: substring to be verify onto the file name. 
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    cwd = os.path.abspath(cwd)
    # print(f"Current working directory: {cwd}")
    file_list = []
    for (root,_, file) in os.walk(cwd, followlinks=True):
        for f in file:
            if substring.lower() in f.lower():
                file_list.append(os.path.join(root,f))
    return file_list

def listALLFilesInDirByExt(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList: list = []
    for (root, _, _) in os.walk(cwd):
         fullList.extend(listFreeFilesInDirByExt(root, ext)) 
    return fullList

def listALLFilesInDirByExt_fullPath(cwd, ext = '.csv'):
    '''
    @ext: NOTE <ext> must contain the "." ex: '.csv'; '.tif'; etc...
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList = []
    for (root, _, _) in os.walk(cwd):
        # print(f"Roots {root}")
        localList = listFreeFilesInDirByExt_fullPath(root, ext)
        # print(f"Local List len :-->> {len(localList)}")
        fullList.extend(localList) 
        return fullList

def listALLFilesInDirBySubstring_fullPath(cwd, substring = '.csv')->list:
    '''
    @substring: substring to be verify onto the file name.    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList = []
    for (root, _, _) in os.walk(cwd):
        # print(f"Roots {root}")
        localList = listFreeFilesInDirBySubstring_fullPath(root, substring)
        print(f"Local List len :-->> {len(localList)}")
        fullList.extend(localList) 
        return fullList

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

def replaceExtention(inPath,newExt: str)->os.path :
    '''
    Just remember to add the poin to the new ext -> '.map'
    '''
    dir,fileName = ntpath.split(inPath)
    _,actualExt = ntpath.splitext(fileName)
    return os.path.join(dir,ntpath.basename(inPath).replace(actualExt,newExt))

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

def buildShapefilePointFromCsvDataframe(csvDataframe:os.path, outShepfile:os.path='', EPGS:int=3979):
    '''
    Creates a shapefile of points from a Dataframe <df>. The DataFrame is expected to have a HEADER, and the two first colums with the x_coordinates and y_coordinates respactivelly.
    @csvDatafame:os.path: Path to the *csv containing the Dataframe with the list of points to add to the Shapefile. 
    @outShepfile:os.path: Output path to the shapefile (Optional). If Non, the shapefile will have same path and name that csvDataframe.
    @EPGS: EPGS value of a valid reference system (Optional).(Default = 4326).
    '''
    df = pd.read_csv(csvDataframe)
    #### Create a new shapefile
    ## Set shapefile path.
    if outShepfile:
        outShp = outShepfile
    else: 
        outShp = replaceExtention(csvDataframe,'.shp')
        print(outShp)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(outShp)

    # Set the spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPGS)  # WGS84

    # Create a new layer
    layer = ds.CreateLayer("", srs, ogr.wkbPoint)

    # Add fields
    for column in df.columns:
        field = ogr.FieldDefn(column, ogr.OFTReal)
        field.SetWidth(10)  # Total field width
        field.SetPrecision(2)  # Width of decimal part
        layer.CreateField(field)

    # Add points
    for idx,row in df.iterrows():
        # Create a new feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes
        for column in df.columns:
            feature.SetField(column, row[column])
        # Create a new point geometry
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(row[0], row[1])
        # Set the feature geometry
        feature.SetGeometry(point)
        # Create the feature in the layer
        layer.CreateFeature(feature)
        # Dereference the feature
        feature = None

    # Dereference the data source
    ds = None

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
        # coloredlogs.install(logger=self.logger)
        # coloredlogs.install(fmt='%(asctime)s,%(levelname)s %(message)s')
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

#####  Parallelizer
def parallelizerWithProcess(function, args:list, executors:int = 4):
    '''
    Parallelize the <function> in the input to the specified number of <executors>.
    @function: python function
    @args: list: list of argument to pas to the function in parallel. 
    '''
    with concurrent.futures.ProcessPoolExecutor(executors) as executor:
        # start_time = time.perf_counter()
        result = list(executor.map(function,args))
        # finish_time = time.perf_counter()
    # print(f"Program finished in {finish_time-start_time} seconds")
    print(result)

def parallelizerWithThread(function, args:list, executors:int = None):
    '''
    Parallelize the <function> in the input to the specified number of <executors>.
    @function: python function
    @args: list: list of argument to pas to the function in parallel. 
    '''
    with concurrent.futures.ThreadPoolExecutor(executors) as executor:
            start_time = time.perf_counter()
            result = list(executor.map(function, args))
            finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)

def maxParalelizer(function,args):
    '''
    Same as paralelizer, but optimize the pool to the capacity of the current processor.
    NOTE: To be tested
    '''
    # print(args)
    pool = multiprocessing.Pool()
    result = pool.map(function,args)
    print(result)

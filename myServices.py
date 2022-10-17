import os
import joblib
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler 

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

def makeNameByTime():
    name = time.strftime("%y%m%d%H%M")
    return name

## modeling manipulation
def saveModel(estimator, id):
    name = id + ".pkl" 
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

def pseudoClassCreation(dataset, conditionVariable, threshold, pseudoClass, targetClassName):
    '''
    Replace <targetClass> by  <pseudoClass> where <conditionVariable >= threshold>. 
    Return:
      dataset with new classes group. 
    '''
    datsetReclassified = dataset.copy()
    actualTarget = (np.array(dataset[targetClassName])).ravel()
    conditionVar = (np.array(dataset[conditionVariable])).ravel()
    datsetReclassified[targetClassName] = [ pseudoClass if conditionVar[j] >= threshold 
                                           else actualTarget[j]
                                           for j in range(len(actualTarget))]
    print(Counter(datsetReclassified[targetClassName]))
    return  datsetReclassified

def revertPseudoClassCreation(dataset, originalClass, pseudoClass, targetClassName):
    '''
    Restablich  <targetClass> with <originalClass> where <targetClassName == pseudoClass>. 
    Return:
      dataset with original classes group. 
    '''
    datsetReclassified = dataset.copy()
    actualTarget = (np.array(dataset[targetClassName])).ravel()
    datsetReclassified[targetClassName] = [ originalClass if actualTarget[j] == pseudoClass
                                           else actualTarget[j]
                                           for j in range(len(actualTarget))]
    print(Counter(datsetReclassified[targetClassName]))
    return  datsetReclassified

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
        return
    else:
        os.mkdir(pathToCheck)
        print(f"Created directory at path: {pathToCheck} ")
        return True

def importDataSet(dataSetName, targetCol: str):
    '''
    Import datasets and filling NaN values          
    @input: DataSetName => The dataset path. 
    @Output: Features(x) and tragets(y) 
    ''' 
    x  = pd.read_csv(dataSetName, index_col = None)
    y = x[targetCol]
    x.drop([targetCol], axis=1, inplace = True)
    return x, y

   
def accuracyFromConfisionMatrix(confusion_matrix):
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
    print("Accuracy of MLPClassifier : ", accuracyFromConfisionMatrix(cm)) 





class searchHiddenLayerSizeMLP():
    def __init__(self, model, params, trainSetPath, testSetPath) -> None:
        self.params = params
        self.model = model
        self.trainSetPath = trainSetPath 
        self.testSetPath = testSetPath
        pass

    def buildInterval(n, center):
        '''
        n = number of times the function has been called. 
        center = center of interval
        '''
        if n>1:
                start = center-50
                end = center+60
                stepSize = 10  
        else:
            start = center-10
            end = center+11
            stepSize = 2
            if start < 0 : start = 0
        return np.arange(start,end,stepSize)  
    
    def iters(X,interval,function:function):
        print("X = ", X)
        if X>1:
            for i in interval:
                function()
            X-=1
            newCenterIdx = np.random.choice(len(interval), size=1, replace=False)
            center = interval[newCenterIdx]
            interval = searchHiddenLayerSizeMLP.buildInterval(X,center)
            searchHiddenLayerSizeMLP.iters(X,interval)
        else:
            for i in interval:
                function()
        print('Final loop')
        return 




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



def importListFromExelCol(excell_file_location,Shet_id, col_id):  
    '''
    @return: list from <col_id> in <excell_file_location>.
    Argument:
    @excell_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Conside index 0 for the first column)
    '''       
    x=[]
    df = pd.ExcelFile(excell_file_location).parse(Shet_id)
    for i in df[col_id]:
        x.append(i)
    return x

def clipRasterWithPoligon(rastPath, polygonPath,outputPath):
    '''
    Clip a raster (*.GTiff) with a single polygon feature 
    '''
    os.system("gdalwarp -datnodata -9999 -q -cutline" + polygonPath + " crop_to_cutline" + " -of GTiff" + rastPath + " " + outputPath)
   
   
def separateClippingPolygonss(inPath,field, outPath = "None"):
    '''
    Crete individial *.shp for each Clip in individual directories. 
    @input: 
       @field: Flield in the input *.shp to chose.
       @inPath: The path to the original *.shp.
    '''
    if outPath != "None":
        ensureDirectory(outPath)
        os.mkdir(os.path.join(outPath,"/clipingPolygons"))
        saveingPath = os.path.join(outPath,"/clipingPolygons") 
    else: 
        ensureDirectory(os.path.join(getLocalPath(),"/clipingPolygons"))
        saveingPath = os.path.join(outPath,"/clipingPolygons")

    driverSHP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSHP.Open(inPath)
    if ds in None:
        print("Layer not found")
        return False
    lyr = ds.GetLayer()
    spatialRef = lyr.GetSpatialRef()
    for feautre in lyr:
        fieldValue = feautre.GetField(field)
        os.mkdir(os.path.join(saveingPath,str(fieldValue)))
        outName = str(fieldValue)+"Clip.shp"
        outds = driverSHP.CreateDataSorce("clipingPolygons/" + str(fieldValue) + "/" + outName)
        outLayer = outds.CreateLayer(outName, srs=spatialRef,geo_type = ogr.wkbPolygon)
        outDfn = outLayer.GetLayerDef()
        inGeom = feautre.GetGeometryRef()
        outFeature = ogr.Feature(outDfn)
        outFeature.SetGeometry(inGeom)
        outLayer.CreateFeature(outFeature)
    
    return True

def clipRaster(rasterPath,polygonPath,field, outputPath):
    ''' 
    '''
    driverSHP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSHP.Open(polygonPath)
    if ds in None:
        print("Layer not found")
        return False
    lyr = ds.GetLayer()
    spatialRef = lyr.GetSpatialRef()
    for feautre in lyr:
        fieldValue = feautre.GetField(field)
        clipRasterWithPoligon(rasterPath, polygonPath, outputPath)
    return True
    

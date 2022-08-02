import os
from re import A
import pandas as pd
import numpy as np
from datetime import datetime

### General applications ##

class timeit(): 
    '''
    to compute ececution time do:
    with timeit():
         your code, e.g., 
    '''
    def __enter__(self):
        self.tic = datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(datetime.now() - self.tic))

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

def logTransformation(x):
    '''
    Logarithmic transformation to redistribute values between 0 and 1. 
    '''
    x_nonZeros = np.where(x <= 0.0000001, 0.0001, x)
    return np.max(np.log(x_nonZeros)**2) - np.log(x_nonZeros)**2
            
### GIS ###
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
    



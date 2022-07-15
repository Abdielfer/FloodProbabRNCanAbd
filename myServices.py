from http.client import UnknownTransferEncoding
import os
import pandas as pd
from osgeo import gdal, ogr

def getLocalPath():
    return os.getcwd()

def ensureDirectory(pathToCheck):
    if os.path.isdir(pathToCheck): 
        return True
    elif os.path.exists(pathToCheck):
        os.mkdir(pathToCheck)
        print(f"Created directory at path: {pathToCheck} ")
        return True
    print("Invalid path")
    return False       

def importListFromExelCol(excell_file_location,Shet_id, col_id):  
    '''
    @return: list from <col_id> in <excell_file_location>.
    Argument:
    @excell_file_location: file name if is in the project directory, full file laction otherwise.
    @col_id : number of the desired collumn to extrac info from (Conside index 0 for the first column)
    '''       
    x=[]
    df = pd.ExcelFile(excell_file_location).parse(Shet_id)
    for i in df[col_id]:
        x.append(i)
    return x

def clipRasterFromPoligons(rastPath, polygonPath,outputPath):
    
    return
   


def separateClippingPolygonss(inPath,field):
    '''
    Crete individial *.shp for each Clip in separeted directories. 
    @input: 
       @field: Flield in the input *.shp to chose to build tha Clip. 
       @inPath: The path to the original *.shp.
    '''
    driverSHP = ogr.GetDriverByName("ESRI Shapefile")
    ds = driverSHP.Open(inPath)
    if ds in None:
        print("Layer not found")
    lyr = ds.GetLayer()
    spatialRef = lyr.GetSpatialRef()
    for feautre in lyr:
        fieldValue = feautre.GetField(field)
        os.mkdir("clipingPolygons/" + str(fieldValue))
        outName = str(fieldValue)+"Clip.shp"
        outds = driverSHP.CreateDataSorce("clipingPolygons/" + str(fieldValue) + "/" + outName)
        outLayer = outds.CreateLayer(outName, srs=spatialRef,geo_type = ogr.wkbPolygon)
        outDfn = outLayer.GetLayerDef()
        inGeom = feautre.GetGeometryRef()
        outFeature = ogr.Feature(outDfn)
        outFeature.SetGeometry(inGeom)
        outLayer.CreateFeature(outFeature)



    return


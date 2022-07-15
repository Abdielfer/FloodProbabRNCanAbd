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
    


import geopandas as gpd
from shapely.geometry import shape, Polygon
import utm
from pyproj import CRS

class importSHPFIles():
    def __init__(self, pathSHP:str) -> None:
        self.SHPObject = readSHP(pathSHP)
        pass
    
    def print_headAttributeTable(self):
        print(self.SHPObject.head(5))

    def getCRS(self):
        return self.SHPObject.crs

    def getGeometryType(self):
        return self.SHPObject.geom_type
    
    def getGPDObject(self):
        return self.SHPObject





#### General use functions    ####
def readSHP(pathSHP:str):
    shpFile = gpd.read_file(pathSHP)
    return shpFile


def findtheutm(aGeometry):
    '''
    A function to find a coordinates UTM zone
    '''
    
    x, y, parallell, latband = utm.from_latlon(aGeometry.centroid.y, aGeometry.centroid.x)
    if latband in 'CDEFGHJKLM': #https://www.lantmateriet.se/contentassets/379fe00e09d74fa68550f4154350b047/utm-zoner.gif
        ns = 'S'
    else:
        ns = 'N'
    crs = "+proj=utm +zone={0} +{1}".format(parallell, ns) #https://gis.stackexchange.com/questions/365584/convert-utm-zone-into-epsg-code
    crs = CRS.from_string(crs)
    _, code = crs.to_authority()
    return int(code)
    
def computeGeometrArea(gpdFrame: gpd.geodataframe.GeoDataFrame):
    '''
        You must extrac de polygon from the GeoDataFrame befor passing to the function.
    '''
    epsg = findtheutm(gpdFrame.geometry.iloc[0])
    gpdFrame['Area_m2'] = gpdFrame.to_crs(epsg).area
    return gpdFrame['Area_m2']

def addAreaColum(gpdFrame: gpd.geodataframe.GeoDataFrame, factor = 1) -> None:
    '''
    Add a new column to the GeoDataFrame with the area of features
     The defoult result is in CSR units. YOu can use <factor> to maninipulate the area units.
    '''
    for i in range(len(gpdFrame)):

        pass 

import geopandas as gpd
from shapely.geometry import shape, Polygon

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

def computePolygonArea(polygon):
    '''
        You must extrac de polygon from the GeoDataFrame befor passing to the function.
    '''
    return shape(polygon).area

def addAreaColum(gpdFrame: gpd.geodataframe.GeoDataFrame, factor = 1) -> None:
    '''
    Add a new column to the GeoDataFrame with the area of features
     The defoult result is in CSR units. YOu can use <factor> to maninipulate the area units.
    '''
    for i in range(len(gpdFrame)):

    

    
        pass 

import geopandas as gpd
from shapely.geometry import shape


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


#### General use functions    ####
def readSHP(pathSHP:str):
    shpFile = gpd.read_file(pathSHP)
    return shpFile

def computePolygonArea(polygon):
    return shape(polygon).area

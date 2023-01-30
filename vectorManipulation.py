import geopandas as gpd


class importSHPFIles():
    def __init__(self, pathSHP:str) -> None:
        self.SHPObject = readSHP(pathSHP)
        self.headAttributeTable = self.SHPObject.head(5)
        pass





#### General use functions    ####
def readSHP(pathSHP:str):
    shpFile = gpd.read_file(pathSHP)
    return shpFile


import os
import myServices as ms
import whiteboxApplications as wba

from whitebox.whitebox_tools import WhiteboxTools
from torchgeo.datasets.utils import download_url


## LocalPaths and global variables: to be adapted to your needs ##
#to get the current working directory
currentDirectory = os.getcwd()
wbt = WhiteboxTools()
wbt.set_working_dir(currentDirectory)
wbt.set_verbose_mode(True)
wbt.set_compress_rasters(True) # compress the rasters map. Just ones in the code is needed




def main():
    # list = ms.importListFromExelCol('/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodProbabRNCanAbd/saint_john_NFL_DTM.xlsx','Feuil1','ftp_dtm')
    # importer = dtmTailImporter(list, '/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodMaps/testZone')
    # importer.impotTailToLocalDir()
    wba.setWBTWorkingDir('/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodMaps/testZone')
    transformer = wba.dtmTransformer('/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodMaps/testZone')
    transformer.computeMosaic()
    
if __name__ == "__main__":
    main()
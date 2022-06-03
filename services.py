'''
All classes needed to perfom:
- DTM extraction from remote(url needed)
- Generate DTM product for hydrological pruposes. 
- Resampling tools 
'''

import os
import tempfile
import pandas as pd
from sqlalchemy import false, null
from torch.utils.data import DataLoader
import whitebox as WTB
from whitebox.whitebox_tools import WhiteboxTools, default_callback

from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler

## LocalPaths and global variables: to be adapted to your needs ##
dtmLocalPath = 'C:/Users/abfernan/floodprobabilityProject/dtm'
raster_outputPath ='C:/Users/abfernan/floodprobabilityProject/outputDtmProduct'
wbt = WhiteboxTools()
wbt.set_working_dir(dtmLocalPath)
wbt.set_verbose_mode(True)
wbt.set_compress_rasters(True) # compress the rasters map. Just ones in the code is needed

## importing section 
class dtmTailImporter():
    '''
    This is a class to import DTM's from the URL.
    Arguments at creation:
     @tail_URL_NamesList : list of url for the tail to import
    '''
    def __init__(self, tail_URL_NamesList):
        self.tail_URL_NamesList = tail_URL_NamesList
        self.dtmLocalPath = dtmLocalPath

    def impotTailToLocalDir(self):
        '''
        import the tails in the url list to the local directory defined in <dtmLocalPath> 
        '''
        for path in self.tail_URL_NamesList:
            download_url(path, dtmLocalPath)

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

## Pretraitment #

class transformingDTM():
    def __init__(self):
        return None

    def computeSlope(self,inDTMName):
        outSlope = 'slope_'+ inDTMName
        wbt.slope(inDTMName, 
                outSlope, 
                zfactor=None, 
                units="degrees", 
                callback=default_callback)
    
    def computeAspect(self,inDTMName):
        outAspect = 'aspect_'+ inDTMName
        wbt.aspect(inDTMName, 
                outAspect, 
                zfactor=None, 
                callback=default_callback)

    def computeMosaic(self, outpouFileName):
        ''' 
        @return: A mosaic opperation's result, creates a single tile containing all imputs tails.
        Argument
        @outpouFileName: The output file name. IMPORTANT: include the extention (e.i. .tif ) 
        '''
        print(raster_outputPath)
        if os.path.isdir(raster_outputPath) != True: # Creates output dir if it does not already exist 
            os.mkdir('C:/Users/abfernan/Desktop/raste_output')      

        outfile = os.path.join(raster_outputPath,outpouFileName)
        
        if wbt.mosaic(
            output=outfile, 
            method = "nn"  # Calls mosaic tool with nearest neighbour as the resampling method ("nn")
            ) != 0:
            print('ERROR running mosaic')  # Non-zero returns indicate an error.
    



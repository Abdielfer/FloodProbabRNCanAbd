'''
All classes needed to perfom:
- DTM extraction from remote(url needed)
- Generate DTM product for hydrological pruposes. 
- Resampling tools 
'''

from ast import Return
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
#to get the current working directory
currentDirectory = os.getcwd()
wbt = WhiteboxTools()
wbt.set_working_dir(currentDirectory)
wbt.set_verbose_mode(True)
wbt.set_compress_rasters(True) # compress the rasters map. Just ones in the code is needed

## importing section 
class dtmTailImporter():
    '''
    This is a class to import DTM's from the URL.
    Arguments at creation:
     @tail_URL_NamesList : list of url for the tail to import
    '''
    def __init__(self, tail_URL_NamesList, localPath = "None"):
        self.tail_URL_NamesList = tail_URL_NamesList
        self.localPath = localPath

    def impotTailToLocalDir(self):
        '''
        import the tails in the url <tail_URL_NamesList> to the local directory defined in <localPath> 
        '''
        if os.path.isfile(self.localPath): 
            for path in self.tail_URL_NamesList:
                download_url(path, self.localPath)
            print(f"Tails dawnloaded to: {self.localPath}")  
        else:
            outputPath = input('Enter a destiny path to download:')
            print(outputPath)
            if ensureDirectory(outputPath):
                for path in self.tail_URL_NamesList:
                    download_url(path, outputPath)
                print(f"Tails dawnloaded to: {outputPath}")      
            else:
                for path in self.tail_URL_NamesList:
                    download_url(path, currentDirectory)
                print(f"Tails dawnloaded to: {currentDirectory}") 
               
        

## Pretraitment #

class dtmTransformer():
    '''
     This class contain some functions to generate geomorphological and hydrological features from DTM.
    Functions are mostly based on Whitebox libraries. For optimal functionality DTM’s most be high resolution, 
    ideally Lidar 1 m or < 2m. 
    '''
    def __init__(self, workingDir = "None"):
        if os.path.isdir(workingDir): # Creates output dir if it does not already exist 
            self.workingDir = workingDir
            wbt.set_working_dir(workingDir)
        else:
            self.workingDir = input('Enter working directory')
            if ensureDirectory(self.workingDir):
                wbt.set_working_dir(self.workingDir)
        
    def computeMosaic(self, outpouFileName = "None"):
        ''' 
        @return: Return True if mosaic succeed, False otherwise. Result is saved to wbt.work_dir. 
        Argument
        @outpouFileName: The output file name. IMPORTANT: include the extention (e.i. .tif ) 
        '''
        if ".tif" not in outpouFileName:
            outpouFileName = input("enter a valid file name with the '.tif' extention")
        outFilePathAndName = os.path.join(wbt.work_dir,outpouFileName)
        if wbt.mosaic(
            output=outFilePathAndName, 
            method = "nn"  # Calls mosaic tool with nearest neighbour as the resampling method ("nn")
            ) != 0:
            print('ERROR running mosaic')  # Non-zero returns indicate an error.
            return False
        return True
    

    #####  TODO ####
    def fixNoDataAndfillDTM(self, dtmName, keepNoDataRasters = False):
        '''
        Ref: https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#filldepressions
        To ensure the quality of this process, this method 
       
        @argument: 
        @return a filled dtm. 
            optionally with <keepNoDataRasters = False>: a dtm with corected no value data after filling
        ''' 
        output = "fille_" + dtmName
        
        dtmNoDataValueSetted = "noDataOK_"+dtmName
        wbt.set_nodata_value(
            dtmName, 
            dtmNoDataValueSetted, 
            back_value=0.0, 
            callback=default_callback
            )
        dtmMissingDataFilled = "correctedNoData_"+dtmName
        wbt.fill_missing_data(
                dtmNoDataValueSetted, 
                dtmMissingDataFilled, 
                filter=11, 
                weight=2.0, 
                no_edges=True, 
                callback=default_callback
            )

        wbt.fill_depressions(
            dtmMissingDataFilled, 
            output, 
            fix_flats=True, 
            flat_increment=None, 
            max_depth=None, 
            callback=default_callback
            )
     # Remove intermediate results
        if keepNoDataRasters:
            try:
                os.rmdir(dtmNoDataValueSetted)
                os.rmdir(dtmMissingDataFilled)
            except OSError as error:
                print("There was an error removing intermediate results.")

        return True


    def rd8FlowPointerCalculation(self, filledDTMName, pointer = 1):
        '''
        @argument: the method to run pointer:
            1 - Dinf 
            2 - D8 
            3 - Rho8
        '''
          # case to chose method  3 default. 

           # implement wbt pointer
        filledDTMName ==0
        return False

    def DInfFlowCalculation(self, pointer):
        ''' 
        Compute D infinity frlo accumulation algorithm 

        '''
        return False




    ### Ready  ####

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

    



###  Helper functions  ###
def setWBTWorkingDir(workingDir):
    wbt.set_working_dir(workingDir)

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



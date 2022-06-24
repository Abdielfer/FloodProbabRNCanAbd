'''
All classes needed to perfom:
- DTM extraction from remote(url needed)
- Generate DTM product for hydrological pruposes. 
- Resampling tools 
'''

import os
import pandas as pd
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
        self.mainFileName = " "
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
        self.mainFileName = outpouFileName
        outFilePathAndName = os.path.join(wbt.work_dir,outpouFileName)
        if wbt.mosaic(
            output=outFilePathAndName, 
            method = "nn"  # Calls mosaic tool with nearest neighbour as the resampling method ("nn")
            ) != 0:
            print('ERROR running mosaic')  # Non-zero returns indicate an error.
            return False
        return True
    
    def fixNoDataAndfillDTM(self, inDTMName, eraseIntermediateRasters = True):
        '''
        Ref:   https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#filldepressions
        To ensure the quality of this process, this method execute several steep in sequence, following the Whitebox’s authors recommendation (For mor info see the above reference).
        Steps:
        1-	Correct no data values to be accepted for all operation. 
        2-	Fill gaps of no data.
        3-	Fill depressions.
        4-	Remove intermediary results to save storage space (Optionally you can keep it. See @Arguments).  
        @Argument: 
        -inDTMName: Input DTM name
        -eraseIntermediateRasters(default = False): Erase intermediate results to save storage space. 
        @Return: True if all process happened successfully, EROR messages otherwise. 
        @OUTPUT: DTM <filled_ inDTMName> Corrected DTM without depressions. 
        '''
        dtmNoDataValueSetted = "noDataOK_"+inDTMName
        wbt.set_nodata_value(
            inDTMName, 
            dtmNoDataValueSetted, 
            back_value=0.0, 
            callback=default_callback
            )
        dtmMissingDataFilled = "correctedNoData_"+inDTMName
        wbt.fill_missing_data(
            dtmNoDataValueSetted, 
            dtmMissingDataFilled, 
            filter=11, 
            weight=2.0, 
            no_edges=True, 
            callback=default_callback
            )
        output = "filled_" + inDTMName
        wbt.fill_depressions(
            dtmMissingDataFilled, 
            output, 
            fix_flats=True, 
            flat_increment=None, 
            max_depth=None, 
            callback=default_callback
            )  
        if eraseIntermediateRasters:
            try:
                os.remove(os.path.join(wbt.work_dir,dtmNoDataValueSetted))
                os.remove(os.path.join(wbt.work_dir,dtmMissingDataFilled))
            except OSError as error:
                print("There was an error removing intermediate results.")
        return True

    def d8FPointerRasterCalculation(self, inFilledDTMName):
        '''
        @argument:
         @inFilledDTMName: DTM without spurious point ar depression.  
        @UOTPUT: D8_pioter: Raster tu use as input for flow direction and flow accumulation calculations. 
        '''
        output = "d8Pointer_" + inFilledDTMName
        wbt.d8_pointer(
            inFilledDTMName, 
            output, 
            esri_pntr=False, 
            callback=default_callback
            )
    
    def d8_flow_accumulation(self, inFilledDTMName):
        d8FAccOutputName = "d8fllowAcc"+inFilledDTMName
        wbt.d8_flow_accumulation(
            inFilledDTMName, 
            d8FAccOutputName, 
            out_type="cells", 
            log=False, 
            clip=False, 
            pntr=False, 
            esri_pntr=False, 
            callback=default_callback
            )
        print("d8 Done")

    def jensePourPoint(self,inOutlest,d8FAccOutputName):
        jensenOutput = "correctedSnapPoints.shp"
        wbt.jenson_snap_pour_points(
            inOutlest, 
            d8FAccOutputName, 
            jensenOutput, 
            snap_dist = 15.0, 
            callback=default_callback
            )
        print("jensePourPoint Done")

    def watershedConputing(self,d8Pointer, jensenOutput):  
        output = "watersheds_" + d8Pointer
        wbt.watershed(
            d8Pointer, 
            jensenOutput, 
            output, 
            esri_pntr=False, 
            callback=default_callback
        )
        print("watershedConputing Done")




    def DInfFlowCalculation(self, inD8Pointer, log = False):
        ''' 
        Compute DInfinity flow accumulation algorithm.
        Ref: https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#dinfflowaccumulation  
        We keep the DEFAULT SETTING  from source, which compute "Specific Contributing Area". 
        See ref for the description of more output’s options. 
        @Argument: 
            @inD8Pointer: D8-Pointer raster
            @log (Boolean): Apply Log-transformation on the output raster
        @Output: 
            DInfFlowAcculation map. 
        '''
        output = "dInf_" + inD8Pointer
        wbt.d_inf_flow_accumulation(
            inD8Pointer, 
            output, 
            out_type="Specific Contributing Area", 
            threshold=None, 
            log=log, 
            clip=False, 
            pntr=True, 
            callback=default_callback
            )

    ### Ready  ####
    def computeSlope(self,inDTMName):
        outSlope = 'slope_'+ inDTMName
        wbt.slope(inDTMName,
                outSlope, 
                zfactor=None, 
                units="degrees", 
                callback=default_callback
                )
    
    def computeAspect(self,inDTMName):
        outAspect = 'aspect_'+ inDTMName
        wbt.aspect(inDTMName, 
                outAspect, 
                zfactor=None, 
                callback=default_callback
                )

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



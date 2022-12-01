import os
import myServices as ms
from whitebox.whitebox_tools import WhiteboxTools, default_callback
from torchgeo.datasets.utils import download_url

## LocalPaths and global variables: to be adapted to your needs ##
currentDirectory = os.getcwd()
wbt = WhiteboxTools()
wbt.set_working_dir(currentDirectory)
wbt.set_verbose_mode(True)
wbt.set_compress_rasters(True) # compress the rasters map. Just ones in the code is needed
       
## Pretraitment #
class dtmTransformer():
    '''
     This class contain some functions to generate geomorphological and hydrological features from DTM.
    Functions are mostly based on Whitebox libraries. For optimal functionality DTM’s most be high resolution, 
    ideally Lidar 1 m or < 2m. 
    '''
    def __init__(self, workingDir):
        self.mainFileName = " "
        if os.path.isdir(workingDir): # Creates output dir if it does not already exist 
            self.workingDir = workingDir
            wbt.set_working_dir(workingDir)
        else:
            self.workingDir = input('Enter working directory')
            if ms.ensureDirectory(self.workingDir):
                wbt.set_working_dir(self.workingDir)
        
    
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
            
    def dInfFlowAcc(self, inFilledDTMName, id,  userLog: bool):
        output = id
        wbt.d_inf_flow_accumulation(
            inFilledDTMName, 
            output, 
            out_type="ca", 
            threshold=None, 
            log=userLog, 
            clip=False, 
            pntr=False, 
            callback=default_callback
        )
  
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

class rasterTools():
    def __init__(self, workingDir):
        self.mainFileName = " "
        if os.path.isdir(workingDir): # Creates output dir, if it does not already exist. 
            self.workingDir = workingDir
            wbt.set_working_dir(workingDir)
        else:
            self.workingDir = input('Enter working directory')
            if ms.ensureDirectory(self.workingDir):
                wbt.set_working_dir(self.workingDir)

    def computeMosaic(self, outpouFileName = "None"):
        ''' 
        @return: Return True if mosaic succeed, False otherwise. Result is saved to wbt.work_dir. 
        Argument
        @verifiedOutpouFileName: The output file name. IMPORTANT: include the "*.tif" extention.
        '''
        verifiedOutpouFileName = sheckTifExtention(outpouFileName)
        outFilePathAndName = os.path.join(wbt.work_dir,verifiedOutpouFileName)
        if wbt.mosaic(
            output=outFilePathAndName, 
            method = "nn"  # Calls mosaic tool with nearest neighbour as the resampling method ("nn")
            ) != 0:
            print('ERROR running mosaic')  # Non-zero returns indicate an error.
            return False
        return True

    def rasterResampler(sefl, inputRaster, resampledRaster, outputCellSize, resampleMethod = 'cc'):
        '''
        wbt.Resampler ref: https://www.whiteboxgeo.com/manual/wbt_book/available_tools/image_processing_tools.html#Resample
        NOTE: It performes Mosaic if several inputs are provided, in addition to resampling. See refference for details. 
        @arguments: inputRaster, resampledRaster, outputCellSize:int, resampleMethod:str
        Resampling method; options include 'nn' (nearest neighbour), 'bilinear', and 'cc' (cubic convolution)
        '''
        verifiedOutpouFileName = sheckTifExtention(resampledRaster)
        outFilePathAndName = os.path.join(wbt.work_dir,verifiedOutpouFileName)
        wbt.resample(
            inputRaster, 
            outFilePathAndName, 
            cell_size=outputCellSize, 
            base=None, 
            method= resampleMethod, 
            callback=default_callback
            )
    
    def rasterToVectorLine(sefl, inputRaster, outputVector):
        wbt.raster_to_vector_lines(
            inputRaster, 
            outputVector, 
            callback=default_callback
            )

    def rasterVisibility_index(sefl, inputDTM, outputVisIdx):
        '''
        Both, input and output are raster. 
        '''
        wbt.visibility_index(
                inputDTM, 
                outputVisIdx, 
                height=2.0, 
                res_factor=2, 
                callback=default_callback
                )           

    def gaussianFilter(sefl, input, output, sigma = 0.75):
        '''
        input@: kernelSize = integer or tupel(x,y). If integer, kernel is square, othewise, is a (with=x,hight=y) rectagle. 
        '''
        wbt.gaussian_filter(
        input, 
        output, 
        sigma = sigma, 
        callback=default_callback
        )

## importing section 
class dtmTailImporter():
    '''
    This is a class to import DTM's from the URL.
    Arguments at creation:
     @tail_URL_NamesList : list of url for the tail to import
    '''
    def __init__(self, tail_URL_NamesList, localPath):
        self.tail_URL_NamesList = tail_URL_NamesList
        self.localPath = localPath

    def downloadTailsToLocalDir(self):
        '''
        import the tails in the url <tail_URL_NamesList> to the local directory defined in <localPath> 
        '''
        if os.path.isfile(self.localPath): 
            for path in self.tail_URL_NamesList:
                download_url(path, self.localPath)
            print(f"Tails dawnloaded to: {self.localPath}")  
        else:
            outputPath = input('Enter folder path to download tails:')
            print(outputPath)
            if ms.ensureDirectory(outputPath):
                for path in self.tail_URL_NamesList:
                    download_url(path, outputPath)
                print(f"Tails dawnloaded to: {outputPath}")      
            else:
                for path in self.tail_URL_NamesList:
                    download_url(path, currentDirectory)
                print(f"Tails dawnloaded to: {currentDirectory}") 
   

# Helpers
def setWBTWorkingDir(workingDir):
    wbt.set_working_dir(workingDir)

def sheckTifExtention(fileName):
    if ".tif" not in fileName:
            newFileName = input("enter a valid file name with the '.tif' extention")
    return newFileName



#### Exceutable 
def main():
    # list = ms.importListFromExelCol('/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodProbabRNCanAbd/saint_john_NFL_DTM.xlsx','Feuil1','ftp_dtm')
    # importer = dtmTailImporter(list, '/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodMaps/testZone')
    # importer.impotTailToLocalDir()
    setWBTWorkingDir('/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodMaps/testZone')
    transformer = dtmTransformer('/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodMaps/testZone')
    transformer.computeMosaic()
    
if __name__ == "__main__":
    main()
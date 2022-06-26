import os
import services
from whitebox.whitebox_tools import WhiteboxTools, default_callback
from torchgeo.datasets.utils import download_url


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
            if services.ensureDirectory(outputPath):
                for path in self.tail_URL_NamesList:
                    download_url(path, outputPath)
                print(f"Tails dawnloaded to: {outputPath}")      
            else:
                for path in self.tail_URL_NamesList:
                    download_url(path, currentDirectory)
                print(f"Tails dawnloaded to: {currentDirectory}") 
   
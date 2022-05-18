
import os
import tempfile
from torch.utils.data import DataLoader

from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler

class tailImporter():
    def init(self, mainDownloadPath, tailNamesList):
        self.data_root = mainDownloadPath
        self.tailNameList = tailNamesList

    def impotToDirectory(self,destiniPath):
        for tile in self.tailNameList:
            download_url(self.data_root + tile, destiniPath)

    def set_mainDownloadPath(self,new_mainDownloadPath):
       self.data_root = new_mainDownloadPath


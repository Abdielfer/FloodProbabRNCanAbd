'''
'''
import os
import tempfile
from typing_extensions import Self

from torch.utils.data import DataLoader

from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler

class tailImporter():
    def init(mainPath, tailNamesList):
        self.data_root = mainPath
        self.tailNameList = tailNamesList
    
    def impotToDirectory(destiniPath):
        naip_url = "https://naipblobs.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
        for tile in self.tailNameList:
            download_url(naip_url + tile, destiniPath)



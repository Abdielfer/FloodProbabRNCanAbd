import pandas as pd
from numpy import where
from imblearn.under_sampling import NearMiss

class underSampling():
    def __init__(self, dataset, targetCol) -> None:
        self.dataset = dataset
        self.targetCol = targetCol
        return
    
    def stratifiedUndersampling(self, majotityClass):
        row_ix = where(self.dataset[self.targetCol] == majotityClass)[0]
        df = self.datset[row_ix,:]  # Extract features of majority class
        ### in progress   ####
        return df 


class overSampling():
    def __init__(self) -> None:
        pass
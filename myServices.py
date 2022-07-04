import os
import pandas as pd

def getLocalPath():
    return os.getcwd()

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



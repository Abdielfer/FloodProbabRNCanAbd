import os
import services as svs

# # Import tails form an xxcell doc. ]
excell_file = '/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodProbabRNCanAbd/saint_john_NFL_DTM.xlsx'
listTails = svs.importListFromExelCol(excell_file,'Feuil1','ftp_dtm')
importer = svs.dtmTailImporter(listTails)
importer.impotTailToLocalDir()

# ## my work directory: /Users/abdielfer/DESS/Internship2022/RNCanWork/FloodMaps/testZone
# transformer = svs.dtmTransformer('/Users/abdielfer/DESS/Internship2022/RNCanWork/FloodMaps/testZone')

# outpouDTMName = 'zone_one.tif' 
# inDTMFilled = "filled_"+outpouDTMName
# inD8Pointer = "d8Pointer_filled_zone_one.tif"  
# d8FAccOutputName = "d8fllowAcc"+inDTMFilled

# # Output from Mosaic is input for the rest of the process
# transformer.computeMosaic(outpouDTMName)
# transformer.fixNoDataAndfillDTM(outpouDTMName)
# transformer.d8FPointerRasterCalculation(inDTMFilled)
# transformer.d8_flow_accumulation(inDTMFilled)
# inOutlest = os.path.join(transformer.workingDir,"outlets.shp")
# transformer.jensePourPoint(inOutlest,d8FAccOutputName)
# jensenOutput =os.path.join(transformer.workingDir,"correctedSnapPoints.shp")
# transformer.watershedConputing(inD8Pointer, jensenOutput)
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 09:33:41 2020

@author: mpoujol
"""

# Reset of the environment before starting (if you have problem with that because of a missing package, just remove this part and restart the evironnement by hand)
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-f')         # magic interpreter to run "%reset -f" to erase workspace
get_ipython().run_line_magic('matplotlib', 'qt5')    # magic interpreter to run "%matplotlib qt"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import fnmatch
from sklearn import preprocessing
DEG_RAD = np.pi/180.                    # conversion of degrees to radians

#%%
'''
COMPLETE THIS PART BEFORE STARTING
'''

#%%  Usefull variables
dataFolder ="P:\mpoujol\modeles\Python\Projets\EUvsVirus\Data"    #data path
rawSafeDataFolder = "P:\mpoujol\modeles\Python\Projets\EUvsVirus\Data\RawData\Safe"   #paths of data files with safe situations
rawRiskyDataFolder = "P:\mpoujol\modeles\Python\Projets\EUvsVirus\Data\RawData\Risky"   #paths of data files with risky situations
dirSep="\\"    #separator
fileType = '*.txt'

SEQLENGTH = 12                         # sequences lengths of input data
MINSEQSHIFT = SEQLENGTH               # minimum shift between consecutive sequences (if = SEQLENGTH, there is no overlapping)
varNames = ["sinPitch","cosPitch","sinRoll","cosRoll","sinYaw","cosYaw"]
nbVar = len(varNames)
datasetFilename = "DataSet3_shift_" + str(MINSEQSHIFT) + "_"+ str(SEQLENGTH)

#%%
listDataArray = list()  # Data of valid sequences 
listFreq = list()       #number of estimated
topMargin = 3
botMargin = 2
lrMargin = 1            # number of left and right columns to drop

nbSeqTot = 0   
nbFilesTot = 0
Xlist = list()
Ylist = list()
for iRisk in range(0,2):
    if iRisk:
        nbSeqTotRisk = 0   
        rawDataFolder = rawRiskyDataFolder
    else:
        rawDataFolder = rawSafeDataFolder
        
    # List of all files 
    listFileNames = listdir(rawDataFolder)
    listFileNames.sort()                                # sorting of filenames by alphabetical order
    listFilteredFileNames = fnmatch.filter(listFileNames, fileType)
    nbFiles = len(listFilteredFileNames)
    nbFilesTot = nbFilesTot + nbFiles

    for ifl in range(0,nbFiles):                    # loop on files
        
        print('File name : ', listFilteredFileNames[ifl])
        
        #read data
        dfRawData_i = pd.read_csv(rawDataFolder + dirSep + listFilteredFileNames[ifl], sep="[;,|]", 
                                  skiprows =topMargin, engine='python')
    
        nbObs_i = dfRawData_i.shape[0]
        nbObs_i = nbObs_i - botMargin
        
        dates = pd.to_datetime(dfRawData_i.iloc[:nbObs_i,0], infer_datetime_format= True)
        timestamps = np.array([dates[it].timestamp() for it in range(0,nbObs_i)], dtype='float')
        timestamps = timestamps - timestamps[0]
        freq_i = (nbObs_i-1.)/timestamps[-1:][0]
        listFreq.append(freq_i)
        print(ifl, freq_i)
        
        dataArray_scl = np.zeros((nbObs_i, nbVar), dtype= float)
        
        # Conversion in sinus and cosinus for both :
        # - normalization of the inputs between [-1, 1]
        # - ensuring continuity of measurements folding angle borders due to periodicity
        # - facilitating the modeling by the neural network of rotations in space (sort of Euler's angles)
        dataArray_scl [:,0] = np.sin(dfRawData_i.iloc[:nbObs_i,1]*DEG_RAD)      # sinus of pitch
        dataArray_scl [:,1] = np.cos(dfRawData_i.iloc[:nbObs_i,1]*DEG_RAD)      # cosinus of pitch
        dataArray_scl [:,2] = np.sin(dfRawData_i.iloc[:nbObs_i,2]*DEG_RAD)      # sinus of roll
        dataArray_scl [:,3] = np.cos(dfRawData_i.iloc[:nbObs_i,2]*DEG_RAD)      # cosinus of roll
        dataArray_scl [:,4] = np.sin(dfRawData_i.iloc[:nbObs_i,3]*DEG_RAD)      # sinus of yaw
        dataArray_scl [:,5] = np.cos(dfRawData_i.iloc[:nbObs_i,3]*DEG_RAD)      # cosinus of yaw
    
        iT1 = 0
        totSeqLen = int(nbObs_i)
        iT2 = iT1 + totSeqLen -1
    
        if totSeqLen > SEQLENGTH:
            nbSeq = int(np.floor(1+ (totSeqLen - SEQLENGTH)/MINSEQSHIFT))
            if nbSeq > 1 :
                seqShift = int(np.floor((totSeqLen - SEQLENGTH)/(nbSeq -1)))
            else:
                seqShift = 0
            
        for iSeq in range(0,nbSeq):
            nbSeqTot = nbSeqTot + 1                
            if iRisk:
                nbSeqTotRisk = nbSeqTotRisk + 1

            it1 = iT1 + iSeq * seqShift
            it2 = it1 + SEQLENGTH
            if it2 > iT2:
                it2 = iT2
                it1 = it2 - SEQLENGTH
    
            X_seq = np.array(dataArray_scl[it1:it2,:])
    
            Xlist.append(X_seq)
            Ylist.append(iRisk)

X = np.array(Xlist).reshape(nbSeqTot, SEQLENGTH, nbVar)            # 3 dimensionnal array
Y = np.array(Ylist).reshape(nbSeqTot,)            # 3 dimensionnal array

#%%
# Saving of numpy arrays along with scaling parameters in a NPZ file embedding all binary files
varIndex = dict(zip(varNames, np.arange(0,nbVar)))
np.savez(dataFolder+dirSep+datasetFilename, X = X, Y = Y)
print("DataSet Saved")

#%%
# Estimation of mean sampling frequency over all data
freqs = np.array(listFreq).reshape(nbFilesTot,)
meanFreq = np.mean(freqs)

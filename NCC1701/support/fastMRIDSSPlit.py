"""
To generate Train and Test set from the Train Set (as supplied by FastMRI)
"""

import os
import pathlib
import random
import pandas as pd
import h5py

def GenerateTrainTestSet(setname, input_xlsx_path, output_xlsx_path, filters=None, nTrain=None, nTest=None, nVal=None, percentTrain=0.5, percentTest = 0.3):
    df = pd.read_excel(input_xlsx_path)
    
    if filters is not None and filters != {}:
        dfFilter = None
        for k,v in filters.items():
            if dfFilter is None:
                dfFilter = df[k].eq(v)
            else:
                dfFilter &= df[k].eq(v)
        df = df[dfFilter]

    if nTrain is None:
        nTrain = int(len(df) * percentTrain)
    if nTest is None:
        nTest = int(len(df) * percentTest)
    if nVal is None:
        nVal = len(df)-nTrain-nTest 

    train_set = df.sample(n=nTrain, replace=False, random_state=1701) 
    train_set.to_excel(os.path.join(output_xlsx_path, 'train_'+setname+'.xlsx'))

    remaining = pd.concat([df, train_set]).drop_duplicates(keep=False)
    test_set = remaining.sample(n=nTest, replace=False, random_state=1701)    
    test_set.to_excel(os.path.join(output_xlsx_path, 'test_'+setname+'.xlsx'))

    remaining2 = pd.concat([remaining, test_set]).drop_duplicates(keep=False)
    if len(remaining2) > 0:
        val_set = remaining2.sample(n=nVal, replace=False, random_state=1701)
        val_set.to_excel(os.path.join(output_xlsx_path, 'val_'+setname+'.xlsx'))

    print('done')
        



#filtersDict = {'nCoil': 16, 'hKSP': 640, 'wKSP': 320}
#setname = 'MultiContrast_16Coil_640x320_Split503020'

filtersDict = {'acquisition': 'AXT2', 'nCoil': 16, 'hKSP': 768, 'wKSP': 396}
setname = 'AXT2_16Coil_768x396_Split503020'



GenerateTrainTestSet(setname,r"B:\Soumick\Challange DSs\fastMRI\Brain\multicoil_train.xlsx",r"B:\Soumick\Challange DSs\fastMRI\Brain\Splits",filtersDict)
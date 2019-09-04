#!/usr/bin/env python3.6

import pandas as pd
import sys

col_names = [
    "x1", # 1 gamma detected x position [cm]
    "y1", # 1 gamma detected y position [cm]
    "z1", # 1 gamma detected z position [cm]
    "t1", # 1 gamma detection time [ps]
    "x2", # 2 gamma detected x position [cm]
    "y2", # 2 gamma detected y position [cm]
    "z2", # 2 gamma detected z position [cm]
    "t2", # 2 gamma detection time [ps]
    "vol1", # 1 gamma volume ID
    "vol2", # 2 gamma volume ID
    "e1", # 1 gamma energy loss during detection [keV]
    "e2", # 2 gamma energy loss during detection [keV]
    "class", # Type of coincidence(1-true, 2-phantom-scattered, 3-detector-scattered, 4-accidental)
    "sX1", # 1 gamma emission x position [cm]
    "sY1", # 1 gamma emission y position [cm]
    "sZ1" # 1 gamma emission z position [cm]
] 

file_name = sys.argv[1]
df = pd.read_csv(file_name, sep = "\t", names = col_names)
convert_dict = {
    'vol1': int, 
    'vol2': int,
    'class': int
} 
df = df.astype(convert_dict)
print(df.head())
df.to_csv(file_name + '_goja', sep = "\t", header = False, index = False)
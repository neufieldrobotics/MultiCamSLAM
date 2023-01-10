import glob
import utm
import os
import pandas as pd
import numpy as np

##############################################################
############### PROCESS Groundtruth Data -GPS ################
##############################################################
# before running this script make sure to run rosbagtookit's
# bag2pandas scripts to get the .pkl file
#read the pkl file of gps
data_path = "/data/05_05_2022_whoi/fixed_expo"
files = glob.glob(os.path.join(data_path, "*.pkl")) #os.listdir(os.path.join(data_path)
files.sort()
print(files[0])
all_data=[]
for f in files:
    df = pd.read_pickle(f)
    df_gps = df['robot']['gps']
    df_gps.reset_index(inplace=True)
    print("Read ", f)
    for i in range(df_gps.shape[0]):
        x, y, _,_ = utm.from_latlon(df_gps.iloc[i,1], df_gps.iloc[i,2])
        z = df_gps.iloc[i,3]
        t = pd.Timestamp(df_gps.iloc[i,0]).timestamp()
        all_data.append([t, -1.0*y, z, x, 0.0, 0.0, 0.0, 1.0])
        #print(t," ",x," ",y," ",z)
np_data = np.array(all_data, np.float64)
pd_data = pd.DataFrame(np_data)
print(pd_data.head())
print(pd_data.tail())
pd_data[[1,2,3]] = pd_data[[1,2,3]] - pd_data.iloc[0, 1:4]
print(pd_data.head())
pd_data.to_csv(os.path.join(data_path, "gps_groundtruth.txt"), sep=' ', header=False, index=False)
print(pd_data.tail())
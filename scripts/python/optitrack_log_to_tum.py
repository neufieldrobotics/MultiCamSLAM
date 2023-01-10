import pandas as pd
from datetime import datetime


def read_optitrack_csv(filename, start_time_string):
    """
    This method read the optitrack log file and returns a pandas dataframe
    with columns in TUM format.
    example usage : optitrack = read_optitrack_csv("/data/05_14_2022/inside_loop_opti.csv","2022-05-14 03.43.29.875PM" )
    Args:
        filename: optitrack log file path
        start_time_string: start time that is present in the logfile . something like "2022-05-14 03.43.29.875PM"

    Returns: pandas dataframe

    """
    dt = datetime.strptime(start_time_string, '%Y-%m-%d %I.%M.%S.%f%p')
    timestamp = datetime.timestamp(dt)
    df = pd.read_csv(filename, header=[5])
    df = df.dropna()
    df['Time (Seconds)'] += timestamp
    df = df.rename(columns={"Time (Seconds)" : "timestamp", "X.1": "tx", "Y.1" : "ty", "Z.1" : "tz"})
    gt_data = df.drop(['Frame', "Unnamed: 9"], axis=1)
    cols = ['timestamp', 'tx', 'ty', 'tz', 'X', 'Y', 'Z', 'W']
    gt_data = gt_data[cols]
    return gt_data

#gt_data = read_optitrack_csv("/data/05_14_2022/inside_loop_opti.csv","2022-05-14 03.43.29.875PM" )
#gt_data = gt_data[gt_data.index % 20 == 0]
#theta = np.pi
#rot_apply = np.array([[np.cos(theta), np.sin(theta), 0],[-1*np.sin(theta), np.cos(theta), 0],[0,0,1]])
#print(rot_apply)
#for row in range(gt_data.shape[0]):
#    quat = np.array(gt_data.iloc[row, 4:8])
#    rot = R.from_quat(quat).as_matrix()
#    rot_1 =  rot @ rot_apply
#    gt_data.iloc[row, -4:] = R.from_matrix(rot_1).as_quat()
#    #print(rot)
#    #print(rot_1)
#    print("---------")
#If we need to transform the data we can - Not needed for now
#gt_data.to_csv("/data/05_14_2022/inside_loop/inside_loop_opti.txt", sep=' ', header=False, index=False)
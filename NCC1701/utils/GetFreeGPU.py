import subprocess
import sys
from io import StringIO
import pandas as pd

def get_free_gpu():
    gpu_stats = subprocess.check_output(["/home/schatter/nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(str(gpu_stats,'utf-8')),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('Current GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]')).astype('int')  
    idx = gpu_df['memory.free'].idxmax()
    print('Selecting GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx



import numpy as np
import pandas as pd
import math
data = pd.read_csv('MRMD_S.csv')
X=np.matrix(data)
optimal_feature =X[:,1:]
feature = pd.read_csv('S_feature_sum.csv')
X1=np.matrix(feature)
X2=X1[:,1:]
A=[int(x) for x in optimal_feature]
set_end_MRMD=X2[:, A[:165]]
data_csv = pd.DataFrame(data=set_end_MRMD)
data_csv.to_csv('S_MRMD.csv')


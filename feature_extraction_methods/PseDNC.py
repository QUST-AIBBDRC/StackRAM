from util import get_data
from util import normalize_index
import pandas as pd
import numpy as np
def check_psenac(lamada, w, k):
    """Check the validation of parameter lamada, w and k.
    """
    try:
        if not isinstance(lamada, int) or lamada <= 0:
            raise ValueError("Error, parameter lamada must be an int type and larger than and equal to 0.")
        elif w > 1 or w < 0:
            raise ValueError("Error, parameter w must be ranged from 0 to 1.")
        elif not isinstance(k, int) or k <= 0:
            raise ValueError("Error, parameter k must be an int type and larger than 0.")
    except ValueError:
        raise


class PseDNC():
    def __init__(self, lamada=3, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 2
        check_psenac(self.lamada, self.w, self.k)

    def make_psednc_vec(self, input_data, phyche):

        sequence_list = get_data(input_data)
        #sequence_list, phyche_value = get_sequence_list_and_phyche_value_psednc(input_data, extra_phyche_index)
        phyche_value=phyche
        from psenacutil import make_pseknc_vector

        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)

        return vector

phy=pd.read_csv('phy.csv',header=-1,index_col=None)
phyche_index=np.array(phy)
phyche_index_dict=normalize_index(phyche_index, is_convert_dict=True)
psednc = PseDNC(lamada=23, w=0.05)
vec = psednc.make_psednc_vec(open('S_data.txt'),phyche=phyche_index_dict)
#print(len(vec[0]))
feature=np.array(vec)
data_new=np.matrix(feature)
data_PseDNC=pd.DataFrame(data=data_new)
data_PseDNC.to_csv('PseDNC_S.csv')


    
import sys,re,os
from functools import reduce
from collections import Counter
import pandas as pd
import numpy as np
import itertools
import platform
from math import sqrt
from math import pow
def readRNAFasta(file):
	with open(file) as f:
		records = f.read()

	if re.search('>', records) == None:
		print('The input RNA sequence must be fasta format.')
		sys.exit(1)
	records = records.split('>')[1:]
	myFasta = []
	for fasta in records:
		array = fasta.split('\n')
		name, sequence = array[0].split()[0], re.sub('[^ACGU-]', '-', ''.join(array[1:]).upper())
		myFasta.append([name, sequence])
	return myFasta



#ALPHABET='ACGU'

def chemical(input_data):
#    fastas=readRNAFasta(input_data)
    vector = []
    for i in fastas:
        sequence =  i[1]
        code = []
        for aa in sequence:
            if aa == 'A':
                code = code + [1,1,1]
            elif aa == 'C':
                code = code + [0,1,0]
            elif aa == 'G':
                code = code + [1,0,0]
            elif aa == 'U':
                code = code + [0,0,1]
        vector.append(code)
    return vector


input_data='S_data.txt'
fastas=readRNAFasta(input_data)
vector=chemical(fastas)
csv_data=pd.DataFrame(data=vector)
csv_data.to_csv('binary_S.csv',header=False,index=False)


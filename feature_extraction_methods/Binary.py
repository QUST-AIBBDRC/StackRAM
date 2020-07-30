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



ALPHABET='ACGU'

def binary(input_data):
#    fastas=readRNAFasta(input_data)
    vector = []
#    header = ['#']
#    for i in range(1, len(fastas[0][1]) * 4 + 1):#得到长度
#        header.append('BINARY.F'+str(i))
#    vector.append(header)
    for i in fastas:
        sequence =  i[1]
        code = []
        for aa in sequence:
            if aa == '-':
                code = code + [0, 0, 0, 0]
                continue
            for aa1 in ALPHABET:
                tag = 1 if aa == aa1 else 0
                code.append(tag)
        vector.append(code)
    return vector


input_data='S_data.txt'
fastas=readRNAFasta(input_data)
vector=binary(fastas)
csv_data=pd.DataFrame(data=vector)
csv_data.to_csv('binary_S.csv',header=False,index=False)
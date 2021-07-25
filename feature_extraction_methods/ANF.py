import sys,re,os
from functools import reduce
from collections import Counter
import pandas as pd
import numpy as np
import itertools
import platform
from math import sqrt
from math import pow
from collections import Counter
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

def ANF(input_data):
#    fastas=readRNAFasta(input_data)
    vector = []
    
    for i in fastas:
        sequence =  i[1]#获得每条序列
        l = len (sequence)#序列长度
        code=[]
        for j in range(l):
            str1=sequence[:j+1]#获得子序列
            count = Counter(str1)
            key=str1[-1]            
            fea = count[key]/len(str1)
            code.append(fea)
        vector.append(code[1:])
    return vector



input_data='S_data.txt'
fastas=readRNAFasta(input_data)
vector=ANF(fastas)
csv_data=pd.DataFrame(data=vector)
csv_data.to_csv('ANF_S.csv',header=False,index=False)

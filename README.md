##StackRAM
StackRAM: a cross-species method for identifying RNA N6-methyladenosine sites based on stacked ensemble.

###StackRAM uses the following dependencies:
* python 3.6 
* numpy
* scipy
* scikit-learn

##Guiding principles:
**The dataset file contains three categories N6-methyladenosine sites datasets, which contain training dataset and independent test dataset.

**feature extraction methods:
   ANF.py is the implementation of nucleotide frequency.
   Binary.py is the implementation of binary encoding.
   NCP.py is the implementation of chemical property.
   K-mer.py is the implementation of k-mer nucleotide frequency.
   PseDNC.py is the implementation of pseudo dinucleotide composition.
   PSTNP.py is the implementation of position-specific trinucleotide propensity.
   
** feature selection methods:
   LLE.py represents the locally linear embedding .
   MRMD.py represents the max-relevance-max-distance.
   SE.py represents the spectral embedding.
   SVD.py represents the singular value decomposition.
   MI.py represents the mutual information.
   ET.py represents the extra-trees.
   Elastic_Net.py represents the Elastic Net.
     
** classifier:
   AdaBoost.py is the implementation of AdaBoost.
   ERT.py is the implementation of ERT.
   KNN.py is the implementation of KNN.
   XGBoost.py is the implementation of XGBoost.
   RF.py is the implementation of RF.
   LightGBM.py is the implementation of LightGBM.
   SVM.py is the implementation of SVM.
   Stack_LR.py is the implementation of stacked ensemble, which second-stage classifier is LR.
   Stack_SVM.py is the implementation of stacked ensemble, which second-stage classifier is SVM.


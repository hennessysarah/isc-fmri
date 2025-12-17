# computes permutation test to assess whether the ROI-based ISC is significant

import numpy as np
import matplotlib.pyplot as plt
from brainiak.isc import permutation_isc
import pickle


import pickle
import os
import brainiak
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, norm
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import pandas as pd
import numpy as np

import nibabel as nib
from copy import deepcopy
from brainiak import image, io

#from scipy.stats import pearsonr, norm
from nilearn.glm import fdr_threshold


pink = '\033[95m'
yellow = '\033[93m'
blue = '\033[94m'
cyan = '\033[96m'


n_permutations = 10000
summary_statistic = 'median'

ks = int((n_permutations)/1000)
kperm = str(ks) + "kperm"


alphalevel = 0.05

alphalevelname = str(alphalevel).replace('.', '')


results_dir = "/ISC/results/OA_fam_groupwise/ROI"
beatles_dir = "/ISC/data/"

roiresults_path = results_dir + "/ROI_results_%s_%s.csv" %(alphalevelname,kperm)

roi_list = [elem for elem in os.listdir(results_dir) if elem.endswith('_groupwise.pkl')]

#roi_list = [elem for elem in os.listdir(results_dir) if elem.endswith('_groupwise.pkl') and elem.startswith('L_')]
print(pink+ "YOUR ROI LIST IS: \n", roi_list)


colnames = ['ROI', 'p_avg']
p_avg = []
ROI = []


newdf = pd.DataFrame(columns=colnames, index = range(len(roi_list)) )

i = -1

for roi in roi_list:
    i = i + 1
    roi_name = roi[:-29]
    ROI.append(roi_name)
    roi_pkl_path = results_dir + "/" + roi
    print("about to load pickle file!")


    with open(roi_pkl_path, 'rb') as f:
        isc_maps = pickle.load(f)

    print(pink+ "loaded successfully")

    print(yellow + "##### STARTING PERMUTATION TESTING for: %s ##########" %(roi_name))

    observed_pickle_path = results_dir + '/%s_permuted_ISC_observed_%s.pkl' %(roi_name, kperm)
    p_value_pickle_path = results_dir + '/%s_permuted_ISC_p_%s.pkl' %(roi_name, kperm)
    permuted_ISC_distribution_pickle_path = results_dir + '/%s_permuted_ISC_distribution_%s.pkl' %(roi_name, kperm)

    if not os.path.exists(permuted_ISC_distribution_pickle_path):
        print(" please do not close out of this window. i am still running. ")

        observed, p, distribution = permutation_isc(
            isc_maps,
            pairwise= False,
            summary_statistic=summary_statistic,
            n_permutations=n_permutations
        )

        print(pink+ "####### DONE PERMUTING!!! phew. ########")

        p = p.ravel()
        observed = observed.ravel()

        print('observed: {}'.format(np.shape(observed)))
        print('p: {}'.format(np.shape(p)))
        print('distribution: {}'.format(np.shape(distribution)))



        # Save observed and p-values to pickle files



        print(yellow + "saving files......")


        with open(observed_pickle_path, 'wb') as f:
            pickle.dump(observed, f)

        with open(p_value_pickle_path, 'wb') as f:
            pickle.dump(p, f)

        with open(permuted_ISC_distribution_pickle_path, 'wb') as f:
            pickle.dump(distribution, f)

        print(pink+ "Done saving permuted file for %s." %(roi_name))

    else:
        with open(observed_pickle_path, 'rb') as f:
            observed = pickle.load(f)

        with open(p_value_pickle_path, 'rb') as f:
            p = pickle.load(f)

        with open(permuted_ISC_distribution_pickle_path, 'rb') as f:
            distribution = pickle.load(f)


    print(cyan + "################ Beginning thresholding ####################")


    avgallp = np.nanmean(p)
    print("Average total p value for %s is: %.3f" %(roi_name, avgallp))
    p_avg.append(avgallp)



newdf['p_avg'] = p_avg

newdf['ROI'] = ROI


newdf.to_csv(roiresults_path, index=False)
print("Saved results to ", roiresults_path)

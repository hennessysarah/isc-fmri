# takes pkl file of p values from ISC permutation.
# 1. thresholds so that p values & R values that pass FDR are only included
# 2. turns those files into a nifti readable by fsleyes, etc. for visualization

import numpy as np
import nibabel as nib
import pickle
import os
from copy import deepcopy
from brainiak import image, io
from scipy.stats import pearsonr, norm
from nilearn.glm import fdr_threshold

import re

#########################
## Set up
########################
pink = '\033[95m'
yellow = '\033[93m'
beatles_dir = "/ISC/data/"

results_path = "/ISC/results/OA_fam_groupwise/"

p_value_pickle_path = results_path+ "permuted_ISC_p_10kperm.pkl"
r_value_pickle_path = results_path + "permuted_ISC_observed_10kperm.pkl"


print(pink + 'loading pickle file of p values...')

# Load the p-value data from the pickle file
with open(p_value_pickle_path, 'rb') as f:
    p_values = pickle.load(f)

print("loaded successfully. ")


print(yellow + 'loading pickle file of R values...')

# Load the p-value data from the pickle file
with open(r_value_pickle_path, 'rb') as f:
    r_values = pickle.load(f)

print("loaded successfully. ")



#########################
## FDR Correction
########################

# convert p map to z map

z_map = norm.ppf(1 - (p_values / 2))

print("Z MAP")
print(z_map)

alphalevel = 0.01

alphalevelname = str(alphalevel).replace('.', '')
# use nilearn.glm.fdr_threshold to get a thresholded map
thresh = np.empty(shape=z_map.shape)
thresh = fdr_threshold(z_map, alpha=alphalevel)
print(pink + "MY FDR THRESHOLD IS Z= %f" %(thresh))
pthresh = 1 - norm.cdf(thresh)
print("FDR P treshold is %f" %(pthresh))



# plot the thresholded r value map
thresh_r_map = deepcopy(r_values)
non_zero_count = np.count_nonzero(thresh_r_map)
print(yellow + "Number of NON ZEROs before thresholding ...")
print(non_zero_count)

thresh_r_map[z_map < thresh] = 0
non_zero_count = np.count_nonzero(thresh_r_map)
print("Number of NON ZEROs after thresholding...")
print(non_zero_count)

#kperm = re.search(r'(\d+kperm)\.pkl', p_value_pickle_path)
kperm = "10kperm"

significant_r_values_pickle_path = results_path + "FDRcorrected_r_values_%s_%s.pkl" %(alphalevelname, kperm)


if os.path.exists(significant_r_values_pickle_path):
    with open(significant_r_values_pickle_path, 'rb') as f:
        thresh_r_map = pickle.load(f)
        print("Significant R-values pickle file loaded successfully!")
else:
    with open(significant_r_values_pickle_path, 'wb') as f:
        pickle.dump(thresh_r_map, f)
        print("Significant R-values pickle file saved successfully!")


non_zero_count = np.count_nonzero(thresh_r_map)
print("Number of NON ZEROs in my loaded file...")
print(non_zero_count)

# Assuming you have a brain template (e.g., MNI152 template)
# Load the template to get header information


dir_mask = os.path.join(beatles_dir, 'masks/')
mask_name = os.path.join(dir_mask, 'MNI152_T1_2mm_brain_mask.nii.gz')

print(pink+ 'loading mask')
brain_nii = nib.load(mask_name)
brain_mask = io.load_boolean_mask(mask_name)
template_header = brain_nii.header
print('loaded template successfully.')



# Make the ISC output a volume
# Load in the brain data for reference
#brain_nii = nib.load(mask_name)

# Get the list of nonzero voxel coordinates
coords = np.where(brain_mask)



# Save the significant p-value NIfTI file
output_path = results_path+ "/wholebrain_ISC_FDRcorrected_r_%s_%s.nii.gz" %(alphalevelname, kperm)

if not os.path.exists(output_path):
# Map the ISC data for the first participant into brain space
    print('my shape is:')
    print(thresh_r_map.shape)
    isc_vol = np.zeros(brain_nii.shape)

#matt put this:
    #isc_vol[coords] = isc_map[:,f]
#but this is what works?
    isc_vol[coords] = thresh_r_map

    print("# of non-zeros in my volume to save:")
    non_zero_count = np.count_nonzero(isc_vol)
    print(non_zero_count)

    #print(isc_vol)


    # Save the ISC data as a volume
    isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)
    print(pink+ "saving nifti")
    nib.save(isc_nifti,output_path)



#nib.save(significant_p_value_img, output_path)

print(yellow+ "######\n Significant, FDR corrected, R value NIfTI file saved successfully! Go take a look!!!!!\n #####")

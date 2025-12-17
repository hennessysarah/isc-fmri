

# align the pairs present in isc file with the pairs present in behavioral pairs file
# for ISC analysis music-evoked nostalgia

# in this example, had a sheet called "SNS" (southampton nostalgia scale), whch is a trait-level measure of nostalgia proneness
# comparing trait-level differences to ISC (brain)

import pickle
import os
import brainiak
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, norm

import nibabel as nib
from copy import deepcopy
from brainiak import image, io
import pandas as pd

#from scipy.stats import pearsonr, norm
from nilearn.glm import fdr_threshold


pink = '\033[95m'
yellow = '\033[93m'
blue = '\033[94m'
cyan = '\033[96m'


alphalevel = 0.05
alphalevelname = str(alphalevel).replace('.', '')

## -----------
## Set up
## -------------

results_path = "/ISC/results/OA_fam_groupwise/"
beatles_dir = "/ISC/data/"

behavioral = results_path + "SNS_ratings.csv"
isc_maps = results_path+ "isc_beatles_n27_groupwise.pkl"

print(pink+"loading behavioral ratings..")
behavioral = pd.read_csv(behavioral)

print("shape of behavioral ratings:")
print(behavioral.shape)


print("loading ISC maps..")
with open(isc_maps, 'rb') as f:
    isc_maps = pickle.load(f)

print(yellow+ "shape of isc maps:")
print(isc_maps.shape)

## -----------
## Create Null distribution

## -------------

print(blue + "#########\n Creating Null Distribution\n #########")

n_perm = 100000 #make this larger  later

# do permutations on just one voxel
# we operate on the assumption that every voxel has a similar-enough null distribution
# run permutation testing for the brain to consensus correlation to test null hypothesis that there is no correlation
# between one voxel and correlation of behavioral ratings
vox_idx = 1000  # pick just one voxel to do permutations on

perm_vox = np.empty(shape=(n_perm, 2))  #1 rating, n_perm, r and p

kperm = "100kperm"

perm_vox_path = results_path + "SNS_perm_vox_%s_alpha%s.pkl" %(kperm,alphalevelname)

# Create the null distribution at 1 voxel
if not os.path.exists(perm_vox_path):  # only compute if filedoes not exist
    rng = np.random.default_rng()  # for rng.permutation. generates random numbers
    for i in tqdm(range(n_perm)):  # number of permutations
        #nan_mask = ~np.isnan(behavioral[:, 1])  # mask to ignore nans for any given pair (shouldn't ever happen.)
        missing_mask = behavioral.isna().any(axis=1)
        not_missing_mask = ~missing_mask

        x = isc_maps.T[vox_idx][not_missing_mask]
        y= behavioral.iloc[:, 2][not_missing_mask].values #taking meancentered values

        perm_vox[i] = pearsonr(x,rng.permutation(y))

    # save perm to pickle
    with open(perm_vox_path, 'wb') as f:
        pickle.dump(perm_vox, f)
else:
    with open(perm_vox_path, 'rb') as f:
        perm_vox = pickle.load(f)

print( "finished Null Distribution. ")

alpha = alphalevel

critical_value = np.percentile(perm_vox, 100 * (1 - alpha))
print(yellow+  "Critical Value is :")
print(critical_value)


## -----------
## COMPUTE CORRELATION BETWEEN BEH AND BRAIN
## -------------


# compute correlation between each SNSalgia score(behavioral) and each r value at each voxel


SNS_map = np.empty(shape=(isc_maps.shape[1], 2))
SNS_map_path = results_path + "SNS_map_raw.pkl"

if not os.path.exists(SNS_map_path):
    
       #### resulting map is only ONE R value for each voxel,
    # because you're taking a correlation of all of the isc r values with all of the SNS scores


    print(cyan+ '######### \ncomputing SNS_map\n #########')


    for i in tqdm(range(isc_maps.shape[1])): # for each voxel
        #print(i)

        #find the missing values of ISC maps. (there should be no missing values of beh..)
        x = isc_maps.T[i]
        missing_mask = np.isnan(x)
        not_missing_mask = ~missing_mask

        x = x[not_missing_mask]

        if np.isnan(x).any():
            print("isc_maps.T[i] contains missing values")

        y= behavioral.iloc[:, 2][not_missing_mask].values #taking meancentered values

        SNS_map[i] = pearsonr(x, y)

    # save SNS_map to pickle
    with open(SNS_map_path, 'wb') as f:
        pickle.dump(SNS_map, f)
else:
    with open(SNS_map_path, 'rb') as f:
        SNS_map = pickle.load(f)

# that will make another array, 1 R value for each voxel

print(yellow+ "Shape of SNS Map is:")
print(SNS_map.shape)

print(cyan + "taking only the Rs")

SNS_map = SNS_map[:,0]

## -----------
## Use P value value to threshold the SNSalgia mask with FDR correction
## -------------
print(pink+ "######\n starting thresholding....\n######### ")

print("calculating p values") # note.... i think the pearsonr function already does this.. .... oops


p_values = np.empty(shape=(SNS_map.shape[0], ))
p_value_path = results_path + "SNS_p_values_alphalevel_%s_%s_raw.pkl" %(alphalevelname, kperm)
for i in tqdm(range(SNS_map.shape[0])):
     # for each voxel
     observed_R = SNS_map[i]
     p_values[i]= np.mean(perm_vox >= observed_R)

with open(p_value_path, 'wb') as f:
    pickle.dump(p_values, f)

print("finding sig p values with FDR correction")


z_map = norm.ppf(1 - (p_values / 2))

print("Z MAP")
print(z_map)

#alphalevel = 0.01
# use nilearn.glm.fdr_threshold to get a thresholded map
thresh = np.empty(shape=z_map.shape)
thresh = fdr_threshold(z_map, alpha=alphalevel)
print(pink + "MY FDR THRESHOLD for alpha = %s IS Z= %f" %(alphalevelname, thresh))
pthresh = 1 - norm.cdf(thresh)
print("FDR P treshold is %f" %(pthresh))


r_values = SNS_map
# plot the thresholded r value map
thresh_r_map = deepcopy(r_values)
non_zero_count = np.count_nonzero(thresh_r_map)
print(yellow + "Number of NON ZEROs before thresholding ...")
print(non_zero_count)

thresh_r_map[z_map < thresh] = 0
non_zero_count = np.count_nonzero(thresh_r_map)
print("Number of NON ZEROs after thresholding...")
print(non_zero_count)

significant_r_values_pickle_path = results_path + "SNS_FDRcorrected_r_values_%s_%s_raw.pkl" %(kperm,alphalevelname)


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


## -----------
## make a nifti file of the sig ones
## -------------

print(yellow+ "###### \nSaving FDR corrected r val mask as a nifty lil' NIFTI....\n######### ")

dir_mask = os.path.join(beatles_dir, 'masks/')
mask_name = os.path.join(dir_mask, 'MNI152_T1_2mm_brain_mask.nii.gz')

print(pink+ 'loading mask')
brain_nii = nib.load(mask_name)
brain_mask = io.load_boolean_mask(mask_name)
template_header = brain_nii.header
print('loaded template successfully.')

# Get the list of nonzero voxel coordinates
coords = np.where(brain_mask)

# Save the significant r-value NIfTI file
output_path = results_path+ "/SNS_wholebrain_ISC_FDRcorrected_r_%s_%s_raw.nii.gz" %(kperm, alphalevelname)


if not os.path.exists(output_path):

    print('my shape is:')
    print(thresh_r_map.shape)
    isc_vol = np.zeros(brain_nii.shape)

    print("# of non-zeros in my volume to save:")
    non_zero_count = np.count_nonzero(isc_vol)
    print(non_zero_count)


    # Save the ISC data as a volume
    isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)
    print(pink+ "saving nifti")
    nib.save(isc_nifti,output_path)



print(pink + "######\n Significant FDR corrected r-value NIfTI file FOR SNS saved successfully!\n #####")


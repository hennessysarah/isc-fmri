

### Beatles ISC script...
## for Sarah Hennessy beatles ISC project 

# this is adapted from the brainiak.org tutorial (section 10-isc)
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import glob
import time
from copy import deepcopy
import numpy as np
import pandas as pd

from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import nibabel as nib

from brainiak import image, io
from brainiak.isc import isc, isfc, permutation_isc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



sectionColor = "\033[94m"
sectionColor2 = "\033[96m"
groupColor = "\033[90m"
mainColor = "\033[92m"

pink = '\033[95m'
yellow = '\033[93m'
red = '\033[91m'



# ---------------------
#  Set up
# ---------------------


# Set up experiment metadata
beatles_dir = "/ISC/data/"
results_path = "/ISC/results/OA_fam_groupwise/"
# from utils import beatles_dir, results_path
print(yellow + 'Data directory is: %s' % beatles_dir)


dir_mask = os.path.join(beatles_dir, 'masks/')

mask_name = os.path.join(dir_mask, 'MNI152_T1_2mm_brain_mask.nii.gz')


all_task_names = ['beatles']

#this is really for if you have multiple tasks.. but thats ok
group_assignment_dict = {task_name: i for i, task_name in enumerate(all_task_names)}


# Where do you want to store the data
dir_out = results_path + 'isc/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
    print(pink+'Dir %s created ' % dir_out)


subjectDir = "%s/subs/" %(beatles_dir)
groupcsv = results_path + "nostalgia_ratings_final.csv"
groupcsv = pd.read_csv(groupcsv)
mylist = list(groupcsv['id'])
subjectList = [elem for elem in os.listdir(subjectDir) if "." not in elem and elem in mylist]

subjectList.sort()
print(subjectList)


n_subjs_total = len(subjectList)
print("there are this many subjects")
print(n_subjs_total)

# Reduce the number of subjects per condition to make this notebook faster
upper_limit_n_subjs = n_subjs_total

def get_file_names(data_dir_, verbose = False):
    """
    Get all the participant file names

    Parameters
    ----------
    data_dir_ [str]: the data root dir
    task_name_ [str]: the name of the task

    Return
    ----------
    fnames_ [list]: file names for all subjs
    """
    c_ = 0
    fnames_ = []
    # Collect all file names
    for subjnum in range(0, n_subjs_total):
        subj = subjectList[subjnum]

        #find the functional nifti that has been denoised with ICA and transformed to standard space
        fname = os.path.join(
            data_dir_, 'subs/%s/filtered_func_data_clean_standard_cut.nii.gz' % (subj))

           
        # If the file exists
        if os.path.exists(fname):

            # Add to the list of file names
            fnames_.append(fname)
            if verbose:
                print(yellow+ fname)
            c_+= 1
            if c_ >= upper_limit_n_subjs:
                break
    return fnames_


task_name_ = "beatles"
get_file_names(beatles_dir)


print(red + "##### LOADING BRAIN TEMPLATE #####")

# Load the brain mask
brain_mask = io.load_boolean_mask(mask_name)

# Get the list of nonzero voxel coordinates
coords = np.where(brain_mask)

# Load the brain nii image
brain_nii = nib.load(mask_name)

print("brain template loaded")

print(yellow + "####### LOADING BOLD DATA ########")

# ---------------------
#  load the functional data
# ---------------------

fnames = {}
images = {}
masked_images = {}
bold = {}
group_assignment = []
n_subjs = {}

for task_name in all_task_names:
    fnames[task_name] = get_file_names(beatles_dir, task_name)
    images[task_name] = io.load_images(fnames[task_name])
    masked_images[task_name] = image.mask_images(images[task_name], brain_mask)
    # Concatenate all of the masked images across participants
    bold[task_name] = image.MaskedMultiSubjectData.from_masked_images(
        masked_images[task_name], len(fnames[task_name])
    )
    # Convert nans into zeros
    bold[task_name][np.isnan(bold[task_name])] = 0
    # compute the group assignment label
    n_subjs_this_task = np.shape(bold[task_name])[-1]
    group_assignment += list(
        np.repeat(group_assignment_dict[task_name], n_subjs_this_task)
    )
    n_subjs[task_name] = np.shape(bold[task_name])[-1]
    print('Data loaded: {} \t shape: {}' .format(task_name, np.shape(bold[task_name])))



#subj_names = fnames['beatles'].split('/')[-2]

## ISC is the correlation of each voxel's time series for a participant with the corresponding (anatomically aligned) voxel time series in the average of the other participants' brains.
## BrainIAK has functions for computing ISC by feeding in the concatenated participant data.

# ---------------------
# compute and Save ISC
# ---------------------

# run ISC, loop over conditions (lol we only have 1 condition)
print(pink + "####### BEGINNING GROUPWISE ISC #########")


isc_path = f"{results_path}/isc_beatles_n{n_subjs_this_task}_groupwise.pkl"

isc_maps = {}
if not os.path.exists(isc_path):
    print(pink+ "didnt find an existing pickle! \n doing ISC and saving it to pickle now <3")
    for task_name in all_task_names:
        isc_maps[task_name] = isc(bold[task_name], pairwise=False)
        print(mainColor +'Shape of %s condition:' % task_name, np.shape(isc_maps[task_name]))
        with open(isc_path, 'wb') as f:
            pickle.dump(isc_maps[task_name], f)
            print(pink+'saved to pickle!')
else:
    print(red + "I already made a pkl file for this ISC! (%s) \n go look at it!") %(isc_path)

print(yellow+ "#####################")
print("great job bestie. go check out your pickle here: \n %s" %(isc_path))
print("#####################")



# The output of ISC is:
# a voxel by participant matrix (showing the result of each individual with the group).

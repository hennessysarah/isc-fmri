

### Beatles ISC script...
# specifically for ROI-based analysis
# constraining ISC now to look at specific ROIs 
# rois are spherical masks
# will cycle through each ROI and save map separately

# this is from the brainiak.org tutorial (10-isc)

#
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
from tqdm import tqdm



mainColor = "\033[92m"
red = '\033[91m'
pink = '\033[95m'
yellow = '\033[93m'
blue = '\033[94m'
cyan = '\033[96m'



# ---------------------
#  Set up
# ---------------------


# Set up experiment metadata
beatles_dir = "/ISC/data/"
results_path = "ISC/results/OA_fam_groupwise/ROI/"

### Because i already masked my subjects!!! (check this)
masked_subs_path = "/ISC/results/groupwise/ROI/masked_subs"

print(yellow + 'Data directory is: %s' % beatles_dir)


dir_mask = os.path.join(beatles_dir, 'masks/')
mask_name = os.path.join(dir_mask, 'MNI152_T1_2mm_brain_mask.nii.gz')

roi_folder = "//roi_masks/2023_final"

roi_list = [elem for elem in os.listdir(roi_folder) if elem.endswith('sphere.nii.gz') and not elem.startswith('L_AG')]

roi_list.sort()

print(pink + "You are doing an ROI-based ISC. \n Your ROI masks are: \n")
print(roi_list)


all_task_names = ['beatles']


subjectDir = "%s/subs/" %(beatles_dir)
groupcsv = results_path + "nostalgia_ratings_final.csv"
groupcsv = pd.read_csv(groupcsv)
mylist = list(groupcsv['id'])
subjectList = [elem for elem in os.listdir(subjectDir) if "." not in elem and elem in mylist]

subjectList.sort()
print(subjectList)



n_subjs_total = len(subjectList)


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
    ok = []
    c_ = 0
    fnames_ = []
    # Collect all file names
    print(cyan + "Collecting file names..\n")
    for subjnum in tqdm(range(0, n_subjs_total)):
        subj = subjectList[subjnum]

        #find the functional nifti that has been denoised with ICA and transformed to standard space
        fname = os.path.join(
            data_dir_, 'subs/%s/filtered_func_data_clean_standard_cut.nii.gz' % (subj))
       # print(fname)

        # If the file exists
        if os.path.exists(fname):

            # Add to the list of file names
            fnames_.append(fname)
            if verbose:
                #print(yellow+ fname)
                ok = 'ok' #idk to satsify the thing..

            c_+= 1
            if c_ >= upper_limit_n_subjs:
                break
    return fnames_


#

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
    n_subjs_this_task = np.shape(bold[task_name])[-1]
    n_subjs[task_name] = np.shape(bold[task_name])[-1]
    print('Data loaded: {} \t shape: {}' .format(task_name, np.shape(bold[task_name])))


###########
##### GET ROI MASKERS
###########

print(yellow + "Collecting ROI masks...")
# Collect all ROIs
all_roi_names = []
all_roi_nii = {}
all_roi_masker = {}

for roi in roi_list:
    # Compute ROI name
 
    roi_name = roi[:-14]
    print(mainColor + "ROI: %s" %(roi_name))
    all_roi_names.append(roi_name)

    # Load roi nii file
    roi_path = os.path.join(roi_folder, roi)
    roi_nii = nib.load(roi_path)
    all_roi_nii[roi_name] = roi_nii

   
    mask_img= roi_nii
    mask_data = mask_img.get_fdata()  # Get the mask data as a NumPy array
    binary_mask_data = np.where(mask_data > 0, 1, 0)  # Thresholding to convert to binary

    # Create a new NIfTI image with the binary mask data
    binary_mask_img = nib.Nifti1Image(binary_mask_data, mask_img.affine)

    # Make roi maskers
    all_roi_masker[roi_name] = NiftiMasker(mask_img=binary_mask_img)
  
print("loaded all masks\n")



def load_roi_data(roi_name,fnames,task_name): 
    # Pick a roi masker
    roi_path = roi_folder + "/" + roi_name + "_sphere.nii.gz"
    print(blue+ "LOADING ROI DATA: %s" %(roi_name))

    roi_masker = all_roi_masker[roi_name]    
    
    # Preallocate 
    bold_roi = {task_name:[] for i, task_name in enumerate(all_task_names)}
    
    # Gather data 
    for task_name in all_task_names:

        output_file_path = results_path + "bold_roi/bold_roi_%s.pkl" %(roi_name)
        if not os.path.exists(output_file_path):
            print("no existing file detected... beginning now.")
            for subj_id in tqdm(range(n_subjs[task_name])):

                subname = subjectList[subj_id]
                #print(subname)
                # check if it already exists...
                roi_outfile = masked_subs_path + "/%s_%s.pkl" %(subname, roi_name)
               # roi_outfile = results_path + "masked_subs/%s_%s.pkl" %(subname, roi_name)
                
                if not os.path.exists(roi_outfile):
                    # Get the data for task t, subject s 
                    print("making masked data...")

                    #load in participants bold
                    nii_t_s = nib.load(fnames[task_name][subj_id])
                
                    # apply the mask to the data and extract the relevant voxels within the mask.
                    masked_toroi_all = roi_masker.fit_transform(nii_t_s)

                    print("SHAPE should be 286xvoxelsx1")
                    print(masked_toroi_all.shape)


            

                          # now take an average at each time point so simplify the model

                    masked_toroi = np.mean(masked_toroi_all, axis=1, keepdims=True) 
                    print("SHAPE should be 286x1x1")
                    print(masked_toroi.shape)


                    #Save the file
                    with open(roi_outfile, 'wb') as f:
                        pickle.dump(masked_toroi, f)

                    bold_roi[task_name].append(masked_toroi)
                    print("saving...", roi_outfile)

                else:
                    # load it
                    print("already made! opening now..")

                    with open(roi_outfile, 'rb') as f:
                        masked_toroi = pickle.load(f)
                   
                    bold_roi[task_name].append(masked_toroi)
                    
                
            # Reformat the data to std form 
            bold_roi[task_name] = np.transpose(np.array(bold_roi[task_name]), [1,2,0])
            #save it as  pkl

            with open(output_file_path, 'wb') as f:
                pickle.dump(bold_roi[task_name], f)
        else: 
            print("existing file found! loading now.")
            with open(output_file_path, 'rb') as f:
                bold_roi[task_name] = pickle.load(f)

        # save as a nifti
    return bold_roi


# # ---------------------
# # compute and Save ISC
# # ---------------------

print(pink + "############# COMPUTE ISC ################# \n ")

for roi in tqdm(roi_list):

    roi_name = roi[:-14]


    print(cyan + "####### BEGINNING GROUPWISE ISC FOR: %s #########" %(roi_name))


    isc_path = f"{results_path}{roi_name}_isc_beatles_n{n_subjs_this_task}_groupwise.pkl"
    
    # run ISC
    isc_maps = {}

    if not os.path.exists(isc_path):
        print("Beginning.")


        print(pink+ "didnt find an existing pickle! \n doing ISC and saving it to pickle now <3")
        for task_name in all_task_names:
            bold_roi = load_roi_data(roi_name, fnames, task_name)
            isc_maps[task_name] = isc(bold_roi[task_name], pairwise=False)
            
       
            with open(isc_path, 'wb') as f:
                pickle.dump(isc_maps[task_name], f)
                print(blue+'saved to pickle!')

        print(yellow+ "#####################")
        print("DONE with %s. \n pkl saved to: \n %s" %(roi_name, results_path))
        print("#####################")
    
    else:
        print(red + "I already made a pkl file for this ISC! (%s) \n  moving on. \n" %(isc_path))

        


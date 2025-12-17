
## For ISC analysis
# after ROI analysis, goes through each PKL
# and retrieves the mean & se of correlation values for each ROI


import numpy as np
import pickle
import os
import sys
import pandas as pd
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")


pink = '\033[95m'
yellow = '\033[93m'
blue = '\033[94m'
cyan = '\033[96m'


results_path = "/ISC/results/OA_fam_groupwise/ROI/"
roi_list = [elem for elem in os.listdir(results_path) if elem.endswith('groupwise.pkl')]
roi_list.sort()
print(pink+ "YOUR ROI LIST IS: \n", roi_list)
output_path = results_path + "ROI_corvalues_all.csv"
colnames = ['ROI', 'cor_mean', 'cor_se']
newdf = pd.DataFrame(columns=colnames, index = range(len(roi_list)) )

def retrieve_cors(roi_list, results_path, newdf, output_path):
    i = -1
    for roi in roi_list:
        i += 1
        roi_name = roi[:-29]
        print(cyan + "Processing %s" % roi_name)

        newdf.loc[i, 'ROI'] = roi_name
        roi_pkl_path = results_path + "/" + roi

        with open(roi_pkl_path, 'rb') as f:
            roi_pkl = pickle.load(f)

        cor_mean = np.nanmean(roi_pkl)
        newdf.loc[i, 'cor_mean'] = cor_mean

        # Calculate the standard deviation


        std_dev = np.nanstd(roi_pkl)
        # Calculate the standard error
        sample_size = np.sum(~np.isnan(roi_pkl))
        # Calculate the standard error

        cor_se = std_dev / np.sqrt(sample_size)
        newdf.loc[i, 'cor_se'] = cor_se

        print(yellow + "%s: \n mean = %f, se = %f" % (roi_name, cor_mean, cor_se))
    return newdf



if not os.path.exists(output_path):
    # Data processing when the output file doesn't exist
    retrieve_cors(roi_list = roi_list, results_path = results_path, newdf = newdf, output_path = output_path)

else:
    overwrite = input(yellow + "This output file exists! \nDo you want to overwrite? y or n: \n")
    if overwrite == "y":
        print(pink + "Ok, overwriting existing csv.")
        # Data processing when the user chooses to overwrite
        retrieve_cors(roi_list=roi_list, results_path=results_path, newdf=newdf, output_path=output_path)
        newdf.to_csv(output_path, index=False)
        print(cyan + "file saved: %s" % output_path)
    else:
        print("Ok, quitting now.")
        sys.exit()

# Save the DataFrame to CSV
newdf.to_csv(output_path, index=False)

# computes permutation test to assess whether the whole brain ISC is significant

import numpy as np
import matplotlib.pyplot as plt
from brainiak.isc import permutation_isc


pink = '\033[95m'
yellow = '\033[93m'


n_permutations = 10000
summary_statistic = 'median'
# median might be better
# However, with the typical range of values of groupwise ISCs,
# the effects of this transformation are relatively small reaching ~10%
# at the higher end of the scale of r=0.5.
# More recently, it has been suggested that computing the median,
#particularly when using the groupwise approach,
# provides a more accurate summary of the correlation values (Chen et al., 2016).



results_dir = "/ISC/results/OA_fam_groupwise"
picklefile = "/ISC/results/OA_fam_groupwise/isc_beatles_n27_groupwise.pkl"
import pickle
print("about to load pickle file!")
with open(picklefile, 'rb') as f:
    isc_maps = pickle.load(f)

print(pink+ "loaded successfully")


print(yellow + "##### STARTING PERMUTATION TESTING. ##########")
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

ks = int((n_permutations)/1000)
kperm = str(ks) + "kperm"


# Save observed and p-values to pickle files

observed_pickle_path = results_dir + '/permuted_ISC_observed_%s.pkl' %(kperm)
p_value_pickle_path = results_dir + '/permuted_ISC_p_%s.pkl' %(kperm)
permuted_ISC_distribution_pickle_path = results_dir + '/permuted_ISC_distribution_%s.pkl' %(kperm)

print(yellow + "saving files......")
with open(observed_pickle_path, 'wb') as f:
    pickle.dump(observed, f)

with open(p_value_pickle_path, 'wb') as f:
    pickle.dump(p, f)

with open(permuted_ISC_distribution_pickle_path, 'wb') as f:
    pickle.dump(distribution, f)

print(pink+ "Done saving! great job, you little isc queen.")

import numpy as np
import pandas as pd
import os
from scipy.stats import bootstrap

dirname = os.path.dirname(__file__)

#See end of file for the pulse and chase data summary statistics

#############################################################################################################################
#                                                                                                                           #
#                                                       PULSE - CHASE                                                       #
#                                                           DATA                                                            #
#                                                        EXTRACTION                                                         #
#                                                                                                                           #
#############################################################################################################################

############ --------------------------------   PULSE EXPERIMENT DATA EXTRACTION ------------------------------ ############################ 

###---------------------- Extracting data for experiment 1 --------------------------###
excel_file_1hr= pd.ExcelFile(os.path.join(dirname, 'pulse_data/assay1/06022024 No FBS Fibroblasts 1 hr.xlsx'))
excel_file_3hr= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay1/06022024 No FBS Fibroblasts 3 hr.xlsx"))
excel_file_7hr= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay1/06022024 No FBS Fibroblasts 7 hr.xlsx"))
excel_file_24hr= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay1/06022024 No FBS Fibroblasts 24 hr.xlsx"))

#Extracting into panda dataframes
mito_length_1hr = pd.read_excel(excel_file_1hr, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 38)) #mtvolume
mito_length_3hr = pd.read_excel(excel_file_3hr, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 39))
mito_length_7hr = pd.read_excel(excel_file_7hr, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 28))
mito_length_24hr = pd.read_excel(excel_file_24hr, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 31))
edu_number_1hr = pd.read_excel(excel_file_1hr, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 38)) #edu number
edu_number_3hr = pd.read_excel(excel_file_3hr, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 39))
edu_number_7hr = pd.read_excel(excel_file_7hr, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 28))
edu_number_24hr = pd.read_excel(excel_file_24hr, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 31)) 
dna_number_1hr = pd.read_excel(excel_file_1hr, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 38)) #dna number
dna_number_3hr = pd.read_excel(excel_file_3hr, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 39))
dna_number_7hr = pd.read_excel(excel_file_7hr, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 28))
dna_number_24hr = pd.read_excel(excel_file_24hr, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 31))
cell_volume_1hr = pd.read_excel(excel_file_1hr, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 38)) #cell volume
cell_volume_3hr = pd.read_excel(excel_file_3hr, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 39))
cell_volume_7hr = pd.read_excel(excel_file_7hr, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 28))
cell_volume_24hr = pd.read_excel(excel_file_24hr, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 31))

#Converting into numpy arrays
mito_length_1hr = np.array(mito_length_1hr["total volume"]) #mtvolume
mito_length_3hr = np.array(mito_length_3hr["total volume"])
mito_length_7hr = np.array(mito_length_7hr["total volume"])
mito_length_24hr = np.array(mito_length_24hr["total volume"])
edu_number_1hr = np.array(edu_number_1hr['count.1']) #edu number
edu_number_3hr = np.array(edu_number_3hr['count.1'])
edu_number_7hr = np.array(edu_number_7hr['count.1'])
edu_number_24hr = np.array(edu_number_24hr['count.1'])
dna_number_1hr = np.array(dna_number_1hr["count.2"]) #dna number
dna_number_3hr = np.array(dna_number_3hr["count.2"])
dna_number_7hr = np.array(dna_number_7hr["count.2"])
dna_number_24hr = np.array(dna_number_24hr["count.2"])
cell_volume_1hr = np.array(cell_volume_1hr["Cell volume"]) #cell volume
cell_volume_3hr = np.array(cell_volume_3hr["Cell volume"])
cell_volume_7hr = np.array(cell_volume_7hr["Cell volume"])
cell_volume_24hr = np.array(cell_volume_24hr["Cell volume"])

###---------------------- Extracting data for experiment 2 --------------------------###
excel_file_1hr2= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay2/25012024 No FBS Fibroblasts 1 hr.xlsx"))
excel_file_3hr2= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay2/25012024 No FBS Fibroblasts 3 hr.xlsx"))
excel_file_7hr2= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay2/25012024 No FBS Fibroblasts 7 hr.xlsx"))
excel_file_24hr2= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay2/25012024 No FBS Fibroblasts 24 hr.xlsx"))

#Extracting into panda dataframes
mito_length_1hr2 = pd.read_excel(excel_file_1hr2, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 28)) #mtvolume
mito_length_3hr2 = pd.read_excel(excel_file_3hr2, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 37))
mito_length_7hr2 = pd.read_excel(excel_file_7hr2, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 36))
mito_length_24hr2 = pd.read_excel(excel_file_24hr2, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 50))
edu_number_1hr2 = pd.read_excel(excel_file_1hr2, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 28)) #edu number
edu_number_3hr2 = pd.read_excel(excel_file_3hr2, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 37))
edu_number_7hr2 = pd.read_excel(excel_file_7hr2, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 36))
edu_number_24hr2 = pd.read_excel(excel_file_24hr2, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 50))
dna_number_1hr2 = pd.read_excel(excel_file_1hr2, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 28)) #dna number
dna_number_3hr2 = pd.read_excel(excel_file_3hr2, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 37))
dna_number_7hr2 = pd.read_excel(excel_file_7hr2, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 36))
dna_number_24hr2 = pd.read_excel(excel_file_24hr2, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 50))
cell_volume_1hr2 = pd.read_excel(excel_file_1hr2, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 28)) #cell volume
cell_volume_3hr2 = pd.read_excel(excel_file_3hr2, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 37))
cell_volume_7hr2 = pd.read_excel(excel_file_7hr2, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 36))
cell_volume_24hr2 = pd.read_excel(excel_file_24hr2, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 50))

#Converting into arrays
mito_length_1hr2 = np.array(mito_length_1hr2["total volume"]) #mtvolume
mito_length_3hr2 = np.array(mito_length_3hr2["total volume"])
mito_length_7hr2 = np.array(mito_length_7hr2["total volume"])
mito_length_24hr2 = np.array(mito_length_24hr2["total volume"])
edu_number_1hr2 = np.array(edu_number_1hr2['count.1']) #edu number
edu_number_3hr2 = np.array(edu_number_3hr2['count.1'])
edu_number_7hr2 = np.array(edu_number_7hr2['count.1'])
edu_number_24hr2 = np.array(edu_number_24hr2['count.1'])
dna_number_1hr2 = np.array(dna_number_1hr2["count.2"]) #dna number
dna_number_3hr2 = np.array(dna_number_3hr2["count.2"])
dna_number_7hr2 = np.array(dna_number_7hr2["count.2"])
dna_number_24hr2 = np.array(dna_number_24hr2["count.2"])
cell_volume_1hr2 = np.array(cell_volume_1hr2["Cell volume"]) #cell volume
cell_volume_3hr2 = np.array(cell_volume_3hr2["Cell volume"])
cell_volume_7hr2 = np.array(cell_volume_7hr2["Cell volume"])
cell_volume_24hr2 = np.array(cell_volume_24hr2["Cell volume"])

###---------------------- Extracting data for experiment 1 --------------------------###
excel_file_1hr3= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay3/01022024 No FBS Fibroblasts 1 hr.xlsx"))
excel_file_3hr3= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay3/01022024 No FBS Fibroblasts 3 hr.xlsx"))
excel_file_7hr3= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay3/01022024 No FBS Fibroblasts 7 hr.xlsx"))
excel_file_24hr3= pd.ExcelFile(os.path.join(dirname, "pulse_data/assay3/01022024 No FBS Fibroblasts 24 hr.xlsx"))

#Extracting into panda dataframes
mito_length_1hr3 = pd.read_excel(excel_file_1hr3, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 33)) #mtvolume
mito_length_3hr3 = pd.read_excel(excel_file_3hr3, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 38))
mito_length_7hr3 = pd.read_excel(excel_file_7hr3, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 38))
mito_length_24hr3 = pd.read_excel(excel_file_24hr3, usecols = 'E:K', skiprows = lambda x: (x < 4) or (x >= 46))
edu_number_1hr3 = pd.read_excel(excel_file_1hr3, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 33)) #edu number
edu_number_3hr3 = pd.read_excel(excel_file_3hr3, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 38))
edu_number_7hr3 = pd.read_excel(excel_file_7hr3, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 38))
edu_number_24hr3 = pd.read_excel(excel_file_24hr3, usecols = 'M:S', skiprows = lambda x: (x < 4) or (x >= 46))
dna_number_1hr3 = pd.read_excel(excel_file_1hr3, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 33)) #dna number
dna_number_3hr3 = pd.read_excel(excel_file_3hr3, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 38))
dna_number_7hr3 = pd.read_excel(excel_file_7hr3, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 38))
dna_number_24hr3 = pd.read_excel(excel_file_24hr3, usecols = 'U:AA', skiprows = lambda x: (x < 4) or (x >= 46))
cell_volume_1hr3 = pd.read_excel(excel_file_1hr3, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 33))
cell_volume_3hr3 = pd.read_excel(excel_file_3hr3, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 38))
cell_volume_7hr3 = pd.read_excel(excel_file_7hr3, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 38))
cell_volume_24hr3 = pd.read_excel(excel_file_24hr3, usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 46))

#Converting into arrays
mito_length_1hr3 = np.array(mito_length_1hr3["total volume"]) #mtvolume
mito_length_3hr3 = np.array(mito_length_3hr3["total volume"])
mito_length_7hr3 = np.array(mito_length_7hr3["total volume"])
mito_length_24hr3 = np.array(mito_length_24hr3["total volume"])
edu_number_1hr3 = np.array(edu_number_1hr3['count.1']) #edu number
edu_number_3hr3 = np.array(edu_number_3hr3['count.1'])
edu_number_7hr3 = np.array(edu_number_7hr3['count.1'])
edu_number_24hr3 = np.array(edu_number_24hr3['count.1'])
dna_number_1hr3 = np.array(dna_number_1hr3["count.2"]) #dna number
dna_number_3hr3 = np.array(dna_number_3hr3["count.2"])
dna_number_7hr3 = np.array(dna_number_7hr3["count.2"])
dna_number_24hr3 = np.array(dna_number_24hr3["count.2"])
cell_volume_1hr3 = np.array(cell_volume_1hr3["Cell volume"]) #cell volume
cell_volume_3hr3 = np.array(cell_volume_3hr3["Cell volume"])
cell_volume_7hr3 = np.array(cell_volume_7hr3["Cell volume"])
cell_volume_24hr3 = np.array(cell_volume_24hr3["Cell volume"])

###---------------------- Constructing pulse training data (assays 1 + 2 ) --------------------------###

training_mito_length_1hr = np.concatenate([mito_length_1hr, mito_length_1hr2])
training_mito_length_3hr = np.concatenate([mito_length_3hr, mito_length_3hr2])
training_mito_length_7hr = np.concatenate([mito_length_7hr, mito_length_7hr2])
training_mito_length_24hr = np.concatenate([mito_length_24hr, mito_length_24hr2])

training_edu_number_1hr = np.concatenate([edu_number_1hr, edu_number_1hr2])
training_edu_number_3hr = np.concatenate([edu_number_3hr, edu_number_3hr2])
training_edu_number_7hr = np.concatenate([edu_number_7hr, edu_number_7hr2])
training_edu_number_24hr = np.concatenate([edu_number_24hr, edu_number_24hr2])

training_dna_number_1hr = np.concatenate([dna_number_1hr, dna_number_1hr2])
training_dna_number_3hr = np.concatenate([dna_number_3hr, dna_number_3hr2])
training_dna_number_7hr = np.concatenate([dna_number_7hr, dna_number_7hr2])
training_dna_number_24hr = np.concatenate([dna_number_24hr, dna_number_24hr2])

training_cell_volumes = np.concatenate([cell_volume_1hr, cell_volume_1hr2, cell_volume_3hr, cell_volume_3hr2, 
                                   cell_volume_7hr, cell_volume_7hr2, cell_volume_24hr, cell_volume_24hr2])
training_mito_lengths = np.concatenate([mito_length_1hr, mito_length_1hr2, mito_length_3hr, mito_length_3hr2, 
                                   mito_length_7hr, mito_length_7hr2, mito_length_24hr, mito_length_24hr2])

training_dna_numbers = np.concatenate([dna_number_1hr, dna_number_1hr2, dna_number_3hr, dna_number_3hr2, 
                                  dna_number_7hr, dna_number_7hr2, dna_number_24hr, dna_number_24hr2])
training_edu_numbers = np.concatenate([edu_number_1hr, edu_number_1hr2, edu_number_3hr, edu_number_3hr2, 
                                  edu_number_7hr, edu_number_7hr2, edu_number_24hr, edu_number_24hr2])

ind1 = np.ones(len(cell_volume_1hr) + len(cell_volume_1hr2))
ind3 = 3*np.ones(len(cell_volume_3hr) + len(cell_volume_3hr2))
ind7 = 7*np.ones(len(cell_volume_7hr) + len(cell_volume_7hr2))
ind24 = 24*np.ones(len(cell_volume_24hr) + len(cell_volume_24hr2))

training_time_indicator = np.concatenate([ind1, ind3, ind7, ind24])

#------------------------Constructing pulse validation data (assay 3)-----------------------#############
validation_mito_length_1hr = mito_length_1hr3
validation_mito_length_3hr = mito_length_3hr3
validation_mito_length_7hr = mito_length_7hr3
validation_mito_length_24hr = mito_length_24hr3

validation_edu_number_1hr = edu_number_1hr3
validation_edu_number_3hr = edu_number_3hr3
validation_edu_number_7hr = edu_number_7hr3
validation_edu_number_24hr = edu_number_24hr3

validation_dna_number_1hr = dna_number_1hr3
validation_dna_number_3hr = dna_number_3hr3
validation_dna_number_7hr = dna_number_7hr3
validation_dna_number_24hr = dna_number_24hr3

validation_cell_volumes = np.concatenate([cell_volume_1hr3, cell_volume_3hr3, cell_volume_7hr3, cell_volume_24hr3])
validation_mito_lengths = np.concatenate([mito_length_1hr3, mito_length_3hr3, mito_length_7hr3, mito_length_24hr3])

validation_dna_numbers = np.concatenate([dna_number_1hr3, dna_number_3hr3, dna_number_7hr3, dna_number_24hr3])
validation_edu_numbers = np.concatenate([edu_number_1hr3, edu_number_3hr3, edu_number_7hr3, edu_number_24hr3])

vind1 = np.ones(len(cell_volume_1hr3))
vind3 = 3*np.ones(len(cell_volume_3hr3))
vind7 = 7*np.ones(len(cell_volume_7hr3))
vind24 = 24*np.ones(len(cell_volume_24hr3))

validation_time_indicator = np.concatenate([vind1, vind3, vind7, vind24])

###---------------------- Constructing all pulse data (assays 1 + 2 + 3) for posterior predictive checks --------------------------###
all_assays_cell_volumes = np.concatenate([cell_volume_1hr, cell_volume_1hr2, cell_volume_1hr3, cell_volume_3hr, cell_volume_3hr2, cell_volume_3hr3,
                                   cell_volume_7hr, cell_volume_7hr2, cell_volume_7hr3, cell_volume_24hr, cell_volume_24hr2, cell_volume_24hr3])
all_assays_mito_lengths = np.concatenate([mito_length_1hr, mito_length_1hr2, mito_length_1hr3, mito_length_3hr, mito_length_3hr2, mito_length_3hr3,
                                   mito_length_7hr, mito_length_7hr2, mito_length_7hr3, mito_length_24hr, mito_length_24hr2, mito_length_24hr3])

all_assays_dna_numbers = np.concatenate([dna_number_1hr, dna_number_1hr2,dna_number_1hr3, dna_number_3hr, dna_number_3hr2, dna_number_3hr3,
                                  dna_number_7hr, dna_number_7hr2, dna_number_7hr3, dna_number_24hr, dna_number_24hr2, dna_number_24hr3])
all_assays_edu_numbers = np.concatenate([edu_number_1hr, edu_number_1hr2, edu_number_1hr3, edu_number_3hr, edu_number_3hr2, edu_number_3hr3,
                                  edu_number_7hr, edu_number_7hr2, edu_number_7hr3, edu_number_24hr, edu_number_24hr2, edu_number_24hr3])

aind1 = np.ones(len(cell_volume_1hr) + len(cell_volume_1hr2) + len(cell_volume_1hr3))
aind3 = 3*np.ones(len(cell_volume_3hr) + len(cell_volume_3hr2) + len(cell_volume_3hr3))
aind7 = 7*np.ones(len(cell_volume_7hr) + len(cell_volume_7hr2) + len(cell_volume_7hr3))
aind24 = 24*np.ones(len(cell_volume_24hr) + len(cell_volume_24hr2) + len(cell_volume_24hr3))

all_assays_time_indicator = np.concatenate([aind1, aind3, aind7, aind24])

all_assays_mito_length_1hr = np.concatenate([mito_length_1hr, mito_length_1hr2, mito_length_1hr3])
all_assays_mito_length_3hr = np.concatenate([mito_length_3hr, mito_length_3hr2, mito_length_3hr3])
all_assays_mito_length_7hr = np.concatenate([mito_length_7hr, mito_length_7hr2, mito_length_7hr3])
all_assays_mito_length_24hr = np.concatenate([mito_length_24hr, mito_length_24hr2, mito_length_24hr3])

all_assays_edu_number_1hr = np.concatenate([edu_number_1hr, edu_number_1hr2, edu_number_1hr3])
all_assays_edu_number_3hr = np.concatenate([edu_number_3hr, edu_number_3hr2, edu_number_3hr3])
all_assays_edu_number_7hr = np.concatenate([edu_number_7hr, edu_number_7hr2, edu_number_7hr3])
all_assays_edu_number_24hr = np.concatenate([edu_number_24hr, edu_number_24hr2, edu_number_24hr3])

all_assays_dna_number_1hr = np.concatenate([dna_number_1hr, dna_number_1hr2, dna_number_1hr3])
all_assays_dna_number_3hr = np.concatenate([dna_number_3hr, dna_number_3hr2, dna_number_3hr3])
all_assays_dna_number_7hr = np.concatenate([dna_number_7hr, dna_number_7hr2, dna_number_7hr3])
all_assays_dna_number_24hr = np.concatenate([dna_number_24hr, dna_number_24hr2, dna_number_24hr3])

############ --------------------------------   CHASE EXPERIMENT DATA EXTRACTION ------------------------------ ############################ 

excel_file_chase1 = pd.ExcelFile(os.path.join(dirname, "chase_data/Assay 1 Pulse Chase.xlsx"))
excel_file_chase2 = pd.ExcelFile(os.path.join(dirname, "chase_data/Assay 2 Pulse Chase.xlsx"))
excel_file_chase3 = pd.ExcelFile(os.path.join(dirname, "chase_data/Assay 3 Pulse Chase.xlsx"))

###---------------------- Extracting data for assay 1 directly into arrays--------------------------###
mito_length_0dy1 = np.array(pd.read_excel(excel_file_chase1,"0 hr", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 30))["total volume"])
cell_volume_0dy1 = np.array(pd.read_excel(excel_file_chase1,"0 hr", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 30))["Cell volume"])
dna_number_0dy1 = np.array(pd.read_excel(excel_file_chase1,"0 hr", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 30))["count.2"])
edu_number_0dy1 = np.array(pd.read_excel(excel_file_chase1,"0 hr", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 30))["count.1"])
mito_length_1dy1 = np.array(pd.read_excel(excel_file_chase1,"24 hr", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 25))["total volume"])
cell_volume_1dy1 = np.array(pd.read_excel(excel_file_chase1,"24 hr", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 25))["Cell volume"])
dna_number_1dy1 = np.array(pd.read_excel(excel_file_chase1,"24 hr", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 25))["count.2"])
edu_number_1dy1 = np.array(pd.read_excel(excel_file_chase1,"24 hr", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 25))["count.1"])
mito_length_2dy1 = np.array(pd.read_excel(excel_file_chase1,"48 hrs", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 38))["total volume"])
cell_volume_2dy1 = np.array(pd.read_excel(excel_file_chase1,"48 hrs", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 38))["Cell volume"])
dna_number_2dy1 = np.array(pd.read_excel(excel_file_chase1,"48 hrs", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 38))["count.2"])
edu_number_2dy1 = np.array(pd.read_excel(excel_file_chase1,"48 hrs", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 38))["count.1"])
mito_length_4dy1 = np.array(pd.read_excel(excel_file_chase1,"96 hrs", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 27))["total volume"])
cell_volume_4dy1 = np.array(pd.read_excel(excel_file_chase1,"96 hrs", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 27))["Cell volume"])
dna_number_4dy1 = np.array(pd.read_excel(excel_file_chase1,"96 hrs", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 27))["count.2"])
edu_number_4dy1 = np.array(pd.read_excel(excel_file_chase1,"96 hrs", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 27))["count.1"])

###---------------------- Extracting data for assay 2 directly into arrays--------------------------###
mito_length_0dy2 = np.array(pd.read_excel(excel_file_chase2,"0 hr", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 19))["total volume"])
cell_volume_0dy2 = np.array(pd.read_excel(excel_file_chase2,"0 hr", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 19))["Cell volume"])
dna_number_0dy2 = np.array(pd.read_excel(excel_file_chase2,"0 hr", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 19))["count.2"])
edu_number_0dy2 = np.array(pd.read_excel(excel_file_chase2,"0 hr", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 19))["count.1"])
mito_length_1dy2 = np.array(pd.read_excel(excel_file_chase2,"24 hr", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 29))["total volume"])
cell_volume_1dy2 = np.array(pd.read_excel(excel_file_chase2,"24 hr", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 29))["Cell volume"])
dna_number_1dy2 = np.array(pd.read_excel(excel_file_chase2,"24 hr", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 29))["count.2"])
edu_number_1dy2 = np.array(pd.read_excel(excel_file_chase2,"24 hr", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 29))["count.1"])
mito_length_2dy2 = np.array(pd.read_excel(excel_file_chase2,"48 hrs", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 23))["total volume"])
cell_volume_2dy2 = np.array(pd.read_excel(excel_file_chase2,"48 hrs", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 23))["Cell volume"])
dna_number_2dy2 = np.array(pd.read_excel(excel_file_chase2,"48 hrs", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 23))["count.2"])
edu_number_2dy2 = np.array(pd.read_excel(excel_file_chase2,"48 hrs", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 23))["count.1"])
mito_length_4dy2 = np.array(pd.read_excel(excel_file_chase2,"96 hrs", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 30))["total volume"])
cell_volume_4dy2 = np.array(pd.read_excel(excel_file_chase2,"96 hrs", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 30))["Cell volume"])
dna_number_4dy2 = np.array(pd.read_excel(excel_file_chase2,"96 hrs", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 30))["count.2"])
edu_number_4dy2 = np.array(pd.read_excel(excel_file_chase2,"96 hrs", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 30))["count.1"])

###---------------------- Extracting data for assay 3 directly into arrays--------------------------###
mito_length_0dy3 = np.array(pd.read_excel(excel_file_chase3,"0 hr", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 24))["total volume"])
cell_volume_0dy3 = np.array(pd.read_excel(excel_file_chase3,"0 hr", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 24))["Cell volume"])
dna_number_0dy3 = np.array(pd.read_excel(excel_file_chase3,"0 hr", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 24))["count.2"])
edu_number_0dy3 = np.array(pd.read_excel(excel_file_chase3,"0 hr", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 24))["count.1"])
mito_length_1dy3 = np.array(pd.read_excel(excel_file_chase3,"24 hr", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 25))["total volume"])
cell_volume_1dy3 = np.array(pd.read_excel(excel_file_chase3,"24 hr", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 25))["Cell volume"])
dna_number_1dy3 = np.array(pd.read_excel(excel_file_chase3,"24 hr", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 25))["count.2"])
edu_number_1dy3 = np.array(pd.read_excel(excel_file_chase3,"24 hr", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 25))["count.1"])
mito_length_2dy3 = np.array(pd.read_excel(excel_file_chase3,"48 hrs", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 26))["total volume"])
cell_volume_2dy3 = np.array(pd.read_excel(excel_file_chase3,"48 hrs", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 26))["Cell volume"])
dna_number_2dy3 = np.array(pd.read_excel(excel_file_chase3,"48 hrs", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 26))["count.2"])
edu_number_2dy3 = np.array(pd.read_excel(excel_file_chase3,"48 hrs", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 26))["count.1"])
mito_length_4dy3 = np.array(pd.read_excel(excel_file_chase3,"96 hrs", usecols = 'G', skiprows = lambda x: (x < 4) or (x >= 29))["total volume"])
cell_volume_4dy3 = np.array(pd.read_excel(excel_file_chase3,"96 hrs", usecols = 'D', skiprows = lambda x: (x < 4) or (x >= 29))["Cell volume"])
dna_number_4dy3 = np.array(pd.read_excel(excel_file_chase3,"96 hrs", usecols = 'V', skiprows = lambda x: (x < 4) or (x >= 29))["count.2"])
edu_number_4dy3 = np.array(pd.read_excel(excel_file_chase3,"96 hrs", usecols = 'N', skiprows = lambda x: (x < 4) or (x >= 29))["count.1"])

###---------------------- Constructing chase training data (assays 1 + 2 ) --------------------------###
training_mito_lengths_chase = np.concatenate([mito_length_0dy1, mito_length_0dy2, mito_length_1dy1, mito_length_1dy2, 
                                   mito_length_2dy1, mito_length_2dy2, mito_length_4dy1, mito_length_4dy2])

training_cell_volumes_chase = np.concatenate([cell_volume_0dy1, cell_volume_0dy2, cell_volume_1dy1, cell_volume_1dy2, 
                                   cell_volume_2dy1, cell_volume_2dy2, cell_volume_4dy1, cell_volume_4dy2])

ind_chase0 = np.zeros(len(cell_volume_0dy1) + len(cell_volume_0dy2))
ind_chase1 = 24*np.ones(len(cell_volume_1dy1) + len(cell_volume_1dy2))
ind_chase2 = 48*np.ones(len(cell_volume_2dy1) + len(cell_volume_2dy2))
ind_chase4 = 96*np.ones(len(cell_volume_4dy1) + len(cell_volume_4dy2))

training_chase_time_indicator = np.concatenate([ind_chase0, ind_chase1, ind_chase2, ind_chase4])

training_dna_number_0dy = np.concatenate([dna_number_0dy1, dna_number_0dy2])
training_dna_number_1dy = np.concatenate([dna_number_1dy1, dna_number_1dy2])
training_dna_number_2dy = np.concatenate([dna_number_2dy1, dna_number_2dy2])
training_dna_number_4dy = np.concatenate([dna_number_4dy1, dna_number_4dy2])

training_edu_number_0dy = np.concatenate([edu_number_0dy1, edu_number_0dy2])
training_edu_number_1dy = np.concatenate([edu_number_1dy1, edu_number_1dy2])
training_edu_number_2dy = np.concatenate([edu_number_2dy1, edu_number_2dy2])
training_edu_number_4dy = np.concatenate([edu_number_4dy1, edu_number_4dy2])

training_mito_length_0dy = np.concatenate([mito_length_0dy1, mito_length_0dy2])
training_mito_length_1dy = np.concatenate([mito_length_1dy1, mito_length_1dy2])
training_mito_length_2dy = np.concatenate([mito_length_2dy1, mito_length_2dy2])
training_mito_length_4dy = np.concatenate([mito_length_4dy1, mito_length_4dy2])

###---------------------- Constructing chase validation data (assay 3) --------------------------###
validation_mito_lengths_chase = np.concatenate([mito_length_0dy3, mito_length_1dy3, mito_length_2dy3, mito_length_4dy3])

validation_cell_volumes_chase = np.concatenate([cell_volume_0dy3, cell_volume_1dy3, cell_volume_2dy3, cell_volume_4dy3])

vind_chase0 = np.zeros(len(cell_volume_0dy3))
vind_chase1 = 24*np.ones(len(cell_volume_1dy3))
vind_chase2 = 48*np.ones(len(cell_volume_2dy3))
vind_chase4 = 96*np.ones(len(cell_volume_4dy3))

validation_chase_time_indicator = np.concatenate([vind_chase0, vind_chase1, vind_chase2, vind_chase4])

validation_dna_number_0dy = dna_number_0dy3
validation_dna_number_1dy = dna_number_1dy3
validation_dna_number_2dy = dna_number_2dy3
validation_dna_number_4dy = dna_number_4dy3

validation_edu_number_0dy = edu_number_0dy3
validation_edu_number_1dy = edu_number_1dy3
validation_edu_number_2dy = edu_number_2dy3
validation_edu_number_4dy = edu_number_4dy3

validation_mito_length_0dy = mito_length_0dy3
validation_mito_length_1dy = mito_length_1dy3
validation_mito_length_2dy = mito_length_2dy3
validation_mito_length_4dy = mito_length_4dy3

###---------------------- Constructing all chase data (assays 1 + 2 + 3) for posterior predictive checks --------------------------###
all_assays_mito_lengths_chase = np.concatenate([mito_length_0dy1, mito_length_0dy2, mito_length_0dy3, mito_length_1dy1, mito_length_1dy2, mito_length_1dy3,
                                   mito_length_2dy1, mito_length_2dy2, mito_length_2dy3, mito_length_4dy1, mito_length_4dy2, mito_length_4dy3])

all_assays_cell_volumes_chase = np.concatenate([cell_volume_0dy1, cell_volume_0dy2, cell_volume_0dy3, cell_volume_1dy1, cell_volume_1dy2, cell_volume_1dy3,
                                   cell_volume_2dy1, cell_volume_2dy2, cell_volume_2dy3, cell_volume_4dy1, cell_volume_4dy2, cell_volume_4dy3])

aind_chase0 = np.zeros(len(cell_volume_0dy1) + len(cell_volume_0dy2) + len(cell_volume_0dy3))
aind_chase1 = 24*np.ones(len(cell_volume_1dy1) + len(cell_volume_1dy2) + len(cell_volume_1dy3))
aind_chase2 = 48*np.ones(len(cell_volume_2dy1) + len(cell_volume_2dy2) + len(cell_volume_2dy3))
aind_chase4 = 96*np.ones(len(cell_volume_4dy1) + len(cell_volume_4dy2) + len(cell_volume_4dy3))

all_assays_chase_time_indicator = np.concatenate([aind_chase0, aind_chase1, aind_chase2, aind_chase4])

all_assays_dna_number_0dy = np.concatenate([dna_number_0dy1, dna_number_0dy2, dna_number_0dy3])
all_assays_dna_number_1dy = np.concatenate([dna_number_1dy1, dna_number_1dy2, dna_number_1dy3])
all_assays_dna_number_2dy = np.concatenate([dna_number_2dy1, dna_number_2dy2, dna_number_2dy3])
all_assays_dna_number_4dy = np.concatenate([dna_number_4dy1, dna_number_4dy2, dna_number_4dy3])

all_assays_edu_number_0dy = np.concatenate([edu_number_0dy1, edu_number_0dy2, edu_number_0dy3])
all_assays_edu_number_1dy = np.concatenate([edu_number_1dy1, edu_number_1dy2, edu_number_1dy3])
all_assays_edu_number_2dy = np.concatenate([edu_number_2dy1, edu_number_2dy2, edu_number_2dy3])
all_assays_edu_number_4dy = np.concatenate([edu_number_4dy1, edu_number_4dy2, edu_number_4dy3])

all_assays_mito_length_0dy = np.concatenate([mito_length_0dy1, mito_length_0dy2, mito_length_0dy3])
all_assays_mito_length_1dy = np.concatenate([mito_length_1dy1, mito_length_1dy2, mito_length_1dy3])
all_assays_mito_length_2dy = np.concatenate([mito_length_2dy1, mito_length_2dy2, mito_length_2dy3])
all_assays_mito_length_4dy = np.concatenate([mito_length_4dy1, mito_length_4dy2, mito_length_4dy3])

#Computing confidence intervals for the mean nucleoid and edu number for the training, validation and full dataset

#training
training_dna_number_chase_means = np.array([np.mean(training_dna_number_0dy), np.mean(training_dna_number_1dy), np.mean(training_dna_number_2dy), np.mean(training_dna_number_4dy)])
training_dna_num_0dy_CI = bootstrap((training_dna_number_0dy,), np.mean, axis = 0).confidence_interval
training_dna_num_1dy_CI = bootstrap((training_dna_number_1dy,), np.mean, axis = 0).confidence_interval
training_dna_num_2dy_CI = bootstrap((training_dna_number_2dy,), np.mean, axis = 0).confidence_interval
training_dna_num_4dy_CI = bootstrap((training_dna_number_4dy,), np.mean, axis = 0).confidence_interval

training_dna_number_chase_CI = np.array([training_dna_num_0dy_CI, training_dna_num_1dy_CI, training_dna_num_2dy_CI , training_dna_num_4dy_CI])
training_dna_number_chase_err = np.array([np.std(training_dna_number_0dy)/np.sqrt(len(training_dna_number_0dy)),
                                          np.std(training_dna_number_1dy)/np.sqrt(len(training_dna_number_1dy)),
                                          np.std(training_dna_number_2dy)/np.sqrt(len(training_dna_number_2dy)),
                                          np.std(training_dna_number_4dy)/np.sqrt(len(training_dna_number_4dy))])

training_edu_number_chase_means = np.array([np.mean(training_edu_number_0dy), np.mean(training_edu_number_1dy), np.mean(training_edu_number_2dy), np.mean(training_edu_number_4dy)])
training_edu_num_0dy_CI = bootstrap((training_edu_number_0dy,), np.mean, axis = 0).confidence_interval
training_edu_num_1dy_CI = bootstrap((training_edu_number_1dy,), np.mean, axis = 0).confidence_interval
training_edu_num_2dy_CI = bootstrap((training_edu_number_2dy,), np.mean, axis = 0).confidence_interval
training_edu_num_4dy_CI = bootstrap((training_edu_number_4dy,), np.mean, axis = 0).confidence_interval

training_edu_number_chase_CI = np.array([training_edu_num_0dy_CI, training_edu_num_1dy_CI, training_edu_num_2dy_CI , training_edu_num_4dy_CI])
training_edu_number_chase_err = np.array([np.std(training_edu_number_0dy)/np.sqrt(len(training_edu_number_0dy)),
                                          np.std(training_edu_number_1dy)/np.sqrt(len(training_edu_number_1dy)),
                                          np.std(training_edu_number_2dy)/np.sqrt(len(training_edu_number_2dy)),
                                          np.std(training_edu_number_4dy)/np.sqrt(len(training_edu_number_4dy))])

training_dna_number_pulse_means = np.array([np.mean(training_dna_number_1hr), np.mean(training_dna_number_3hr), np.mean(training_dna_number_7hr), np.mean(training_dna_number_24hr)])
training_dna_num_1hr_CI = bootstrap((training_dna_number_1hr,), np.mean, axis = 0).confidence_interval
training_dna_num_3hr_CI = bootstrap((training_dna_number_3hr,), np.mean, axis = 0).confidence_interval
training_dna_num_7hr_CI = bootstrap((training_dna_number_7hr,), np.mean, axis = 0).confidence_interval
training_dna_num_24hr_CI = bootstrap((training_dna_number_24hr,), np.mean, axis = 0).confidence_interval

training_dna_number_pulse_CI = np.array([training_dna_num_1hr_CI, training_dna_num_3hr_CI, training_dna_num_7hr_CI, training_dna_num_24hr_CI])
training_dna_number_pulse_err = np.array([np.std(training_dna_number_1hr)/np.sqrt(len(training_dna_number_1hr)),
                                          np.std(training_dna_number_3hr)/np.sqrt(len(training_dna_number_3hr)),
                                          np.std(training_dna_number_7hr)/np.sqrt(len(training_dna_number_7hr)),
                                          np.std(training_dna_number_24hr)/np.sqrt(len(training_dna_number_24hr))])

training_edu_number_pulse_means = np.array([np.mean(training_edu_number_1hr), np.mean(training_edu_number_3hr), np.mean(training_edu_number_7hr), np.mean(training_edu_number_24hr)])
training_edu_num_1hr_CI = bootstrap((training_edu_number_1hr,), np.mean, axis = 0).confidence_interval
training_edu_num_3hr_CI = bootstrap((training_edu_number_3hr,), np.mean, axis = 0).confidence_interval
training_edu_num_7hr_CI = bootstrap((training_edu_number_7hr,), np.mean, axis = 0).confidence_interval
training_edu_num_24hr_CI = bootstrap((training_edu_number_24hr,), np.mean, axis = 0).confidence_interval

training_edu_number_pulse_CI = np.array([training_edu_num_1hr_CI, training_edu_num_3hr_CI, training_edu_num_7hr_CI, training_edu_num_24hr_CI])
training_edu_number_pulse_err = np.array([np.std(training_edu_number_1hr)/np.sqrt(len(training_edu_number_1hr)),
                                          np.std(training_edu_number_3hr)/np.sqrt(len(training_edu_number_3hr)),
                                          np.std(training_edu_number_7hr)/np.sqrt(len(training_edu_number_7hr)),
                                          np.std(training_edu_number_24hr)/np.sqrt(len(training_edu_number_24hr))])

training_edu_proportion_chase_means = np.array([np.mean(training_edu_number_0dy/training_dna_number_0dy), np.mean(training_edu_number_1dy/training_dna_number_1dy), np.mean(training_edu_number_2dy/training_dna_number_2dy), np.mean(training_edu_number_4dy/training_dna_number_4dy)])
training_edu_prop_0dy_CI = bootstrap((training_edu_number_0dy/training_dna_number_0dy,), np.mean, axis = 0).confidence_interval
training_edu_prop_1dy_CI = bootstrap((training_edu_number_1dy/training_dna_number_1dy,), np.mean, axis = 0).confidence_interval
training_edu_prop_2dy_CI = bootstrap((training_edu_number_2dy/training_dna_number_2dy,), np.mean, axis = 0).confidence_interval
training_edu_prop_4dy_CI = bootstrap((training_edu_number_4dy/training_dna_number_4dy,), np.mean, axis = 0).confidence_interval
training_edu_proportion_chase_CI = np.array([training_edu_prop_0dy_CI, training_edu_prop_1dy_CI, training_edu_prop_2dy_CI, training_edu_prop_4dy_CI])
training_edu_proportion_chase_err = np.array([np.std(training_edu_number_0dy/training_dna_number_0dy)/np.sqrt(len(training_edu_number_0dy)),
                                          np.std(training_edu_number_1dy/training_dna_number_1dy)/np.sqrt(len(training_edu_number_1dy)),
                                          np.std(training_edu_number_2dy/training_dna_number_2dy)/np.sqrt(len(training_edu_number_2dy)),
                                          np.std(training_edu_number_4dy/training_dna_number_4dy)/np.sqrt(len(training_edu_number_4dy))])

#validation
validation_dna_number_chase_means = np.array([np.mean(validation_dna_number_0dy), np.mean(validation_dna_number_1dy), np.mean(validation_dna_number_2dy), np.mean(validation_dna_number_4dy)])
validation_dna_num_0dy_CI = bootstrap((validation_dna_number_0dy,), np.mean, axis = 0).confidence_interval
validation_dna_num_1dy_CI = bootstrap((validation_dna_number_1dy,), np.mean, axis = 0).confidence_interval
validation_dna_num_2dy_CI = bootstrap((validation_dna_number_2dy,), np.mean, axis = 0).confidence_interval
validation_dna_num_4dy_CI = bootstrap((validation_dna_number_4dy,), np.mean, axis = 0).confidence_interval

validation_dna_number_chase_CI = np.array([validation_dna_num_0dy_CI, validation_dna_num_1dy_CI, validation_dna_num_2dy_CI , validation_dna_num_4dy_CI])
validation_dna_number_chase_err = np.array([np.std(validation_dna_number_0dy)/np.sqrt(len(validation_dna_number_0dy)),
                                          np.std(validation_dna_number_1dy)/np.sqrt(len(validation_dna_number_1dy)),
                                          np.std(validation_dna_number_2dy)/np.sqrt(len(validation_dna_number_2dy)),
                                          np.std(validation_dna_number_4dy)/np.sqrt(len(validation_dna_number_4dy))])

validation_edu_number_chase_means = np.array([np.mean(validation_edu_number_0dy), np.mean(validation_edu_number_1dy), np.mean(validation_edu_number_2dy), np.mean(validation_edu_number_4dy)])
validation_edu_num_0dy_CI = bootstrap((validation_edu_number_0dy,), np.mean, axis = 0).confidence_interval
validation_edu_num_1dy_CI = bootstrap((validation_edu_number_1dy,), np.mean, axis = 0).confidence_interval
validation_edu_num_2dy_CI = bootstrap((validation_edu_number_2dy,), np.mean, axis = 0).confidence_interval
validation_edu_num_4dy_CI = bootstrap((validation_edu_number_4dy,), np.mean, axis = 0).confidence_interval

validation_edu_number_chase_CI = np.array([validation_edu_num_0dy_CI, validation_edu_num_1dy_CI, validation_edu_num_2dy_CI , validation_edu_num_4dy_CI])
validation_edu_number_chase_err = np.array([np.std(validation_edu_number_0dy)/np.sqrt(len(validation_edu_number_0dy)),
                                          np.std(validation_edu_number_1dy)/np.sqrt(len(validation_edu_number_1dy)),
                                          np.std(validation_edu_number_2dy)/np.sqrt(len(validation_edu_number_2dy)),
                                          np.std(validation_edu_number_4dy)/np.sqrt(len(validation_edu_number_4dy))])

validation_dna_number_pulse_means = np.array([np.mean(validation_dna_number_1hr), np.mean(validation_dna_number_3hr), np.mean(validation_dna_number_7hr), np.mean(validation_dna_number_24hr)])
validation_dna_num_1hr_CI = bootstrap((validation_dna_number_1hr,), np.mean, axis = 0).confidence_interval
validation_dna_num_3hr_CI = bootstrap((validation_dna_number_3hr,), np.mean, axis = 0).confidence_interval
validation_dna_num_7hr_CI = bootstrap((validation_dna_number_7hr,), np.mean, axis = 0).confidence_interval
validation_dna_num_24hr_CI = bootstrap((validation_dna_number_24hr,), np.mean, axis = 0).confidence_interval

validation_dna_number_pulse_CI = np.array([validation_dna_num_1hr_CI, validation_dna_num_3hr_CI, validation_dna_num_7hr_CI, validation_dna_num_24hr_CI])
validation_dna_number_pulse_err = np.array([np.std(validation_dna_number_1hr)/np.sqrt(len(validation_dna_number_1hr)),
                                          np.std(validation_dna_number_3hr)/np.sqrt(len(validation_dna_number_3hr)),
                                          np.std(validation_dna_number_7hr)/np.sqrt(len(validation_dna_number_7hr)),
                                          np.std(validation_dna_number_24hr)/np.sqrt(len(validation_dna_number_24hr))])

validation_edu_number_pulse_means = np.array([np.mean(validation_edu_number_1hr), np.mean(validation_edu_number_3hr), np.mean(validation_edu_number_7hr), np.mean(validation_edu_number_24hr)])
validation_edu_num_1hr_CI = bootstrap((validation_edu_number_1hr,), np.mean, axis = 0).confidence_interval
validation_edu_num_3hr_CI = bootstrap((validation_edu_number_3hr,), np.mean, axis = 0).confidence_interval
validation_edu_num_7hr_CI = bootstrap((validation_edu_number_7hr,), np.mean, axis = 0).confidence_interval
validation_edu_num_24hr_CI = bootstrap((validation_edu_number_24hr,), np.mean, axis = 0).confidence_interval

validation_edu_number_pulse_CI = np.array([validation_edu_num_1hr_CI, validation_edu_num_3hr_CI, validation_edu_num_7hr_CI, validation_edu_num_24hr_CI])
validation_edu_number_pulse_err = np.array([np.std(validation_edu_number_1hr)/np.sqrt(len(validation_edu_number_1hr)),
                                          np.std(validation_edu_number_3hr)/np.sqrt(len(validation_edu_number_3hr)),
                                          np.std(validation_edu_number_7hr)/np.sqrt(len(validation_edu_number_7hr)),
                                          np.std(validation_edu_number_24hr)/np.sqrt(len(validation_edu_number_24hr))])


validation_edu_proportion_chase_means = np.array([np.mean(validation_edu_number_0dy/validation_dna_number_0dy), np.mean(validation_edu_number_1dy/validation_dna_number_1dy), np.mean(validation_edu_number_2dy/validation_dna_number_2dy), np.mean(validation_edu_number_4dy/validation_dna_number_4dy)])
validation_edu_prop_0dy_CI = bootstrap((validation_edu_number_0dy/validation_dna_number_0dy,), np.mean, axis = 0).confidence_interval
validation_edu_prop_1dy_CI = bootstrap((validation_edu_number_1dy/validation_dna_number_1dy,), np.mean, axis = 0).confidence_interval
validation_edu_prop_2dy_CI = bootstrap((validation_edu_number_2dy/validation_dna_number_2dy,), np.mean, axis = 0).confidence_interval
validation_edu_prop_4dy_CI = bootstrap((validation_edu_number_4dy/validation_dna_number_4dy,), np.mean, axis = 0).confidence_interval
validation_edu_proportion_chase_CI = np.array([validation_edu_prop_0dy_CI, validation_edu_prop_1dy_CI, validation_edu_prop_2dy_CI, validation_edu_prop_4dy_CI])
validation_edu_proportion_chase_err = np.array([np.std(validation_edu_number_0dy/validation_dna_number_0dy)/np.sqrt(len(validation_edu_number_0dy)),
                                          np.std(validation_edu_number_1dy/validation_dna_number_1dy)/np.sqrt(len(validation_edu_number_1dy)),
                                          np.std(validation_edu_number_2dy/validation_dna_number_2dy)/np.sqrt(len(validation_edu_number_2dy)),
                                          np.std(validation_edu_number_4dy/validation_dna_number_4dy)/np.sqrt(len(validation_edu_number_4dy))])


#all_assays
all_assays_dna_number_chase_means = np.array([np.mean(all_assays_dna_number_0dy), np.mean(all_assays_dna_number_1dy), np.mean(all_assays_dna_number_2dy), np.mean(all_assays_dna_number_4dy)])
all_assays_dna_num_0dy_CI = bootstrap((all_assays_dna_number_0dy,), np.mean, axis = 0).confidence_interval
all_assays_dna_num_1dy_CI = bootstrap((all_assays_dna_number_1dy,), np.mean, axis = 0).confidence_interval
all_assays_dna_num_2dy_CI = bootstrap((all_assays_dna_number_2dy,), np.mean, axis = 0).confidence_interval
all_assays_dna_num_4dy_CI = bootstrap((all_assays_dna_number_4dy,), np.mean, axis = 0).confidence_interval

all_assays_dna_number_chase_CI = np.array([all_assays_dna_num_0dy_CI, all_assays_dna_num_1dy_CI, all_assays_dna_num_2dy_CI , all_assays_dna_num_4dy_CI])
all_assays_dna_number_chase_err = np.array([np.std(all_assays_dna_number_0dy)/np.sqrt(len(all_assays_dna_number_0dy)),
                                          np.std(all_assays_dna_number_1dy)/np.sqrt(len(all_assays_dna_number_1dy)),
                                          np.std(all_assays_dna_number_2dy)/np.sqrt(len(all_assays_dna_number_2dy)),
                                          np.std(all_assays_dna_number_4dy)/np.sqrt(len(all_assays_dna_number_4dy))])


all_assays_edu_number_chase_means = np.array([np.mean(all_assays_edu_number_0dy), np.mean(all_assays_edu_number_1dy), np.mean(all_assays_edu_number_2dy), np.mean(all_assays_edu_number_4dy)])
all_assays_edu_num_0dy_CI = bootstrap((all_assays_edu_number_0dy,), np.mean, axis = 0).confidence_interval
all_assays_edu_num_1dy_CI = bootstrap((all_assays_edu_number_1dy,), np.mean, axis = 0).confidence_interval
all_assays_edu_num_2dy_CI = bootstrap((all_assays_edu_number_2dy,), np.mean, axis = 0).confidence_interval
all_assays_edu_num_4dy_CI = bootstrap((all_assays_edu_number_4dy,), np.mean, axis = 0).confidence_interval

all_assays_edu_number_chase_CI = np.array([all_assays_edu_num_0dy_CI, all_assays_edu_num_1dy_CI, all_assays_edu_num_2dy_CI , all_assays_edu_num_4dy_CI])
all_assays_edu_number_chase_err = np.array([np.std(all_assays_edu_number_0dy)/np.sqrt(len(all_assays_edu_number_0dy)),
                                          np.std(all_assays_edu_number_1dy)/np.sqrt(len(all_assays_edu_number_1dy)),
                                          np.std(all_assays_edu_number_2dy)/np.sqrt(len(all_assays_edu_number_2dy)),
                                          np.std(all_assays_edu_number_4dy)/np.sqrt(len(all_assays_edu_number_4dy))])


all_assays_dna_number_pulse_means = np.array([np.mean(all_assays_dna_number_1hr), np.mean(all_assays_dna_number_3hr), np.mean(all_assays_dna_number_7hr), np.mean(all_assays_dna_number_24hr)])
all_assays_dna_num_1hr_CI = bootstrap((all_assays_dna_number_1hr,), np.mean, axis = 0).confidence_interval
all_assays_dna_num_3hr_CI = bootstrap((all_assays_dna_number_3hr,), np.mean, axis = 0).confidence_interval
all_assays_dna_num_7hr_CI = bootstrap((all_assays_dna_number_7hr,), np.mean, axis = 0).confidence_interval
all_assays_dna_num_24hr_CI = bootstrap((all_assays_dna_number_24hr,), np.mean, axis = 0).confidence_interval

all_assays_dna_number_pulse_CI = np.array([all_assays_dna_num_1hr_CI, all_assays_dna_num_3hr_CI, all_assays_dna_num_7hr_CI, all_assays_dna_num_24hr_CI])
all_assays_dna_number_pulse_err = np.array([np.std(all_assays_dna_number_1hr)/np.sqrt(len(all_assays_dna_number_1hr)),
                                          np.std(all_assays_dna_number_3hr)/np.sqrt(len(all_assays_dna_number_3hr)),
                                          np.std(all_assays_dna_number_7hr)/np.sqrt(len(all_assays_dna_number_7hr)),
                                          np.std(all_assays_dna_number_24hr)/np.sqrt(len(all_assays_dna_number_24hr))])

all_assays_edu_number_pulse_means = np.array([np.mean(all_assays_edu_number_1hr), np.mean(all_assays_edu_number_3hr), np.mean(all_assays_edu_number_7hr), np.mean(all_assays_edu_number_24hr)])
all_assays_edu_num_1hr_CI = bootstrap((all_assays_edu_number_1hr,), np.mean, axis = 0).confidence_interval
all_assays_edu_num_3hr_CI = bootstrap((all_assays_edu_number_3hr,), np.mean, axis = 0).confidence_interval
all_assays_edu_num_7hr_CI = bootstrap((all_assays_edu_number_7hr,), np.mean, axis = 0).confidence_interval
all_assays_edu_num_24hr_CI = bootstrap((all_assays_edu_number_24hr,), np.mean, axis = 0).confidence_interval

all_assays_edu_number_pulse_CI = np.array([all_assays_edu_num_1hr_CI, all_assays_edu_num_3hr_CI, all_assays_edu_num_7hr_CI, all_assays_edu_num_24hr_CI])
all_assays_edu_number_pulse_err = np.array([np.std(all_assays_edu_number_1hr)/np.sqrt(len(all_assays_edu_number_1hr)),
                                          np.std(all_assays_edu_number_3hr)/np.sqrt(len(all_assays_edu_number_3hr)),
                                          np.std(all_assays_edu_number_7hr)/np.sqrt(len(all_assays_edu_number_7hr)),
                                          np.std(all_assays_edu_number_24hr)/np.sqrt(len(all_assays_edu_number_24hr))])

all_assays_edu_proportion_chase_means = np.array([np.mean(all_assays_edu_number_0dy/all_assays_dna_number_0dy), np.mean(all_assays_edu_number_1dy/all_assays_dna_number_1dy), np.mean(all_assays_edu_number_2dy/all_assays_dna_number_2dy), np.mean(all_assays_edu_number_4dy/all_assays_dna_number_4dy)])
all_assays_edu_prop_0dy_CI = bootstrap((all_assays_edu_number_0dy/all_assays_dna_number_0dy,), np.mean, axis = 0).confidence_interval
all_assays_edu_prop_1dy_CI = bootstrap((all_assays_edu_number_1dy/all_assays_dna_number_1dy,), np.mean, axis = 0).confidence_interval
all_assays_edu_prop_2dy_CI = bootstrap((all_assays_edu_number_2dy/all_assays_dna_number_2dy,), np.mean, axis = 0).confidence_interval
all_assays_edu_prop_4dy_CI = bootstrap((all_assays_edu_number_4dy/all_assays_dna_number_4dy,), np.mean, axis = 0).confidence_interval
all_assays_edu_proportion_chase_CI = np.array([all_assays_edu_prop_0dy_CI, all_assays_edu_prop_1dy_CI, all_assays_edu_prop_2dy_CI, all_assays_edu_prop_4dy_CI])
all_assays_edu_proportion_chase_err = np.array([np.std(all_assays_edu_number_0dy/all_assays_dna_number_0dy)/np.sqrt(len(all_assays_edu_number_0dy)),
                                          np.std(all_assays_edu_number_1dy/all_assays_dna_number_1dy)/np.sqrt(len(all_assays_edu_number_1dy)),
                                          np.std(all_assays_edu_number_2dy/all_assays_dna_number_2dy)/np.sqrt(len(all_assays_edu_number_2dy)),
                                          np.std(all_assays_edu_number_4dy/all_assays_dna_number_4dy)/np.sqrt(len(all_assays_edu_number_4dy))])


#############################################################################################################################
#                                                                                                                           #
#                                                    NUCLEOID INTENSITY                                                     #
#                                                           DATA                                                            #
#                                                        EXTRACTION                                                         #
#                                                                                                                           #
#############################################################################################################################

chase_intensities_1 = pd.ExcelFile(os.path.join(dirname, "nucleoid_intensity_data/assay1_intensities.xlsx"))
chase_intensities_2 = pd.ExcelFile(os.path.join(dirname, "nucleoid_intensity_data/assay2_intensities.xlsx"))
chase_intensities_3 = pd.ExcelFile(os.path.join(dirname, "nucleoid_intensity_data/assay3_intensities.xlsx"))

intensities_0dy1 = np.array(pd.read_excel(chase_intensities_1, "0 days")).transpose()[0]
intensities_1dy1 = np.array(pd.read_excel(chase_intensities_1, "1 day")).transpose()[0]
intensities_2dy1 = np.array(pd.read_excel(chase_intensities_1, "2 days")).transpose()[0]
intensities_4dy1 = np.array(pd.read_excel(chase_intensities_1, "4 days")).transpose()[0]

intensities_0dy2 = np.array(pd.read_excel(chase_intensities_2, "0 days")).transpose()[0]
intensities_1dy2 = np.array(pd.read_excel(chase_intensities_2, "1 day")).transpose()[0]
intensities_2dy2 = np.array(pd.read_excel(chase_intensities_2, "2 days")).transpose()[0]
intensities_4dy2 = np.array(pd.read_excel(chase_intensities_2, "4 days")).transpose()[0]

intensities_0dy3 = np.array(pd.read_excel(chase_intensities_3, "0 days")).transpose()[0]
intensities_1dy3 = np.array(pd.read_excel(chase_intensities_3, "1 day")).transpose()[0]
intensities_2dy3 = np.array(pd.read_excel(chase_intensities_3, "2 days")).transpose()[0]
intensities_4dy3 = np.array(pd.read_excel(chase_intensities_3, "4 days")).transpose()[0]

#############################################################################################################################
#                                                                                                                           #
#                                                       DEGRADATION                                                         #
#                                                           DATA                                                            #
#                                                        EXTRACTION                                                         #
#                                                                                                                           #
#############################################################################################################################
excel_file_assay1= pd.ExcelFile(os.path.join(dirname, "degradation_data/Assay 1 Ethidium Bromide Summary.xlsx"))
excel_file_assay2= pd.ExcelFile(os.path.join(dirname, "degradation_data/Assay 2 Ethidium Bromide Summary.xlsx"))
excel_file_assay3= pd.ExcelFile(os.path.join(dirname, "degradation_data/Assay 3 Ethidium Bromide Summary.xlsx"))

deg_dna_0hr1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 21), sheet_name = "0 hr")["count.1"])
deg_dna_1hr1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 20), sheet_name = "1 hr")["count.1"])
deg_dna_3hr1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 16), sheet_name = "3 hr")["count.1"])
deg_dna_7hr1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 20), sheet_name = "7 hr")["count.1"])
deg_dna_24hr1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x <1) or (x >= 24), sheet_name = "24 hr")["count.1"])
deg_dna_48hr1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 18), sheet_name = "48 hr")["count.1"])
deg_dna_5dy1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 20), sheet_name = "5 days")["count.1"])
deg_dna_10dy1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 19), sheet_name = "10 days")["count.1"])
deg_dna_15dy1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 18), sheet_name = "15 days")["count.1"])
deg_dna_20dy1 = np.array(pd.read_excel(excel_file_assay1, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 10), sheet_name = "20 days")["count.1"])

deg_dna_0hr2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 19), sheet_name = "0 hr")["count.1"])
deg_dna_1hr2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 17), sheet_name = "1 hr")["count.1"])
deg_dna_3hr2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 20), sheet_name = "3 hr")["count.1"])
deg_dna_7hr2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'M', skiprows = lambda x: (x < 1) or (x >= 13), sheet_name = "7 hr")["count.1"])
deg_dna_24hr2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x <1) or (x >= 14), sheet_name = "24 hr")["count.1"])
deg_dna_48hr2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 16), sheet_name = "48 hr")["count.1"])
deg_dna_5dy2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 20), sheet_name = "5 days")["count.1"])
deg_dna_10dy2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 20), sheet_name = "10 days")["count.1"])
deg_dna_15dy2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 13), sheet_name = "15 days")["count.1"])
deg_dna_20dy2 = np.array(pd.read_excel(excel_file_assay2, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 15), sheet_name = "20 days")["count.1"])

deg_dna_0hr3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 19), sheet_name = "0 hr")["count.1"])
deg_dna_1hr3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 13), sheet_name = "1 hr")["count.1"])
deg_dna_3hr3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 17), sheet_name = "3 hr")["count.1"])
deg_dna_7hr3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 17), sheet_name = "7 hr")["count.1"])
deg_dna_24hr3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x <1) or (x >= 17), sheet_name = "24 hr")["count.1"])
deg_dna_48hr3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 11), sheet_name = "48 hr")["count.1"])
deg_dna_5dy3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 23), sheet_name = "5 days")["count.1"])
deg_dna_10dy3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 23), sheet_name = "10 days")["count.1"])
deg_dna_15dy3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 17), sheet_name = "15 days")["count.1"])
deg_dna_20dy3 = np.array(pd.read_excel(excel_file_assay3, usecols = 'N', skiprows = lambda x: (x < 1) or (x >= 22), sheet_name = "20 days")["count.1"])

deg_nuc_0hr = np.concatenate([deg_dna_0hr1, deg_dna_0hr2, deg_dna_0hr3])
deg_nuc_1hr = np.concatenate([deg_dna_1hr1, deg_dna_1hr2, deg_dna_1hr3])
deg_nuc_3hr = np.concatenate([deg_dna_3hr1, deg_dna_3hr2, deg_dna_3hr3])
deg_nuc_7hr = np.concatenate([deg_dna_7hr1, deg_dna_7hr2, deg_dna_7hr3])
deg_nuc_24hr = np.concatenate([deg_dna_24hr1, deg_dna_24hr2, deg_dna_24hr3])
deg_nuc_48hr = np.concatenate([deg_dna_48hr1, deg_dna_48hr2, deg_dna_48hr3])
deg_nuc_5dy = np.concatenate([deg_dna_5dy1, deg_dna_5dy2, deg_dna_5dy3])
deg_nuc_10dy = np.concatenate([deg_dna_10dy1, deg_dna_10dy2, deg_dna_10dy3])
deg_nuc_15dy = np.concatenate([deg_dna_15dy1, deg_dna_15dy2, deg_dna_15dy3])
deg_nuc_20dy = np.concatenate([deg_dna_20dy1, deg_dna_20dy2, deg_dna_20dy3])

deg_mito_0hr1 = pd.read_excel(excel_file_assay1, usecols = 'F:L', skiprows = lambda x: (x < 1) or (x >= 21), sheet_name = "0 hr")
deg_mito_0hr2 = pd.read_excel(excel_file_assay2, usecols = 'F:L', skiprows = lambda x: (x < 1) or (x >= 19), sheet_name = "0 hr")
deg_mito_0hr3 = pd.read_excel(excel_file_assay3, usecols = 'F:L', skiprows = lambda x: (x < 1) or (x >= 19), sheet_name = "0 hr")
deg_mt_length_0hr1 = np.array(deg_mito_0hr1["total volume"])
deg_mt_length_0hr2 = np.array(deg_mito_0hr2["total volume"])
deg_mt_length_0hr3 = np.array(deg_mito_0hr3["total volume"])

deg_mito_1hr1 = pd.read_excel(excel_file_assay1, usecols = 'F:L', skiprows = lambda x: (x < 1) or (x >= 20), sheet_name = "1 hr")
deg_mito_1hr2 = pd.read_excel(excel_file_assay2, usecols = 'F:L', skiprows = lambda x: (x < 1) or (x >= 17), sheet_name = "1 hr")
deg_mito_1hr3 = pd.read_excel(excel_file_assay3, usecols = 'F:L', skiprows = lambda x: (x < 1) or (x >= 13), sheet_name = "1 hr")
deg_mt_length_1hr1 = np.array(deg_mito_1hr1["total volume"])
deg_mt_length_1hr2 = np.array(deg_mito_1hr2["total volume"])
deg_mt_length_1hr3 = np.array(deg_mito_1hr3["total volume"])

deg_mt_length_0hr = np.concatenate([deg_mt_length_0hr1, deg_mt_length_0hr2, deg_mt_length_0hr3])
deg_mt_length_1hr = np.concatenate([deg_mt_length_1hr1, deg_mt_length_1hr2, deg_mt_length_1hr3])

mean_deg_nuc_list = np.array([np.mean(deg_nuc_0hr), np.mean(deg_nuc_1hr),np.mean(deg_nuc_3hr),np.mean(deg_nuc_7hr),np.mean(deg_nuc_24hr),np.mean(deg_nuc_48hr),
                                                   np.mean(deg_nuc_5dy),np.mean(deg_nuc_10dy),np.mean(deg_nuc_15dy),np.mean(deg_nuc_20dy)])

var_deg_nuc_list = np.array([np.var(deg_nuc_0hr), np.var(deg_nuc_1hr),np.var(deg_nuc_3hr),np.var(deg_nuc_7hr),np.var(deg_nuc_24hr),np.var(deg_nuc_48hr),
                                                   np.var(deg_nuc_5dy),np.var(deg_nuc_10dy),np.var(deg_nuc_15dy),np.var(deg_nuc_20dy)])

len_deg_nuc_list = np.array([len(deg_nuc_0hr), len(deg_nuc_1hr),len(deg_nuc_3hr),len(deg_nuc_7hr),len(deg_nuc_24hr),len(deg_nuc_48hr),
                                                   len(deg_nuc_5dy),len(deg_nuc_10dy),len(deg_nuc_15dy),len(deg_nuc_20dy)])

var_mean_deg_nuc_list = var_deg_nuc_list/len_deg_nuc_list
#############################################################################################################################
#                                                                                                                           #
#                                                        4D IMAGING                                                         #
#                                                           DATA                                                            #
#                                                        EXTRACTION                                                         #
#                                                                                                                           #
#############################################################################################################################

excel_file_assay1 = pd.ExcelFile(os.path.join(dirname, "4d_imaging_data/Cells No FBS fibro 24 hrs Assay 1.xlsx"))
excel_file_assay2 = pd.ExcelFile(os.path.join(dirname, "4d_imaging_data/Cells No FBS fibro 24 hrs Assay 2.xlsx"))
excel_file_assay3 = pd.ExcelFile(os.path.join(dirname, "4d_imaging_data/Cells No FBS fibro 24 hrs Assay 3.xlsx"))

###############################---------------------- ASSAY 1 -------------------------------###############

cell1_4d_dna = np.array(pd.read_excel(excel_file_assay1,"Cell 1", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell1_4d_mtvol = np.array(pd.read_excel(excel_file_assay1,"Cell 1", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell1_4d_cellvol = np.array(pd.read_excel(excel_file_assay1,"Cell 1", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell2_4d_dna = np.array(pd.read_excel(excel_file_assay1,"Cell 2", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell2_4d_mtvol = np.array(pd.read_excel(excel_file_assay1,"Cell 2", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell2_4d_cellvol = np.array(pd.read_excel(excel_file_assay1,"Cell 2", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell3_4d_dna = np.array(pd.read_excel(excel_file_assay1,"Cell 3", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell3_4d_mtvol = np.array(pd.read_excel(excel_file_assay1,"Cell 3", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell3_4d_cellvol = np.array(pd.read_excel(excel_file_assay1,"Cell 3", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell4_4d_dna = np.array(pd.read_excel(excel_file_assay1,"Cell 4", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell4_4d_mtvol = np.array(pd.read_excel(excel_file_assay1,"Cell 4", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell4_4d_cellvol = np.array(pd.read_excel(excel_file_assay1,"Cell 4", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell5_4d_dna = np.array(pd.read_excel(excel_file_assay1,"Cell 5", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell5_4d_mtvol = np.array(pd.read_excel(excel_file_assay1,"Cell 5", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell5_4d_cellvol = np.array(pd.read_excel(excel_file_assay1,"Cell 5", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell6_4d_dna = np.array(pd.read_excel(excel_file_assay1,"Cell 6", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell6_4d_mtvol = np.array(pd.read_excel(excel_file_assay1,"Cell 6", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell6_4d_cellvol = np.array(pd.read_excel(excel_file_assay1,"Cell 6", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell7_4d_dna = np.array(pd.read_excel(excel_file_assay1,"Cell 7", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell7_4d_mtvol = np.array(pd.read_excel(excel_file_assay1,"Cell 7", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell7_4d_cellvol = np.array(pd.read_excel(excel_file_assay1,"Cell 7", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

###---------------------- ASSAY 2 -------------------------------###############

cell8_4d_dna = np.array(pd.read_excel(excel_file_assay2,"Cell 1", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell8_4d_mtvol = np.array(pd.read_excel(excel_file_assay2,"Cell 1", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell8_4d_cellvol = np.array(pd.read_excel(excel_file_assay2,"Cell 1", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell9_4d_dna = np.array(pd.read_excel(excel_file_assay2,"Cell 2", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell9_4d_mtvol = np.array(pd.read_excel(excel_file_assay2,"Cell 2", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell9_4d_cellvol = np.array(pd.read_excel(excel_file_assay2,"Cell 2", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell10_4d_dna = np.array(pd.read_excel(excel_file_assay2,"Cell 3", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell10_4d_mtvol = np.array(pd.read_excel(excel_file_assay2,"Cell 3", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell10_4d_cellvol = np.array(pd.read_excel(excel_file_assay2,"Cell 3", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell11_4d_dna = np.array(pd.read_excel(excel_file_assay2,"Cell 4", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell11_mtvol = np.array(pd.read_excel(excel_file_assay2,"Cell 4", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell11_4d_cellvol = np.array(pd.read_excel(excel_file_assay2,"Cell 4", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell12_4d_dna = np.array(pd.read_excel(excel_file_assay2,"Cell 5", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell12_4d_mtvol = np.array(pd.read_excel(excel_file_assay2,"Cell 5", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell12_4d_cellvol = np.array(pd.read_excel(excel_file_assay2,"Cell 5", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell13_4d_dna = np.array(pd.read_excel(excel_file_assay2,"Cell 6", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell13_4d_mtvol = np.array(pd.read_excel(excel_file_assay2,"Cell 6", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell13_4d_cellvol = np.array(pd.read_excel(excel_file_assay2,"Cell 6", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell14_4d_dna = np.array(pd.read_excel(excel_file_assay2,"Cell 7", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell14_4d_mtvol = np.array(pd.read_excel(excel_file_assay2,"Cell 7", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell14_4d_cellvol = np.array(pd.read_excel(excel_file_assay2,"Cell 7", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

###---------------------- ASSAY 3 -------------------------------###############

cell15_4d_dna = np.array(pd.read_excel(excel_file_assay3,"Cell 1", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell15_4d_mtvol = np.array(pd.read_excel(excel_file_assay3,"Cell 1", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell15_4d_cellvol = np.array(pd.read_excel(excel_file_assay3,"Cell 1", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell16_4d_dna = np.array(pd.read_excel(excel_file_assay3,"Cell 2", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell16_4d_mtvol = np.array(pd.read_excel(excel_file_assay3,"Cell 2", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell16_4d_cellvol = np.array(pd.read_excel(excel_file_assay3,"Cell 2", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell17_4d_dna = np.array(pd.read_excel(excel_file_assay3,"Cell 3", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell17_4d_mtvol = np.array(pd.read_excel(excel_file_assay3,"Cell 3", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell17_4d_cellvol = np.array(pd.read_excel(excel_file_assay3,"Cell 3", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell18_4d_dna = np.array(pd.read_excel(excel_file_assay3,"Cell 4", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell18_4d_mtvol = np.array(pd.read_excel(excel_file_assay3,"Cell 4", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell18_4d_cellvol = np.array(pd.read_excel(excel_file_assay3,"Cell 4", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

cell19_4d_dna = np.array(pd.read_excel(excel_file_assay3,"Cell 5", usecols = 'U', skiprows = lambda x: (x < 4) or (x >= 18))["count.1"])
cell19_4d_mtvol = np.array(pd.read_excel(excel_file_assay3,"Cell 5", usecols = 'F', skiprows = lambda x: (x < 4) or (x >= 18))["total volume"])
cell19_4d_cellvol = np.array(pd.read_excel(excel_file_assay3,"Cell 5", usecols = 'C', skiprows = lambda x: (x < 4) or (x >= 18))["Cell volume"])

all_4d_dnas = np.array([cell1_4d_dna, cell2_4d_dna, cell3_4d_dna, cell4_4d_dna, cell5_4d_dna, cell6_4d_dna, cell7_4d_dna, cell8_4d_dna, cell9_4d_dna, cell10_4d_dna, 
            cell11_4d_dna, cell12_4d_dna, cell13_4d_dna, cell14_4d_dna, cell15_4d_dna, cell16_4d_dna, cell17_4d_dna, cell18_4d_dna, cell19_4d_dna])

all_4d_mtvols = np.array([cell1_4d_mtvol, cell2_4d_mtvol, cell3_4d_mtvol, cell4_4d_mtvol, cell5_4d_mtvol, cell6_4d_mtvol, cell7_4d_mtvol, cell8_4d_mtvol, cell9_4d_mtvol, cell10_4d_mtvol, 
            cell11_mtvol, cell12_4d_mtvol, cell13_4d_mtvol, cell14_4d_mtvol, cell15_4d_mtvol, cell16_4d_mtvol, cell17_4d_mtvol, cell18_4d_mtvol, cell19_4d_mtvol])

all_4d_cellvols = np.array([cell1_4d_cellvol, cell2_4d_cellvol, cell3_4d_cellvol, cell4_4d_cellvol, cell5_4d_cellvol, cell6_4d_cellvol, cell7_4d_cellvol, cell8_4d_cellvol, cell9_4d_cellvol, cell10_4d_cellvol, 
            cell11_4d_cellvol, cell12_4d_cellvol, cell13_4d_cellvol, cell14_4d_cellvol, cell15_4d_cellvol, cell16_4d_cellvol, cell17_4d_cellvol, cell18_4d_cellvol, cell19_4d_cellvol])

#############################################################################################################################
#                                                                                                                           #
#                                                        PULSE + CHASE                                                      #
#                                                           SUMMARY                                                         #
#                                                          STATISTICS                                                       #
#                                                                                                                           #
#############################################################################################################################
#This code chunk estimates the variance of the summary statistic S_{cs} in the data

#Computing the OLS estimate of beta0, beta1 for the training data
X = np.concatenate([np.ones(len(training_mito_lengths))[:,np.newaxis], training_mito_lengths[:,np.newaxis]], axis=1)
Y = training_dna_numbers
XtX = np.dot(X.transpose(), X)
XtX_inv = np.linalg.inv(XtX)
mean = np.dot(XtX_inv, X.transpose())
betas = np.dot(mean, Y)

#Computing the variance of the log residuals at each time point
log_residuals_1hr_var = np.var(np.log(training_dna_number_1hr) - np.log(betas[0] + betas[1]*training_mito_length_1hr))
log_residuals_3hr_var = np.var(np.log(training_dna_number_3hr) - np.log(betas[0] + betas[1]*training_mito_length_3hr))
log_residuals_7hr_var = np.var(np.log(training_dna_number_7hr) - np.log(betas[0] + betas[1]*training_mito_length_7hr))
log_residuals_24hr_var = np.var(np.log(training_dna_number_24hr) - np.log(betas[0] + betas[1]*training_mito_length_24hr))
log_residuals_vars = [log_residuals_1hr_var, log_residuals_3hr_var, log_residuals_7hr_var,log_residuals_24hr_var]

#Computing a symmetric matrix of estimates of S_{cs} in the data. The variance of this matrix will be our estimate of
#Var[S_{cs}]
data_variance_summary_statistic_matrix = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        data_variance_summary_statistic_matrix[i,j] = np.log(log_residuals_vars[i]/log_residuals_vars[j]) 


#-----------------------------Pulse summary statistics---------------------------------------------#

pulse_data_summary_statistics = np.array([np.mean(training_edu_number_1hr), np.mean(training_edu_number_3hr), np.mean(training_edu_number_7hr), 
                                     np.mean(training_edu_number_24hr), 0.33, 0])
#the 0.33 is the mean peak1 proportion. See gaussian_mixture_figures.ipynb

pulse_data_summary_statistics_variance = np.array([np.var(training_edu_number_1hr)/len(training_edu_number_1hr),
            np.var(training_edu_number_3hr)/len(training_edu_number_3hr),
            np.var(training_edu_number_7hr)/len(training_edu_number_7hr),
            np.var(training_edu_number_24hr)/len(training_edu_number_24hr),
            0.002, np.var(data_variance_summary_statistic_matrix)])

pulse_data_covariance_matrix = np.diag(pulse_data_summary_statistics_variance)
#the 0.002 is the variance of the peak1 proportion. See gaussian_mixture_figures.ipynb

#for validation
pulse_data_summary_statistics_val = np.array([np.mean(validation_edu_number_1hr), np.mean(validation_edu_number_3hr), np.mean(validation_edu_number_7hr), 
                                     np.mean(validation_edu_number_24hr), 0.33, 0])

#pulse_summary_statistics but with the heteroscedasticity summary statistic
pulse_data_summary_statistics_with_hetero = np.concatenate([pulse_data_summary_statistics, np.array([0])])

#pulse_summary_statistics but with the heteroscedasticity summary statistic for both nucleoid number and edu number
pulse_data_summary_statistics_with_edu_hetero = np.concatenate([pulse_data_summary_statistics, np.array([0,0,0,0,0])])

def moving_variance(dna, mt, b0, b1, sorted_indices,window_size=20):
    residuals = dna[sorted_indices] - b0 - b1*mt[sorted_indices]
    moving_variance = np.convolve(residuals**2, np.ones(window_size)/window_size, mode='valid')
    moving_variance_x_axis = np.convolve(mt[sorted_indices], np.ones(window_size)/window_size, mode='valid')

    return moving_variance_x_axis, moving_variance

def hetero_summary_statistic(dna, dna2, mt, mt2, b0, b1, window_size=20):
    x2, moving_variance2 = moving_variance(dna2, mt2, b0, b1, np.argsort(mt2), window_size)
    x1, moving_variance1 = moving_variance(dna, mt, b0, b1, np.argsort(mt), window_size)
    x = np.concatenate([x1,x2])
    x.sort()
    more_moving_variance1 = np.interp(x,x1,moving_variance1)
    more_moving_variance2 = np.interp(x, x2, moving_variance2)
    return np.linalg.norm(np.sqrt(more_moving_variance1) - np.sqrt(more_moving_variance2))

#Training
#n: 214, 1.23
#e1: 53, 0.06
#e3: 42,0.13
#e7: 35, 0.27
#e24: 99, 0.51

h1 = np.sqrt(len(training_dna_numbers))/np.sqrt(len(training_dna_number_1hr) + len(training_dna_number_3hr)) * hetero_summary_statistic(training_dna_number_1hr, training_dna_number_3hr, training_mito_length_1hr, training_mito_length_3hr, 214, 1.23)
h2 = np.sqrt(len(training_dna_numbers))/np.sqrt(len(training_dna_number_1hr) + len(training_dna_number_7hr)) * hetero_summary_statistic(training_dna_number_1hr, training_dna_number_7hr, training_mito_length_1hr, training_mito_length_7hr, 214, 1.23)
h3 = np.sqrt(len(training_dna_numbers))/np.sqrt(len(training_dna_number_1hr) + len(training_dna_number_24hr)) * hetero_summary_statistic(training_dna_number_1hr, training_dna_number_24hr, training_mito_length_1hr, training_mito_length_24hr, 214, 1.23)
h4 = np.sqrt(len(training_dna_numbers))/np.sqrt(len(training_dna_number_3hr) + len(training_dna_number_7hr)) * hetero_summary_statistic(training_dna_number_3hr, training_dna_number_7hr, training_mito_length_3hr, training_mito_length_7hr, 214, 1.23)
h5 = np.sqrt(len(training_dna_numbers))/np.sqrt(len(training_dna_number_3hr) + len(training_dna_number_24hr)) * hetero_summary_statistic(training_dna_number_3hr, training_dna_number_24hr, training_mito_length_3hr, training_mito_length_24hr, 214, 1.23)
h6 = np.sqrt(len(training_dna_numbers))/np.sqrt(len(training_dna_number_7hr) + len(training_dna_number_24hr)) * hetero_summary_statistic(training_dna_number_7hr, training_dna_number_24hr, training_mito_length_7hr, training_mito_length_24hr, 214, 1.23)
hetero_var = np.mean(np.array([h1,h2,h3,h4,h5,h6])**2)

pulse_data_summary_statistics_with_hetero_variance = np.concatenate([pulse_data_summary_statistics_variance, np.array([hetero_var])])

he1_var = hetero_summary_statistic(edu_number_1hr, edu_number_1hr2, mito_length_1hr, mito_length_1hr2, 53,0.06)**2
he3_var = hetero_summary_statistic(edu_number_3hr, edu_number_3hr2, mito_length_3hr, mito_length_3hr2, 42,0.13)**2
he7_var = hetero_summary_statistic(edu_number_7hr, edu_number_7hr2, mito_length_7hr, mito_length_7hr2, 35,0.27)**2
he24_var = hetero_summary_statistic(edu_number_24hr, edu_number_24hr2, mito_length_24hr, mito_length_24hr2, 99,0.51)**2

pulse_data_summary_statistics_with_edu_hetero_variance = np.concatenate([pulse_data_summary_statistics_variance, np.array([hetero_var, he1_var, he3_var, he7_var, he24_var])])

#--------------------------Chase summary statistics------------------------------------------------#

chase_data_summary_statistics = np.array([np.mean(training_edu_number_0dy), np.mean(training_edu_number_1dy),
                                          np.mean(training_edu_number_2dy),np.mean(training_edu_number_4dy),
                                          np.mean(training_dna_number_0dy),np.mean(training_dna_number_1dy),
                                          np.mean(training_dna_number_2dy),np.mean(training_dna_number_4dy),
                                          np.mean(training_edu_number_0dy/training_dna_number_0dy),
                                          np.mean(training_edu_number_1dy/training_dna_number_1dy),
                                          np.mean(training_edu_number_2dy/training_dna_number_2dy),
                                          np.mean(training_edu_number_4dy/training_dna_number_4dy),
                                          0.33, 0.56, 0])

chase_data_summary_statistics_variance = np.array([np.var(training_edu_number_0dy)/len(training_edu_number_0dy),
            np.var(training_edu_number_1dy)/len(training_edu_number_1dy),
            np.var(training_edu_number_2dy)/len(training_edu_number_2dy),
            np.var(training_edu_number_4dy)/len(training_edu_number_4dy),
            np.var(training_dna_number_0dy)/len(training_dna_number_0dy),
            np.var(training_dna_number_1dy)/len(training_dna_number_1dy),
            np.var(training_dna_number_2dy)/len(training_dna_number_2dy),
            np.var(training_dna_number_4dy)/len(training_dna_number_4dy),
            np.var(training_edu_number_0dy/training_dna_number_0dy)/len(training_dna_number_0dy),
            np.var(training_edu_number_1dy/training_dna_number_1dy)/len(training_dna_number_1dy),
            np.var(training_edu_number_2dy/training_dna_number_2dy)/len(training_dna_number_2dy),
            np.var(training_edu_number_4dy/training_dna_number_4dy)/len(training_dna_number_4dy),
            0.002, 0.0103, np.var(data_variance_summary_statistic_matrix)])

chase_data_summary_statistics_val = np.array([np.mean(validation_edu_number_0dy), np.mean(validation_edu_number_1dy),
                                          np.mean(validation_edu_number_2dy),np.mean(validation_edu_number_4dy),
                                          np.mean(validation_dna_number_0dy),np.mean(validation_dna_number_1dy),
                                          np.mean(validation_dna_number_2dy),np.mean(validation_dna_number_4dy),
                                          np.mean(validation_edu_number_0dy/validation_dna_number_0dy),
                                          np.mean(validation_edu_number_1dy/validation_dna_number_1dy),
                                          np.mean(validation_edu_number_2dy/validation_dna_number_2dy),
                                          np.mean(validation_edu_number_4dy/validation_dna_number_4dy),
                                          0.33, 0.56, 0])

chase_data_summary_statistics_with_hetero = np.concatenate([chase_data_summary_statistics, np.array([0])])
chase_data_summary_statistics_with_hetero_variance = np.concatenate([chase_data_summary_statistics_variance, np.array([hetero_var])])

cov_0dy = np.cov(np.array([training_dna_number_0dy,training_edu_number_0dy]))[0,1]
cov_1dy = np.cov(np.array([training_dna_number_1dy,training_edu_number_1dy]))[0,1]
cov_2dy = np.cov(np.array([training_dna_number_2dy,training_edu_number_2dy]))[0,1]
cov_4dy = np.cov(np.array([training_dna_number_4dy,training_edu_number_4dy]))[0,1]

chase_data_covariance_matrix = np.diag(chase_data_summary_statistics_variance)
chase_data_covariance_matrix[0,4] = cov_0dy/len(training_dna_number_0dy)
chase_data_covariance_matrix[4,0] = cov_0dy/len(training_dna_number_0dy)
chase_data_covariance_matrix[1,5] = cov_1dy/len(training_dna_number_1dy)
chase_data_covariance_matrix[5,1] = cov_1dy/len(training_dna_number_1dy)
chase_data_covariance_matrix[2,6] = cov_2dy/len(training_dna_number_2dy)
chase_data_covariance_matrix[6,2] = cov_2dy/len(training_dna_number_2dy)
chase_data_covariance_matrix[3,7] = cov_4dy/len(training_dna_number_4dy)
chase_data_covariance_matrix[7,3] = cov_4dy/len(training_dna_number_4dy)
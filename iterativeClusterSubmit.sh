#!/bin/bash

#Set the name of the job. This will be the first part of the error/output filename.

#$ -N BD_RISE

#Set the shell that should be used to run the job.
#$ -S /bin/bash

#Set the current working directory as the location for the error and output files.
#(Will show up as .e and .o files)
#$ -cwd

#Run parallel experiment in all.q several times until a small number of trials remain
#Select the queue to run in
#$ -q PINC

#Select the number of slots the job will use
#$ -pe smp 2 

#Print informationn from the job into the output file
#/bin/echo Here I am: `hostname`. Sleeping now at: `date`
#/bin/echo Running on host: `hostname`.
#/bin/echo In directory: `pwd`
#/bin/echo Starting on: `date`

#Send e-mail at beginning/end/suspension of job
#$ -m n

#E-mail address to send to
#$ -M joseph-shaffer@uiowa.edu

#Run as Array Job
#$ -t 1:10:1

#Do Stuff

module load stack/2020.2

module load python/3.8.5_gcc-8.4.0
module load py-numpy/1.18.5_gcc-8.4.0
module load py-torch/1.6.0_gcc-8.4.0
module load py-pandas/1.0.5_gcc-8.4.0

module load py-setuptools/50.1.0_gcc-8.4.0
module load py-scikit-learn/0.23.2_gcc-8.4.0


cd /Shared/MRRCdata/Bipolar_R01/scripts/MachineLearning

####### Code for RISE Permutation tests - should be unnecessary for future work #######

#Generate a lot of random RISE trials with the provided model
#python -c "import RISE; input_weights, mask = RISE.run_RISE_Parallel('//Shared/MRRCdata/Bipolar_R01/scripts/MachineLearning/Models/BD_model_247_acc_31.pt', 644, 322, 1, '//Shared/MRRCdata/Bipolar_R01/scripts/MachineLearning/Models/BD_model_247_acc_31.csv', 'Group', $SGE_TASK_ID, 1000, 0.25, 'BD_247_RISE')"

#Code for compiling different numbers of RISE iterations
#python -c "import RISE; out = RISE.run_RISE_Simulation('BD_247_RISE', 10000, 10000,1, 40, 644)"

#Calculate the difference between trials for each number of RISE trials
#python -c "import RISE; out = RISE.calc_RISE_MSD('BD_247_RISE_trials_', 1, 10001, 1, 'BD_247_MAD_Results3.json')"

##### Code for generating BinaryClassifier models and then running RISE on them

#Correct for difference between 0-indexing in python and 1-indexing on cluster iterative submission
model_index=$(( SGE_TASK_ID-1 ))


#### Set parameters here ####
input_file='/Shared/MRRCdata/Bipolar_R01/scripts/MachineLearning/BD_ImagingData_cleaned2.csv'
training_group_size_a=45
training_group_size_b=45
verification_group_size_a=15
verification_group_size_b=15
test_group_size_a=15
test_group_size_b=15
group_name='Group'
first_data_column_name='ctx_rh_posterior_insula'
last_data_column_name='T1rho_wm-rh-parietal'
output_prefix='test'
rise_thresh=0.5
#### End parameter settings - seriously this lower section is flaky and grumpy, be careful if you edit below this line #####

# Create directory for storing output
mkdir -p ./${output_prefix}
#Set name for output file prefix
out_name="./${output_prefix}/${output_prefix}"

#Create string of arguments for inputting to python script
arguments="$input_file $training_group_size_a $training_group_size_b $verification_group_size_a $verification_group_size_b $test_group_size_a $test_group_size_b $group_name $first_data_column_name $last_data_column_name 1000 $out_name $model_index"
#echo $arguments

#### Generate BinaryClassfierModel using wrapper function that reads arguments from BASH arguments ####
python -c "import ML_Binary_Classifier; model, pred, h, fp, fn = ML_Binary_Classifier.generate_BinaryClassifier_model_BASH_wrapper()" $arguments


#### Run RISE on model ####

#Find model file with matching index - this step is necessary because the accuracy and layer counts can vary
fileprefix=$(find ./${output_prefix}/ -type f -iname "${output_prefix}_acc_*_In_*_H_*_Out_*_${model_index}_model.pt")

#Note that if you run this twice and put more than one set of models into the same folder, this step will break because the find will return >1 filenames
fileprefix=${fileprefix%_model.pt}
fileprefix=$fileprefix
#echo $fileprefix

#Create string of arguments to pass to python script
arguments="$fileprefix $first_data_column_name $last_data_column_name $rise_thresh"

# Run RISE on the model using the wrapper class to handle BASH inputs
python -c "import RISE; out = RISE.RISE_Test_Model_BASH_wrapper()" $arguments

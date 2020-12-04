# -*- coding: utf-8 -*-
"""
This program includes the class for a Binary Classifier neural network and several functions that are helpful for using the classifier
Joe Shaffer
March 6, 2020
"""

import torch
import numpy as np
import pandas as pd
import sys
from sklearn import preprocessing

class BinaryClassifier(torch.nn.Module):
    """Create neural network object as a pytorch Module that consists of a linear ANN with one hidden layer"""
    def __init__(self, D_in, H, D_out):
        super(BinaryClassifier, self).__init__()
        #Create first linear transform with input size D_in and output size H
        self.linear1 = torch.nn.Linear(D_in, H)
        #Create second linear transform with input size H and output size D_out
        self.linear2 = torch.nn.Linear(H, H)
        self.layer_out = torch.nn.Linear(H, D_out)
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.batchnorm1 = torch.nn.BatchNorm1d(H)
        self.batchnorm2 = torch.nn.BatchNorm1d(H)
        
    def forward(self, inputs): 
        #First layer with rectifying linear unit (i.e. clamping minimum to 0)
        x = self.relu(self.linear1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.linear2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        y_pred = self.layer_out(x)
        
        return y_pred
    

def generate_balanced_datasets(data, train_a, train_b, test_a, test_b, verification_a, verification_b, var):
    """Function for generating datasets with an equal number of participants in each group by resampling """
    
    #Factorize the category in order to account for text
    temp = pd.factorize(data[var])
    
     
    #Replace column with numeric category label  
    data[var] = temp[0]

    #Create separate lists for each variable category
    a_data = data.loc[data[var]==1]
    b_data = data.loc[data[var]!=1]
    
    #Shuffle lists to randomize them
    shuffled_a = a_data.sample(frac=1).reset_index(drop=True)
    shuffled_b = b_data.sample(frac=1).reset_index(drop=True)
    
    #Extract random testing Sets from data
    if len(shuffled_a) > test_a:
        test_a_data = shuffled_a.loc[range(0,test_a), :]
    else:
        print('Error: Insufficient Participants for Test Set')
    
    
    if len(shuffled_b) > test_b:
        test_b_data = shuffled_b.loc[range(0,test_b), :]
    else:
        print('Error: Insufficient Participants for Test Set')
        
     #Extract verification sets from remaining data
     
    if (len(shuffled_a)-test_a) >= verification_a:
        verification_a_data = shuffled_a.loc[range(test_a, test_a + verification_a), :]
        
    else: 
        print('Error: Insufficient Participants for Verification Set')
        
    if (len(shuffled_b)-test_b) >= verification_b:
        verification_b_data = shuffled_b.loc[range(test_b, test_b + verification_b), :]
        
    else: 
        print('Error: Insufficient Participants for Verification Set')   
        
    #Extract training sets from remaining data
    if (len(shuffled_a) - test_a - verification_a) >= train_a:
        train_a_data = shuffled_a.loc[range(test_a + verification_a, test_a + verification_a + train_a), :]
        
    else:
        remaining_a = shuffled_a.loc[range(test_a + verification_a, len(shuffled_a)), :]
        part1 = remaining_a
        part2 = remaining_a.sample(n=(train_a-len(remaining_a)), replace=True).reset_index(drop=True)
        
        train_a_data = pd.concat([part1, part2], axis=0)
        train_a_data = train_a_data.sample(frac=1).reset_index(drop=True)
    
    if (len(shuffled_b) - test_b - verification_b) >= train_b:
        train_b_data = shuffled_b.loc[range(test_b + verification_b, test_b + verification_b + train_b), :]
        
    else:
        remaining_b = shuffled_b.loc[range(test_b + verification_b, len(shuffled_b)), :]
        
        part1 = remaining_b
        part2 = remaining_b.sample(n=(train_b-len(remaining_b)), replace=True).reset_index(drop=True)
        
        train_b_data = pd.concat([part1, part2], axis=0)
        train_b_data = train_b_data.sample(frac=1).reset_index(drop=True)

    
    test_data = pd.concat([test_a_data, test_b_data], axis=0)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    
    train_data = pd.concat([train_a_data, train_b_data], axis=0)
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    verification_data = pd.concat([verification_a_data, verification_b_data], axis = 0)
    verification_data = verification_data.sample(frac=1).reset_index(drop=True)
    
    return train_data, test_data, verification_data
    
def generate_random_datasets(data, n, m, var):
    """ This function generates a random sub-sample with n samples from category 0 and m samples from category 1 of a binary classifier, var"""
    #Shuffle the dataset rows, reset the index
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    
    #Convert categorical labels to numeric values. For group, 0=Case, 1 = Control
    temp = pd.factorize(shuffled_data[var])
    
    #Initialize variables
    labels = temp[0]
    ncount = 0
    mcount = 0
    train_list = []
    test_list = []
    
    #Loop through each row
    for i in range(data.shape[0]):
        #print(labels[i])
        
        #Select n rows with value = 0
        if (labels[i] == 0):
            if (ncount < n):
                test_list.append(i)
                ncount += 1
            else:
                train_list.append(i)
        #Select m rows with value = 1        
        elif(labels[i] == 1):
            if (mcount < m):
                test_list.append(i)
                mcount += 1
            else:
                train_list.append(i)
    
    test_data = shuffled_data.loc[test_list, :]
    train_data = shuffled_data.loc[train_list,:]
    return train_data, test_data

def get_accuracy_metrics(true_array,pred_array):
    """Calculates hits, false positives, and false negatives by comparing known and prediction """
    if(true_array.shape == pred_array.shape):
        test_examples = true_array.shape

        h = np.zeros((true_array.shape), dtype=float)
        fp = np.zeros((true_array.shape), dtype=float)
        fn = np.zeros((true_array.shape), dtype=float)

        for s in range(test_examples[0]):
            #print(str(s), ': ', str(true_array[s]), ': ', str(pred_array[s]))
            if(true_array[s] == 0 and pred_array[s] == 0):
                h[s] = 1
               
            elif(true_array[s] == 0 and pred_array[s] == 1):
                fp[s] = 1
            elif(true_array[s] == 1 and pred_array[s] == 1):
                h[s] = 1
            elif(true_array[s] == 1 and pred_array[s] == 0):
                fn[s] = 1
            
        return h,fp,fn    
    else:
        print ('dimension mismatch')
        return -1, -1, -1

def binary_acc(y_pred, y_test):
    """Function for taking the model prediction, turning it into a binary value, and comparing it with the test value. Returns accuracy as a percentage"""
    
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    
    correct_results_sum = (y_pred_tag == y_test). sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc*100)
    return acc

def train_classifier_model(model, input_tensor, ans_tensor, epochs):
    """ Function for performing training on the Binary Classifier """
    #Set error function
    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCEWithLogitsLoss()

    #Set training algorithm and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    #Set model to training mode
    model.train()
    
    #print(str(input_tensor.shape))
    #print(str(ans_tensor.shape))
    
    #Loop through training cycles
    for t in range(epochs):
        #Get prediction from neural network
        ans_pred = model(input_tensor.float())
                
        ans_pred = torch.Tensor(ans_pred)
        
        #Calculate error & print result so we can see improvement
        loss = criterion(ans_pred, ans_tensor.float())
            
        #Clear the gradients in the optimizer
        optimizer.zero_grad()
        #Backpropagate to train network
        loss.backward()
        ##torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    
    return model, ans_pred, loss

def load_input_data(data_file, test_var_label, data_start_column_label, data_end_column_label):
    """ Loads input and answer data from saved .csv file and generates tensors for input into model """
    
    all_data = pd.read_csv(data_file)
    
    #Get column with variable of interest as answer and convert to tensor
    ans = pd.factorize(all_data[test_var_label])
    ans_tensor = torch.from_numpy(ans[0]).unsqueeze(1)
    
    
    #Get range of imaging data from input file
    imaging_variables = all_data.loc[:, data_start_column_label:data_end_column_label]
    
    #Normalize each column of imaging data and convert to tensor
    scaler = preprocessing.StandardScaler()
    imaging_data = pd.DataFrame(scaler.fit_transform(imaging_variables.values))
    input_tensor = torch.tensor(imaging_data.values)
    
    return input_tensor, ans_tensor
    
def load_model(filename, D_in, H, D_out):
    """ Function for loading a saved model file and rebuilding it into a pyTorch model"""
    #Initialize empty NeuralNet object
    model = BinaryClassifier(D_in, H, D_out)
    
    #Load torch file
    pretrained_model = torch.load(filename)
    
    #Load values of previously saved model into new model object & return that model
    trimmed_model = {k: v for k, v in pretrained_model.items() if k in model.state_dict()}
    model.load_state_dict(trimmed_model)
    model.eval()
    return model

def select_key_features(data_file, sorting_output, outfilename):
    """ Function for taking the sorted output of the find_Consensus_Results function and splitting those results into a new .csv containing only better than average features """
    
    data_df = pd.read_csv(data_file)
    
    t = sorting_output.sum() > 0
    
    name_list = []
    for col in t.index:
        if not t.loc[col]:
            name_list.append(col)
        
    less_data = data_df.drop(name_list, axis=1)
    less_data.to_csv(outfilename, index=False)
    
    return less_data

def generate_BinaryClassifier_model_BASH_wrapper():
    #print(str(len(sys.argv)))
        
    for i in range(0,len(sys.argv)):
        print("arg: ", str(i), ": ", str(sys.argv[i]), ':', str(type(sys.argv[i])))
    
    data_file=sys.argv[1]
    train_group_1_num=int(sys.argv[2])
    train_group_2_num=int(sys.argv[3])
    test_group_1_num=int(sys.argv[4])
    test_group_2_num=int(sys.argv[5])
    verification_group_1_num=int(sys.argv[6])
    verification_group_2_num=int(sys.argv[7])
    test_var_label=sys.argv[8]
    data_start_column_label=sys.argv[9]
    data_end_column_label=sys.argv[10]
    training_cycles=int(sys.argv[11])
    outprefix=sys.argv[12]
    iteration=int(sys.argv[13])
    
    
    
    model, pred, h, fp, fn = generate_BinaryClassifier_model(data_file, train_group_1_num, train_group_2_num, test_group_1_num, test_group_2_num, verification_group_1_num, verification_group_2_num, test_var_label, data_start_column_label, data_end_column_label, training_cycles, outprefix, iteration)

    #model, pred, h, fp, fn = generate_BinaryClassifier_model(str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), str(sys.argv[8]), str(sys.argv[9]), str(sys.argv[10]), int(sys.argv[11]), str(sys.argv[12]), int(sys.argv[13]))
    
    return 1, 1, 1, 1, 1
    #return model, pred, h, fp, fn

def generate_BinaryClassifier_model(data_file, train_group_1_num, train_group_2_num, test_group_1_num, test_group_2_num, verification_group_1_num, verification_group_2_num, test_var_label, data_start_column_label, data_end_column_label, training_cycles, outprefix, iteration):
    """This function creates a binary classifier model by generating a random training, verification, and test set and then training/testing a binary classifier model """
    
    print('RUNNING')
    
    #Read Data File
    all_data = pd.read_csv(data_file)
    
    #Generate random training and test datasets    
    train_data, test_data, verification_data = generate_balanced_datasets(all_data, train_group_1_num, train_group_2_num, test_group_1_num, test_group_2_num, verification_group_1_num, verification_group_2_num, test_var_label)

    #Convert categorical labels for variable of interest to numeric values. For group, 0=Case, 1 = Control
    train_group = pd.factorize(train_data[test_var_label])
    test_group = pd.factorize(test_data[test_var_label])
    
    #Create tensors from variables of interest
    train_ans_tensor = torch.from_numpy(train_group[0]).unsqueeze(1)
    test_ans_tensor = torch.from_numpy(test_group[0]).unsqueeze(1)
  
    print(data_start_column_label, ':', data_end_column_label)
    #Get imaging variable portion of the training and test data to input to NN
    train_imaging_variables= train_data.loc[:, data_start_column_label:data_end_column_label]
    test_imaging_variables = test_data.loc[:, data_start_column_label:data_end_column_label]

    #Normalize each column of training and test data
    scaler = preprocessing.StandardScaler()
    train_imaging_data = pd.DataFrame(scaler.fit_transform(train_imaging_variables.values))
    test_imaging_data = pd.DataFrame(scaler.fit_transform(test_imaging_variables.values))

    #Convert dataframes to tensors
    train_input_tensor = torch.tensor(train_imaging_data.values)
    test_input_tensor = torch.tensor(test_imaging_data.values)

    #Set size of input to neural network as number of columns in input_tensor
    temp = [*train_input_tensor.size()]
    D_in = temp[1]
 
    #We are attempting to predict a binary category, so we only need one output node
    D_out = 1

    #Set size of hidden layer based on size of inputs and outputs
    H = int(np.floor((D_in + D_out)/2))

    #device = torch.device("cpu")
    #Define Neural Network based on input size
    model = BinaryClassifier(D_in, H, D_out)
    
    #Train Predictor Model
    model, ans_pred, loss = train_classifier_model(model, train_input_tensor, train_ans_tensor, training_cycles)
    
    #Test Model Prediction on test set
    model.eval()
    with torch.no_grad():
        #Get prediction from neural network
        test_ans_pred = model(test_input_tensor.float())

    #Test accuracy of model
    test_ans = test_ans_tensor.detach().numpy() 
    binary_test_pred = torch.round(torch.sigmoid(test_ans_pred)).detach().numpy()
    
    h,fp,fn = get_accuracy_metrics(binary_test_pred,test_ans)

    #Store model, test_data
    model_filename = outprefix + '_acc_' + str(round(float(np.mean(h)), 3)) + '_In_' + str(D_in) + '_H_'+ str(H) + '_Out_' + str(D_out) + '_' + str(iteration) + '_model.pt'
    print(model_filename)
    torch.save(model.state_dict(), model_filename)
    data_filename = outprefix + '_acc_' + str(round(float(np.mean(h)), 3)) + '_In_' + str(D_in) + '_H_'+ str(H) + '_Out_' + str(D_out) + '_' + str(iteration) + '_train_data.csv'
    train_data.to_csv(data_filename, index = False)
    data_filename = outprefix + '_acc_' + str(round(float(np.mean(h)), 3)) + '_In_' + str(D_in) + '_H_'+ str(H) + '_Out_' + str(D_out) + '_' + str(iteration)+ '_test_data.csv'
    test_data.to_csv(data_filename, index = False)

    #If a verification set is generated, also test verification data set
    if (len(verification_data) > 0):
        
        #Select predicted category and create tensor for expected values
        verification_group = pd.factorize(verification_data[test_var_label])
        verification_ans_tensor = torch.from_numpy(verification_group[0]).unsqueeze(1)
        
        #Select imaging data columns
        verification_imaging_variables = verification_data.loc[:, data_start_column_label:data_end_column_label]
        
        #Normalize imaging data
        verification_imaging_data = pd.DataFrame(scaler.fit_transform(verification_imaging_variables.values))
        
        #Convert to tensor
        verification_input_tensor = torch.tensor(verification_imaging_data.values)

        #Test Model Prediction on test set
        model.eval()
        with torch.no_grad():
            #Get prediction from neural network
            verification_ans_pred = model(verification_input_tensor.float())

        #Test accuracy of model
        verification_ans = verification_ans_tensor.detach().numpy() 
        binary_verification_pred = torch.round(torch.sigmoid(verification_ans_pred)).detach().numpy()
        
        verification_h,verification_fp,verification_fn = get_accuracy_metrics(binary_verification_pred,verification_ans)

        data_filename = outprefix + '_acc_' + str(round(float(np.mean(h)), 3)) + '_ver-acc_' + str(round(float(np.mean(verification_h)), 3)) + '_In_' + str(D_in) + '_H_'+ str(H) + '_Out_' + str(D_out) + '_' + str(iteration)+ '_verification_data.csv'
        test_data.to_csv(data_filename, index = False)
    
    return model, binary_test_pred, h, fp, fn




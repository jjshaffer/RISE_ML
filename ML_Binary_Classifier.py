# -*- coding: utf-8 -*-
"""
This program includes the class for a Binary Classifier neural network and several functions that are helpful for using the classifier
Joe Shaffer
March 6, 2020
"""

import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing

#Create neural network object as a pytorch Module that consists of a linear ANN with one hidden layer
class BinaryClassifier(torch.nn.Module):
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
    
    #Loop through training cycles
    for t in range(epochs):
        #Get prediction from neural network
        ans_pred = model(input_tensor.float())
                
        #Calculate error & print result so we can see improvement
        loss = criterion(ans_pred, ans_tensor.float())
            
        #Clear the gradients in the optimizer
        optimizer.zero_grad()
        #Backpropagate to train network
        loss.backward()
        optimizer.step()
        
    return model, ans_pred, loss

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



def generate_BinaryClassifier_model(data_file, train_group_1_num, train_group_2_num, test_var_label, data_start_column_label, data_end_column_label, training_cycles, outprefix):
    
    #Read Data File
    all_data = pd.read_csv(data_file)
    
    #Generate random training and test datasets    
    train_data, test_data = generate_random_datasets(all_data, train_group_1_num, train_group_2_num, test_var_label)

    #Convert categorical labels for variable of interest to numeric values. For group, 0=Case, 1 = Control
    train_group = pd.factorize(train_data[test_var_label])
    test_group = pd.factorize(test_data[test_var_label])
   
    #Create tensors from variables of interest
    train_ans_tensor = torch.from_numpy(train_group[0]).unsqueeze(1)
    test_ans_tensor = torch.from_numpy(test_group[0]).unsqueeze(1)

    #Get imaging variable portion of the training and test data to input to NN
    train_imaging_variables=train_data.loc[:, data_start_column_label:data_end_column_label]
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
    model_filename = outprefix + '_acc_' + str(float(np.mean(h))) + '_model.pt'
    torch.save(model.state_dict(), model_filename)
    data_filename = outprefix +  '_train_data.csv'
    train_data.to_csv(data_filename, index = False)
    data_filename = outprefix +  '_test_data.csv'
    test_data.to_csv(data_filename, index = False)

    return model, binary_test_pred, h, fp, fn




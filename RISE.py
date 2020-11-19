# -*- coding: utf-8 -*-
"""
Functions for running RISE to discern the features that were used by a machine learning algorithm to make its decisions
Created on Wed May 13 10:48:53 2020

@author: JJ Shaffer
"""

import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import json
import os
import re

import ML_Binary_Classifier

def combine_RISE_Slices(fileprefix, iterations):
    """ Function for loading set of iterations of RISE slices and combining them - currently incomplete """    
    for i in range(iterations):
        filename = fileprefix + '_slice_' + str(iterations) + '_part0_results.csv'
        
        if filename.isfile():
            print('Loading ', filename)
        
        filename = fileprefix + '_slice_' + str(iterations) + '_part0_mask.csv'
        if filename.isfile():
            print('Loading ', filename)

def run_RISE_Slice(model, input_tensor, ans_tensor, iteration, mask_p, prefix, save_flag):
    """Function for performing a single iteration of the RISE test"""
    
    img_size = [*input_tensor.size()]
    
    #Generate Mask
    mask = generate_binary_mask(1,img_size[1], mask_p)

    masked_input = []
    #Loop through each participant and mask input_tensor
    for j in range(img_size[0]):
        #Multiply input data by mask
        masked_input.append(input_tensor[j,:].numpy() * mask)
        
    #Run neural network on masked data
    masked_input_tensor = torch.tensor(masked_input).float().squeeze()
    test_pred = model(masked_input_tensor)
    
    #Fit results to sigmoid function and round to binary predictor
    model_pred_np = torch.round(torch.sigmoid(test_pred.detach())).numpy().squeeze()
           
    #Calculate error
    ans_np = ans_tensor.numpy().squeeze()
    error = 1 - abs(ans_np - model_pred_np)
           
    #Multiply Mask weights x error for each person
    error_weighted_masks = np.zeros((img_size[0], img_size[1]), dtype=float)
    for j in range(img_size[0]):
        error_weighted_masks[j,:] = error[j] * mask
    

    if save_flag == 1:
        filename = prefix + '_slice_' + str(iteration) + '_results.csv'
        print('Saving ', filename)
        pd.DataFrame(error_weighted_masks).to_csv(filename, index=False)
        
        filename = prefix + '_slice_' + str(iteration) + '_mask.csv'
        print('Saving ', filename)
        pd.DataFrame(mask).to_csv(filename, index=False)
    
    return error_weighted_masks, mask

    

def RISE_Test(model, input_tensor, ans_tensor, iterations, mask_p):
    """Function for running RISE"""    
    img_size = [*input_tensor.size()]
    
    #Allocate space to store weighted sum for each feature and number of trials it was included in
    Cumulative_Input_Weights = np.zeros((img_size[0], img_size[1]), dtype = float)
    Cumulative_Mask = np.zeros((1, img_size[1]), dtype = float)
    
    for i in range(iterations):
        
        if (i%1000 == 0):
            print("Performing RISE Trial: ", i)
        
        #Generate mask using bilinear interpolation
        #The commented-out line below generates a 2d mask - the current problem is one line per participant, so we want a 1d mask
        #Mask.append(generate_mask(int(np.ceil(img_size[0]/3)),int(np.ceil(img_size[1]/3)), img_size[0],img_size[1],mask_p))
        #Mask.append(generate_mask(1,int(np.ceil(img_size[1]/3)), 1,img_size[1],mask_p))

        mask = generate_binary_mask(1,img_size[1], mask_p)
        
        Cumulative_Mask = Cumulative_Mask + mask
        #Mask.append(generate_binary_mask(1, img_size[1], mask_p))
        #print(mask)
        #Mask input tensor
        masked_input = []
        #Loop through each participant
        for j in range(img_size[0]):
            #Multiply input data by mask
            masked_input.append(input_tensor[j,:].numpy() * mask)

        #Store array of masked inputs
        #Masked_Input.append(masked_input)
        
        #Run neural network on masked data
        temp = torch.tensor(masked_input).float().squeeze()
        test_pred = model(temp)
        #print(test_pred.detach().numpy())
        model_pred_np = torch.round(torch.sigmoid(test_pred.detach())).numpy().squeeze()
        #Model_Pred.append(model_pred_np)
        
        #Calculate error
        ans_np = ans_tensor.numpy().squeeze()
        error = 1 - abs(ans_np - model_pred_np)
        #Model_Error.append(error)
        
        #Multiply Mask weights x error for each person
        error_weighted_masks = np.zeros((img_size[0], img_size[1]), dtype=float)

        for j in range(img_size[0]):
            #print(i,j, error[j])
            error_weighted_masks[j,:] = error[j] * mask
            Cumulative_Input_Weights[j,:] = Cumulative_Input_Weights[j,:] + error_weighted_masks[j,:]

        #Mask_Error_Product.append(error_weighted_masks)
    return Cumulative_Input_Weights, Cumulative_Mask

def combine_RISE_results(filename, maskfilename, thresh_flag):
    #Read raw RISE scores
    RISE_data = pd.read_csv(filename)
    
    MASK_data = pd.read_csv(maskfilename)
    
    Num_Trials = MASK_data * RISE_data.shape[0]
    print(Num_Trials)
    
    #Sum up each column
    totals = RISE_data.sum(axis = 0)
    #Trim subject ID    
    totals = totals[1::]
    
    #print(totals.values)
    RISE_Accuracy = totals.values/Num_Trials
    print(RISE_Accuracy)
    
    RISE_Mean = totals.mean(axis = 0)
    RISE_STD = totals.std(axis=0)

    #print(RISE_Mean, RISE_STD)
    DeMeaned_RISE_Results = totals - RISE_Mean
    z_score_RISE_Results = DeMeaned_RISE_Results/RISE_STD
    
    if thresh_flag == 1:
        thresholded_z_score = np.where(z_score_RISE_Results > 1.96, z_score_RISE_Results, 0)
    
    else:
        thresholded_z_score = z_score_RISE_Results

    

    frame = {'Total': totals, 'DeMeaned': DeMeaned_RISE_Results, 'z_score': z_score_RISE_Results, 'thresholded': thresholded_z_score, 'Accuracy': RISE_Accuracy.values.squeeze()}
    
    totals_df = pd.DataFrame(frame)
    #print(DeMeaned_RISE_Results, z_score_RISE_Results)
    
    return totals_df
    #return RISE_Accuracy


""" Functions for Permutation Testing RISE """

#Function for performing the first step of the RISE function in parallel. Each instance generates 1000 masks and uses the model to produce 1000 predictions
def run_RISE_Parallel(model_name, D_in, H, D_out, data_file, var_name, iteration, cycles, mask_p, outprefix):
    model = ML_Binary_Classifier.load_model(model_name, D_in, H, D_out)
    input_tensor, ans_tensor, subjIDs = ML_Binary_Classifier.load_sorted_test_data(data_file, var_name)
    
    RISE_results = []
    RISE_masks = []
    
    part = 0
    if not (os.path.exists(outprefix + '_slice_' + str(iteration) + '_part_' + str(part) + '_results.json') and os.path.exists(outprefix + '_slice_' + str(iteration) + '_part_' + str(part) + '_masks.json')):
    
        for i in range(cycles):
        
            input_weights, mask = run_RISE_Slice(model, input_tensor, ans_tensor, iteration, mask_p, outprefix, 0)
    
            RISE_results.append(input_weights.tolist())
            RISE_masks.append(mask.tolist())
    
            #Break Results into groups of 1000
            if i>0 and i%1000 == 0:
                filename = outprefix + '_slice_' + str(iteration) + '_part_' + str(part) + '_results.json' 
                print('Saving ' +filename)
                with open(filename, 'w') as f:
                    json.dump(RISE_results, f)
            
            
                filename = outprefix + '_slice_' + str(iteration) + '_part_' + str(part) + '_masks.json' 
                print('Saving ' + filename)
                with open(filename, 'w') as f:
                    json.dump(RISE_masks, f)
            
            #part+=1
                RISE_results = []
                RISE_masks = []
            
        #Print remaining results
        if len(RISE_results) > 0: 
            filename = outprefix + '_slice_' + str(iteration) + '_part_' + str(part) + '_results.json' 
            print('Saving ' +filename)
            with open(filename, 'w') as f:
                json.dump(RISE_results, f)
                f.close()
        
            filename = outprefix + '_slice_' + str(iteration) + '_part_' + str(part) + '_masks.json' 
            print('Saving ' + filename)
            with open(filename, 'w') as f:
                json.dump(RISE_masks, f)
                f.close()
        
    return RISE_results, RISE_masks

#Function for running permutation testing of RISE method using the output of run_RISE_Parallel. This creates files of up to 1000
def run_RISE_Simulation(fileprefix, file_index_max, num_iterations, step_size, subjects, features):
    
    file_list = random.sample(range(1,file_index_max+1), file_index_max)
    results = np.zeros((subjects, features), dtype=float)
    
    #Initialize counter and load first file
    count = 0
    filename = fileprefix + '_slice_' + str(file_list[count]) + '_part_0_results.json'
    #filename = fileprefix + '_slice_1_part_0_results.json'

    if os.path.isfile(filename):
        with open(filename) as f:
            r = json.load(f)
            f.close()
    else:
        print('Missing file: ' + filename)
        return 0
    
    out_step_count = 1
    
    #Loop through desired number of total iterations        
    for n in range(0,num_iterations, 1): 
        #Every 1000 iterations, step to next input file and open it
        if n%1000 == 0 and n>0:        
            count = count + 1
            filename = fileprefix + '_slice_' + str(file_list[count]) + '_part_0_results.json'
            if os.path.isfile(filename):
                with open(filename) as f:
                    r = json.load(f)
                    f.close()
            else:
                print('Missing file: ' + filename)
                return 0
        
        #Add successive elements of r to results
        results = results + r[n%1000]
        #print(str(n%1000))
        
        #Save every step_size iterations
        if n%step_size==0:
     
            print('N:' + str(n) + ' Count:' + str(count) + ' inFile:' + str(file_list[count]))
            outfilename = fileprefix + '_trials_' + str(out_step_count*step_size)+ '.json'
            results_list = results.tolist()
            with open(outfilename, 'w') as f:
                json.dump(results_list, f)
                f.close()
            out_step_count=out_step_count +1
      
        
    return results


#def run_RISE_Simulation_Experiment(fileprefix, start_range, end_range, step_size, subjects, features, i):
    
#    trial_list = range(start_range,end_range, step_size)
    
#    prev_trial = np.zeros((subjects, features), dtype=float)
    
    #for n in trial_list:
#    n = trial_list[i]
    
#    out = run_RISE_Simulation(fileprefix,n, max_iterations, subjects, features)
    

#    filename = fileprefix + '_trials_'+str(n)+'.json'    
#    out = out.tolist()
#    with open(filename, 'w') as f:
#        json.dump(out, f)
#        f.close()

#    return out

#Function for compiling distributions of model accuracies
    
def compile_subject_frequencies(model_path):
    #Get list of files in directory
    file_list = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]
    
    subj_list = dict()
    max_trials = 0;
    
    for i in range(0,len(file_list)):
        if re.search(r"test\_data\.csv", file_list[i]):
            print(file_list[i])
            
            tmp = file_list[i].split('_')
            for j in range(0,len(tmp)):
                if tmp[j]== 'acc':
                    acc = float(tmp[j+1])
                    #print(str(acc))
                    
            
            #Open file as dataframe
            file_name = model_path + '/' + file_list[i]
            file_data = pd.read_csv(file_name)
            id_list = file_data['Participant_ID']
            
            for j in range(0,len(id_list)):
                
                if id_list[j] in subj_list:
                    #Add to count, weighted sum
                    temp = subj_list.get(id_list[j])
                    
                    temp[0]+= 1
                    temp[1]+= acc
                    
                    if temp[0] > max_trials:
                        max_trials = temp[0]
                    
                    temp.append(acc)
                    
                    subj_list[id_list[j]]= temp
                    #acc=[1,acc]
                    
                    #temp = [temp[i] + acc[i] for i in range(len(temp))]
                    
                    subj_list[id_list[j]] = temp
                else:
                    #Create key
                    subj_list[id_list[j]] = [1, acc, acc]
                    
    #print(str(max_trials))
    
    subj_data = pd.DataFrame(np.zeros((len(subj_list), max_trials+4), dtype=float))
    key_list = list(subj_list)
    #count = 0
    for i in range(0,len(subj_data)):
        subj_data.iloc[i,0] = str(key_list[i])
        
        t = subj_list[key_list[i]]
        subj_data.iloc[i,1] = t[2]/t[1]
        for j in range(0,t[0]):
            subj_data.iloc[i,j+2] = t[j]
        
        
        #count +=1
                    
    return subj_data

def sort_by_index(a,b, a_col):
    
    for i in range(0,len(a)):
        for j in range(0,len(b)):
            if a.iloc[i,0] == b.iloc[j,0]:
                #print(str(a.iloc[i,0]))
                a.iloc[i,a_col] = b.iloc[j,1]
                
    return a

def calc_z_normalized_df(t, col_range):
    
    tz = pd.DataFrame(np.zeros(t.shape, dtype=float))
    
    tz.iloc[:,0] = t.iloc[:,0]
    tz.columns = t.columns
    for i in col_range:
        a = t.iloc[:,i].to_numpy()
        b = a.std(axis=0)
        c = a.mean(axis=0)
        tz.iloc[:,i] = (a-c)/b
        
    return tz
        

def graph_subject_influence(t, num_trials, x_labels, outprefix):
    
    x = range(0,num_trials)
    
    for i in range(len(t)):
        plt.plot(x, t.iloc[i, range(1,num_trials+1)])
        
    plt.xlabel("Features", size=14)
    plt.ylabel("Mean Effect", size=14)
    plt.title("Influence of Training Group Composition")
    #plt.legend(loc='upper right')
    plt.xticks(x, x_labels)    
    
    plt.show()
    
    outfilename = outprefix + '.jpg'
        
    plt.savefig(outfilename, dpi = 300)


def compile_model_accuracies(model_path, num_iters):
    
    #Get list of files in directory
    file_list = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]

    
    acc = np.zeros((num_iters,1), dtype=float)

    count=0
    for i in range(0,len(file_list)):
        if re.search(r"model\.pt", file_list[i]):
            #print(file_list[i])
           
            tmp = file_list[i].split('_')
            for j in range(0,len(tmp)):
                if tmp[j]== 'acc':
                    acc[count] = float(tmp[j+1])
                    #print(str(count), ':', str(acc[count]))
            count += 1

    return acc

def graph_accuracy_histogram(acc_list, label_list, list_range, outprefix):
    
    #a, b = np.histogram(acc_list, bins=10, range=(0,1))
    
    if len(acc_list) == len(label_list):
        
        for i in list_range:
                plt.hist(acc_list[i], bins=10, range = (0,1), alpha = 0.25, label=label_list[i])


        plt.xlabel("Accuracy", size=14)
        plt.ylabel("Count", size=14)
        plt.title("Model accuracy distribution")
        plt.legend(loc='upper right')
        
        plt.show()
    
        outfilename = outprefix + '.jpg'
        
        plt.savefig(outfilename, dpi = 300)
        
    else:
         print('Size of lists do not match')
         

    acc_df = pd.DataFrame(np.squeeze(acc_list))
    
    acc_df.index = label_list

    return acc_df

#Function for calculating the Mean Squared Error (or Mean Absolute Error) between subsequently increasing numbers of RISE trials. The goal is to test whether the models converge.
def calc_RISE_MSD(prefix, start_range, end_range, step_size, outfile):
    sizes = range(start_range, end_range, step_size)
    
    #Load the first file
    file2 = prefix + str(sizes[0]) + '.json'
    with open(file2, 'r') as f:
        file2_data = json.load(f)
        f.close()

    results = np.zeros((len(sizes)-1, len(file2_data[0])), dtype=float)
    #Loop through each remaining file    
    for i in range(1,len(sizes) - 1):
        
        #Set previous "second" file as "first" file
        file1_data = file2_data
        
        #Load next file as "second" file
        file2 = prefix + str(sizes[i]) + '.json'
        with open(file2, 'r') as f:
            file2_data = json.load(f)
            f.close()
        
        #Calculate difference between first and second maps
        for x in range(0,len(file1_data)):
            for y in range(0, len(file1_data[0])):
                if(file1_data[x][y] != 0):
                    #Calculate percent difference for each feature
                    results[i][y] = results[i][y] + np.abs(file2_data[x][y] - file1_data[x][y])/file1_data[x][y]
                
          
    #Save matrix of changes & return results
    with open(outfile, 'w') as f:
        json.dump(results.tolist(), f)
        f.close()
    return results


""" Utility Functions """

def linear_interp(x,x0, y0, x1, y1):
    """Function for performing linear interpolation"""
    return y0*(1-(x-x0)/(x1-x0)) + y1*((x-x0)/(x1-x0))


def bilinear_interp(x,y, x1, y1, x2,y2, Q11, Q12, Q21, Q22):
    """Function for performing bilinear interpolation for coordinate x,y based on coordinates and values of surrounding points"""    
    return 1/((x2-x1)*(y2-y1)) * np.matmul(np.matmul([x2-x, x-x1], [[Q11, Q12], [Q21, Q22]]), [[y2-y], [y-y1]])

def generate_binary_mask(h,w,p):
    """Function for generating a random binary mask of size h x w with probability p"""
    #Create empty mask of size h x w
    mask = np.zeros((h,w), dtype=float)
    #Loop through each element in mask and independently assign masked values with probability p
    for i in range(h):
        for j in range(w):
            if p > random.random():
                mask[i,j] = 1
    return mask             
                
def generate_mask(h,w,H,W,p):
    """Function for creating a random mask by randomly generating an h x w mask and expanding it to an H x W using bilinear interpolation"""
    #Create empty mask of size h x w
    mask = np.zeros((h+1,w+1), dtype=float)
    #Loop through each element in mask and independently assign masked values with probability p
    for i in range(h):
        for j in range(w):
            if p < random.random():
                mask[i,j] = 1


    #Pad out of range row and column by repeating last row and column
    mask[h,:] = mask[h-1,:]
    mask[:,w] = mask[:,w-1]
    
    
    #Use bilinear interpolation to expand mask
    
    #Calculate initial step size
    Ch = H/h
    Cw = W/w
    
    #Calculate scale ratio  
    h1 = (h+1)*Ch
    w1 = (w+1)*Cw
    
    #Convert step to integer     
    i1 = int(np.floor(h1))
    j1 = int(np.floor(w1))
    
    #Generate oversized image to be cropped later
    M = np.zeros((i1,j1), dtype=float)
    
    for i in range(i1):
        #Identify neighboring x coords
        x = i/(h1/h)
        x1 = int(np.floor(i/(h1/h)))
        x2 = x1+1
        #print (x, x1)
        for j in range(j1):
            #print(y)
            
            #Identify neighoring y coords
            y = j/(w1/w)
            y1 = int(np.floor(j/(w1/w)))
            y2 = y1+1
            
            #print(x, ',', y, ': x1:', x1, ', y1:', y1, ', x2:',x2, ', y2:',y2)
                       
            #Perform bilinear interpolation to fit to larger mask size
            M[i,j] = bilinear_interp(x,y, x1, y1, x2,y2, mask[x1,y1], mask[x1,y2], mask[x2,y1], mask[x2,y2])
           
    
    #Randomly crop mask
    x_offset = random.randint(0,i1-H)
    y_offset = random.randint(0,j1-W)
    #print(x_offset, y_offset)
    
    Mask = M[x_offset:x_offset+H, y_offset:y_offset+W]
       
    return Mask

def find_Consensus_Results(filepath, filelist, col_names, outfileprefix):
    
    results = []
    
    #fig, ax0 = plt.subplots(figsize=(48.0, 10.0))
    #ax0.set_title('RISE Effect of Each Feature Across Models:')
    #ax0.set_ylabel('Mean z-stat')
    #ax0.set_xlabel('Imaging Measure')
    
    #plt.xticks(rotation = 'vertical', fontsize = 4)
    
    #plt.subplots_adjust(bottom=0.5)
    
    #Loop through each file in list
    for i in range(0,len(filelist)):
        print(filelist[i])
    
        #Open file to retreive data
        #filename = filepath + filelist[i] + '_RISEResults.csv'
        filename = filepath + filelist[i] + '_RISEResults'
        data = pd.read_csv(filename)
    
        #Normalize data and convert to z-statistic
        z_data = calc_RISE_zStat(data)
        
        #Calculate the mean magnitude of the z-statistic for each region
        #sum_list = np.mean(np.abs(z_data), axis=0)
        sum_list = np.mean(z_data, axis=0)

        
        results.append(sum_list)
        
      
        #if i == 0:
         #   p1 = ax0.bar(col_names, sum_list)
          #  prev_list = sum_list
        #else:
         #   p1 = ax0.bar(col_names, sum_list, bottom=prev_list)
          #  prev_list = sum_list


    
    results_df = pd.DataFrame(data=results, columns=col_names)
    
    results_sum = np.mean(results, axis=0)
    results_sum = results_sum.reshape(1, len(results_sum))
    results_sum = pd.DataFrame(data=results_sum, columns=col_names)

    sorted_sum = results_sum.sort_values(by=results_sum.index[0], axis=1)
    sorted_results = results_df[sorted_sum.columns]

    transpose_results = sorted_results.transpose()
    
    
    
    ax0 = transpose_results.plot(kind="bar", stacked=True, figsize=(48.0, 10.0), title='RISE Effect of Each Feature Across Models')
    ax0.set_ylabel('Cumulative z-stat')
    ax0.set_xlabel('Imaging Measure')
    
    plt.xticks(rotation = 'vertical', fontsize = 3)
    plt.subplots_adjust(bottom=0.25)
    
    plt.show()
    outfilename = outfileprefix + '.jpg'
    fig = ax0.get_figure()
    fig.savefig(outfilename, dpi = 300)
    
    return sorted_results, col_names, transpose_results

def RISE_Test_Model(fileprefix, data_start_column_label, data_end_column_label, acc_thresh):

    
    modelfile = fileprefix + '_model.pt'
    In = 0
    H = 0
    Out = 0
    acc =0
    tmp = fileprefix.split('_')
    
    for i in range(0,len(tmp)):
        if tmp[i] =='In':
            In = int(tmp[i+1])
        elif tmp[i] == 'H':
            H = int(tmp[i+1])
        elif tmp[i] == 'Out':
            Out = int(tmp[i+1])
        elif tmp[i]== 'acc':
            acc = float(tmp[i+1])
    
    #Exit if accuracy is too low
    if acc < acc_thresh:
        print('Accuracy below threshold')
        return 0, 0, 0
    
    model = ML_Binary_Classifier.load_model(modelfile, In, H, Out)
    
    test_data_file = fileprefix + '_test_data.csv'
    
    #data_start_column_label = 'ctx_rh_posterior_insula'
    #data_end_column_label = 'T1rho_VenousBlood'
    
    #Variable & input ranges are hard coded for now, should be modified for more generalizable usage
    input_tensor, ans_tensor = ML_Binary_Classifier.load_input_data(test_data_file, 'Group', data_start_column_label, data_end_column_label)
    
    all_data = pd.read_csv(test_data_file)
    imaging_variables = all_data.loc[:, data_start_column_label:data_end_column_label]
    column_labels = imaging_variables.columns.tolist()
    
    subjIDs = all_data['Participant_ID'].tolist()
    
    #Number of iterations and masking probability hard-coded, could be modified for more generalizable usage
    Cumulative_Input_Weights, Cumulative_Mask = RISE_Test(model, input_tensor, ans_tensor, 10000, 0.5)

    figure_name = fileprefix + '_Figure'
    generate_RISE_figure(Cumulative_Input_Weights, Cumulative_Mask, figure_name, subjIDs, column_labels)


    #Sort by influence size and test removal of features from least to most important
    outfile_name = fileprefix + '_sortedResults'
    trials_results = input_masking_experiment(model, input_tensor, ans_tensor, Cumulative_Input_Weights, outfile_name, column_labels, 0)

    #Create Sorted Bar Graphs of Data
    graph_name = fileprefix + '_SortedBarGraph1'
    test = generate_Sorted_Bargraph(Cumulative_Input_Weights, graph_name, column_labels, 0)

    graph_name = fileprefix + '_SortedBarGraph2'
    test = generate_Sorted_Bargraph(Cumulative_Input_Weights, graph_name, column_labels, 1)
    
    graph_name = fileprefix + '_AccGraph'
    generate_acc_figure(trials_results['h'].tolist(), trials_results['fp'].tolist(), trials_results['fn'].tolist(), graph_name)


    #Sort by influence size and test removal of features from most to least important
    outfile_name = fileprefix + '_sortedResultsReverse'
    trials_results2 = input_masking_experiment(model, input_tensor, ans_tensor, Cumulative_Input_Weights, outfile_name, column_labels, 1)

    graph_name = fileprefix + '_AccGraphReverse'
    generate_acc_figure(trials_results2['h'].tolist(), trials_results2['fp'].tolist(), trials_results2['fn'].tolist(), graph_name)

    

    #Save Results of RISE analysis
    RISE_results_df = pd.DataFrame(data=Cumulative_Input_Weights, columns=column_labels, index=subjIDs)
    results_name = fileprefix + '_RISEResults'
    pd.DataFrame(RISE_results_df).to_csv(results_name, index=False)

    RISE_mask_df = pd.DataFrame(data=Cumulative_Mask, columns=column_labels)
    mask_name = fileprefix+'_Mask'
    pd.DataFrame(RISE_mask_df).to_csv(mask_name, index=False)

    return trials_results
    
""" Display Functions """

def deMean_RISE_Results(Cumulative_Input_Weights):
    #calculate mean of each row
    RISE_Row_Means = Cumulative_Input_Weights.mean(axis=1)
    #De-mean each datapoint by subtracting the row mean
    DeMeaned_RISE_Results = Cumulative_Input_Weights - RISE_Row_Means[:,np.newaxis]
    
    return DeMeaned_RISE_Results

def calc_RISE_zStat(Cumulative_Input_Weights):
    #Calculate standard deviation for each row
    RISE_Row_STDs = Cumulative_Input_Weights.std(axis=1)
    #De-mean the datapoints
    DeMeaned_RISE_Results = deMean_RISE_Results(Cumulative_Input_Weights)
    
    #Divide de-meaned results by standard deviation (where stdev !=0) to calculate z-score
    z_score_RISE_Results = np.divide(DeMeaned_RISE_Results, RISE_Row_STDs[:,np.newaxis], out=np.zeros_like(DeMeaned_RISE_Results), where=RISE_Row_STDs[:,np.newaxis]!=0)

    return z_score_RISE_Results

def input_masking_experiment(model, input_tensor, ans_tensor, Cumulative_Input_Weights, outfileprefix, col_names, sort_direction_flag):
    
    #Get list of regions sorted by magnitude of z-statistic
    abs_val_flag = 1
    sorted_region_list_df = generate_Sorted_RegionList(Cumulative_Input_Weights, col_names, abs_val_flag, sort_direction_flag)
    trials = np.zeros((len(col_names), 3), dtype=float)


    input_df = pd.DataFrame(input_tensor.detach().numpy()).astype('float')
    input_df.columns = col_names

    #Start with empty mask
    #zero_data = np.zeros(shape=(1, len(col_names)))
    #mask = pd.DataFrame(zero_data, columns=col_names)
    
    #Loop through each column
    for i in range(0, len(sorted_region_list_df)):
    #for i in range(0,500):
       
        #Mask out next column in sorted list
        input_df[sorted_region_list_df.index[i]].values[:] = 0
        masked_input_tensor = torch.tensor(input_df.values)
        
        #Test Model Prediction on test set
        model.eval()
        with torch.no_grad():
            #Get prediction
            test_ans_pred = model(masked_input_tensor.float())
        
        
        test_ans = ans_tensor.detach().numpy() 
        binary_test_pred = torch.round(torch.sigmoid(test_ans_pred)).detach().numpy()
        #binary_test_pred = binary_test_pred.astype(int)
        
        #print(str(test_ans))
        #print(str(binary_test_pred))
        
        #Measure accuracy
        h,fp,fn = ML_Binary_Classifier.get_accuracy_metrics(test_ans, binary_test_pred)
        trials[i, :] = [np.mean(h), np.mean(fp), np.mean(fn)]
        
        print('Trial: ', str(i), ' - H:', str(np.mean(h)))
    
    
    trials_df = pd.DataFrame(data=trials, columns = ['h', 'fp', 'fn'])
    filename = outfileprefix + '.json'
    with open(filename, 'w') as f:
      
        json.dump(trials.tolist(), f)
        f.close()
    return trials_df
    
def generate_Sorted_Mask(sorted_region_list_df, col_names, n):
    
    zero_data = np.zeros(shape=(1, len(col_names)))
    mask = pd.DataFrame(zero_data, columns=col_names)
    
    #for i in range(len(col_names)-1, -1, -1):
    for i in range(n,len(col_names)):
        #print(str(i), ': ', sorted_region_list_df.index[i])
        mask[sorted_region_list_df.index[i]] = 1

    return mask    
    
def generate_Sorted_RegionList(Cumulative_Input_Weights, col_names, abs_val_flag, sort_direction_flag):

    z_score_RISE_Results = calc_RISE_zStat(Cumulative_Input_Weights)
    
    if abs_val_flag==1:
        #Calculate sum of RISE results for each region (i.e. column of Cumulative_Input_Weights)
        sum_list = np.mean(np.abs(z_score_RISE_Results), axis=0)
    else:
        sum_list = np.mean(z_score_RISE_Results, axis=0)


    #print(str(len(sum_list)), ': ', str(len(col_names)))
    #Create a dataframe with the region names as the index labels
    region_list_df = pd.DataFrame(data=sum_list, index=col_names)
    
    #Sort dataframe by values
    if sort_direction_flag==1:
        sorted_region_list_df = region_list_df.sort_values(by=[0], ascending=False)
        
    else:
        sorted_region_list_df = region_list_df.sort_values(by=[0])
    
    return sorted_region_list_df
    
def generate_Sorted_Bargraph(Cumulative_Input_Weights, outfileprefix, col_names, abs_val_flag):
    
    sorted_region_list_df = generate_Sorted_RegionList(Cumulative_Input_Weights, col_names, abs_val_flag, 0)
    
    fig, ax0 = plt.subplots(figsize=(48.0, 10.0))
    ax0.set_title('RISE Effect of Each Feature:')
    ax0.set_ylabel('Mean z-stat')
    ax0.set_xlabel('Imaging Measure')
    ax0.bar(sorted_region_list_df.index, sorted_region_list_df[0].tolist())
    
    plt.xticks(rotation = 'vertical', fontsize = 4)
    
    plt.subplots_adjust(bottom=0.5)
    plt.show()
    
    outfilename = outfileprefix + '.jpg'
    fig.savefig(outfilename, dpi = 300)
    
    return sorted_region_list_df

def generate_acc_figure(h, fp, fn, outfileprefix):
    
    #fig, ax0 = plt.subplots(figsize=(48.0, 10.0))
    #c = ax0.pcolor(acc_list)
    
    fig, ax = plt.subplots(figsize=(48.0, 10.0))
    ax.plot(range(0, len(h)), h, label='hits')
    ax.plot(range(0, len(fp)), fp, color='red', linewidth=1.0, label='false positives')
    ax.plot(range(0, len(fn)), fn, color='black', linewidth=1.0, label='false negatives')
    
    ax.legend(loc='upper left', frameon=False)
    
    ax.set_title('Sorted Region Masking')
    ax.set_ylabel('Rate')
    ax.set_xlabel('Ranked Imaging Measure')
    
    plt.show()
    
    outfilename = outfileprefix + '.jpg'
    fig.savefig(outfilename, dpi = 300)
    
def generate_RISE_figure(Cumulative_Input_Weights, Cumulative_Mask, outfileprefix, subjIDs, col_names):
    
      
    z_score_RISE_Results = calc_RISE_zStat(Cumulative_Input_Weights)

    #Save RISE Results
    
       
    temp_results_df = pd.DataFrame(Cumulative_Input_Weights)
    #temp_results_df.columns = imaging_variables.columns
    #print(temp_results_df)
    
    temp_subjID_df = pd.DataFrame(subjIDs)
    temp_subjID_df.reset_index(drop=True)
    RISE_results_df = pd.concat([temp_subjID_df, temp_results_df], axis=1)
    print(RISE_results_df)
    
    outfilename = outfileprefix + '.csv'
    RISE_results_df.to_csv(outfilename, index = False)
    outfilename = outfileprefix + '_Cumulative_Mask.csv'
    Cumulative_Mask = pd.DataFrame(Cumulative_Mask)
    Cumulative_Mask.to_csv(outfilename, index = False)

    #Threshold z-score values at 1.96, i.e. p < 0.05
    Thresholded_RISE_Results = (z_score_RISE_Results > 1.96) * z_score_RISE_Results
    #Thresholded_RISE_Results = z_score_RISE_Results
    #Thresholded_RISE_Results = DeMeaned_RISE_Results
    
    #Turn this into a dataframe and set the column labels
    Thresholded_RISE_Results = pd.DataFrame(Thresholded_RISE_Results)
    #Thresholded_RISE_Results.columns = imaging_variables.columns

    #print(Thresholded_RISE_Results)
    fig, ax0 = plt.subplots(figsize=(48.0, 10.0))
    c = ax0.pcolor(Thresholded_RISE_Results)
    ax0.set_title('Thresholded RISE Results:')
    ax0.set_ylabel('Participant')
    ax0.set_xlabel('Imaging Measure')

    nonzero_columns = (Thresholded_RISE_Results.sum(axis=0) > 0)
    df_select = [i for i, val in enumerate(nonzero_columns) if val]

    #print(str(df_select))
    
    
    
    #ax0.xticks(rotation = 45)
    #ax0.set_xticks(range(0,len(col_names)))
    #ax0.set_xticklabels(col_names)
    
    xlabels = [col_names[i] for i in df_select]
    #print(len(xlabels))
    plt.xticks(df_select, xlabels, rotation = 'vertical', fontsize = 4)
    
    
    ax0.set_yticks(range(0,len(subjIDs)))
    ax0.set_yticklabels(subjIDs)
   
    
    plt.subplots_adjust(bottom=0.5)
    plt.show()

    outfilename = outfileprefix + '.jpg'
    fig.savefig(outfilename, dpi = 300)

    #return Thresholded_RISE_Results

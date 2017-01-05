# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import sys
from subprocess import check_call

script_dir = os.path.realpath(os.path.dirname(sys.argv[0]))

import warnings
warnings.filterwarnings("ignore")

#all_models = np.load(os.path.join(script_dir, 'all_models_reg_n_grade.npy'))
all_models = np.load('all_models_reg_n_grade.npy')
best_model = all_models[2]

#Define a function to one-hot encode data using the type of columns
#INPUT: Dataframe
#       Columns with categorical values
#OUTPUT: Encoded dataset
#Main functionality: Convert all categorical columns to independent columns and
#                    Standardize all parameters
def encode(data, encode_cols):
    final_cols = ['y', 'Verified', 'n', 'car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 
                  'Source Verified', 'house', 'medical', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation', 
                  'wedding', 'major_purchase', 'RENT', 'Not Verified', 'OTHER', '36 months', '60 months', '1 year', '10+ years', 
                  '2 years', '3 years', 'OWN', '5 years', '6 years', '4 years', '8 years', '9 years', '< 1 year', 'n/a', 
                  'MORTGAGE', '7 years', 'pub_rec_bankruptcies', 'pub_rec', 'delinq_2yrs', 'inq_last_6mths', 'curr_mean_fico', 
                  'sub_grade', 'open_acc', 'earliest_cr_line', 'prev_mean_fico', 'total_acc', 'int_rate', 'funded_amnt', 
                  'revol_util', 'dti', 'annual_inc', 'revol_bal']
    encoded_x = pd.DataFrame({})
    encoded_final_x = pd.DataFrame(columns=final_cols)
    
    for col in data.columns:
        
        if (col in encode_cols[:]):
            encoded = pd.get_dummies(data[col])
            encoded_x = pd.concat([encoded_x, encoded], axis = 1)
        else:
            data[col] = data[col].astype(float)
            #data[col] = (data[col] - data[col].min())/(data[col].max() - data[col].min())
            encoded_x = pd.concat([encoded_x, data[[col]]], axis = 1)
            
    for col in encoded_x.columns:
        encoded_final_x[col] = encoded_x[col]
        
    encoded_final_x = encoded_final_x.fillna(0.0)
    
    return encoded_final_x


#input_list = [5000.00, '36 months', 10.65, 'B2', '10+ years','RENT', 24000, 'Verified',
#              'n','credit_card', 27.65, 0, 'Jan-85',1 , 3, 0, 
#              13648, 83.7, 9, 0, 737, 742]
              
input_list = [200000, '60 months', 50.89, 'F1', '6 years','RENT', 70000, 'Source Verified',
              'n','debt_consolidation', 16.63, 0, 'Nov-02', 10, 10, 0, 
              10470, 79.9, 50, 0, 667, 542]           

all_cols = ['funded_amnt', 'term', 'int_rate', 'sub_grade', 'emp_length','home_ownership', 'annual_inc', 'verification_status',
           'pymnt_plan','purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line','inq_last_6mths', 'open_acc', 'pub_rec', 
           'revol_bal', 'revol_util','total_acc', 'pub_rec_bankruptcies', 'curr_mean_fico','prev_mean_fico']

encode_cols = ['emp_length', 'home_ownership', 'verification_status', 'term', 'pymnt_plan', 'purpose']

input_df = pd.DataFrame(columns = all_cols)
input_df.loc[0] = input_list

input_df['grade'] = input_df['sub_grade'][0][0]
input_df['grade'] = input_df['grade'].map({'A': 0, 'B': 5, 'C': 10, 'D': 15, 'E': 20, 'F': 25, 'G': 30})
input_df['sub_grade'] = input_df['sub_grade'].str.extract('(\d+)').astype(int) + input_df['grade']
input_df.drop('grade', 1, inplace=True)

input_df.earliest_cr_line = pd.to_datetime(input_df.earliest_cr_line)
date_diff = (pd.datetime.today() - input_df.earliest_cr_line)
input_df.earliest_cr_line = date_diff.dt.days/365
input_df.earliest_cr_line = input_df.earliest_cr_line.astype(int)

final_df = encode(input_df, encode_cols)

if best_model.predict(final_df)[0] == 0:
    print "This loan might not get defaulted"
else:
    print "This loan might get defaulted"

#print best_model.predict(final_df)


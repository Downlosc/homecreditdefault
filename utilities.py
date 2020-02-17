#!/usr/bin/python
# This file contains utility functions used throughout the project
# Inspired by the wonderful notebook https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection

import pandas as pd 
import numpy as np
import gc
gc.enable()
import sys



def remove_missing_columns(df, threshold = 70):
    """ remove all features with more than threshold of N.a.N """
    # Calculate missing stats for df (remember to calculate a percent!)
    df_miss = pd.DataFrame(df.isnull().sum())
    df_miss['percent'] = 100 * df_miss[0] / len(df)
    
    
    # list of missing columns for df
    missing_df_columns = list(df_miss.index[df_miss['percent'] > threshold])
    
    # Print information
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_df_columns), threshold))
    
    # Drop the missing columns and return
    df = df.drop(columns = missing_df_columns)
    
    return df




def convert_types(df, print_info = False):
    """ #Shrink down as much as we can the size of the dataframes.
#Note that every numerical value lies within the range indexed with a float/int of 32 bits
#Moreover is wise to convert every object feature into a category one, especially if the number of unique values is far from the number of rows """
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object'):
            df[c] = df[c].astype('category')
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df



def to_csv(df, path):
    """ Efficiently store dataframes by keeping the dtypes stored along """
    import os
    import json
    
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    jtypes = json.dumps(dtypes)

    fileName = os.path.splitext(path)

    # save df as usual along with a json representation of the dictionary
    df.to_csv(path, index=False)

    f = open(fileName[0]+'Types',"w")
    f.write(jtypes)
    f.close()

    # free memory
    gc.enable()
    del df
    gc.collect()




def read_csv(path):
    import os
    import json

    """ Read version of to_csv """
    fileName = os.path.splitext(path)
    
    jtypes = json.load(open(fileName[0]+'Types'))
    
    return pd.read_csv(path, dtype=jtypes)



# For test's purposes
def read_csvTmp(path,nr=10000):
    import os
    import json

    fileName = os.path.splitext(path)
    
    jtypes = json.load(open(fileName[0]+'Types'))
    
    return pd.read_csv(path, dtype=jtypes,nrows=nr)






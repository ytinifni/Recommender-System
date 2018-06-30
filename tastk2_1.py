
# coding: utf-8

# In[49]:

import numpy as np
import csv
import pandas as pd
#import matplotlib.pyplot as plts

def task1():
    with open('ratings_Electronics_50.csv', 'r') as f:
        csv_data = csv.reader(f, delimiter=',')
        data = {}
        df = pd.read_csv('ratings_Electronics_50.csv', header=None)
        items = df.groupby(df[1]).count()
        users = df.groupby(df[0]).count()
        ratings = df.groupby(df[2]).count()
        ratings = ratings[1]
        users = users[1]
        #items = items[1]
        print("\nUSERS:\n",users)
        print("\nITEMS:\n",items)
        print("\nRATINGS:\n",ratings)
        
        
task1()


# In[ ]:




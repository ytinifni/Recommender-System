import operator
import numpy as np
import csv
import pandas as pd
#import matplotlib.pyplot as plts


            
def cosim(x,y):
    df = pd.read_csv('ratings_Electronics_50.csv', header=None)
    #totalItems = df.groupby(df[1]).count()
    items = pd.Series(df[1])
    users = pd.Series(df[0])
    xItems = df[ items == x]
    yItems = df[items == y]
    xItems.loc[:,2] = xItems.loc[:,2] - xItems.loc[:,2].mean()
    yItems.loc[:,2] = yItems.loc[:,2] - yItems.loc[:,2].mean()
    for index,e in xItems.loc[:,2].items():
        if e == 0:
            xItems.at[index, 2] = 0.0001
    for index,e in yItems.loc[:,2].items():
        if e == 0:
            yItems.at[index, 2] = 0.0001    

    soorat = 0
    #print(xItems)
    #print(yItems)
    for i in xItems[0].tolist():
        for j in yItems[0].tolist(): 
            if str(i) is str(j):
                #print(i)
                soorat = soorat + (xItems.loc[:,2].mean() - xItems[xItems[0] == str(i)][2].values[0]) * ( yItems.loc[:,2].mean() - yItems[yItems[0] == str(j)][2].values[0])
                #soorat = soorat + (xItems[xItems[0] == str(i)][2].values[0]) * ( yItems[yItems[0] == str(j)][2].values[0])
    norm1 = np.linalg.norm(np.array((xItems[2].tolist() - xItems.loc[:,2].mean())))
    if norm1 == 0:
        norm1 = 0.00001
    norm2 = np.linalg.norm(np.array((yItems[2].tolist() - yItems.loc[:,2].mean())))
    if norm2 == 0:
        norm2 = 0.00001    

    return (soorat / (norm1 * norm2))

#def predictRating(userId, itemId):

def mostSimilars(userId, itemId):
    df = pd.read_csv('ratings_Electronics_50.csv', header=None)
    users = pd.Series(df[0])
    userItems = df[ users == userId][1].tolist() #Items that active user rated
    similarList = {}
    counter = 0
    for item in userItems:
        similarList[item] = [cosim(str(item), itemId)]
        counter += 1
        #if counter == 5:
         #   break
    sortedList = sorted(similarList.items(), key=lambda kv: kv[1])
    if len(sortedList) > 10:
        sortedList = sortedList[-10:]
    return sortedList #output is a list of tuples (itemId, similarity)
    

#cosim('B000L4D42Q','B000N7VPZE')
#cosim('B000WON6O6', 'B000L4D42Q')
print(mostSimilars('A12LH2100CKQO', 'B000L4D42Q'))


# USEFUL THINGS:
# print(df.iloc[:,0:2]) --> shows two columns of the DF
# a = df[0].unique()

# we have 48190 unique items


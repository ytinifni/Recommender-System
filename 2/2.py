import operator
import numpy as np
import csv
import pandas as pd
import time

"""
    Read the csv file and return a DataFrame
"""
def readFile():
    df = pd.read_csv("ratings_Electronics_50.csv", header=None)
    df = df[:5000]
    return df

"""
    Make the input DataFrame into a User-Item Matrix typeof( dataframe )
    ARGS:
      df = dataFrame
"""
def userItemMatrix(df):
    df2 = pd.DataFrame({'userId': df[0], 'items': df[1], 'rating': df[2]})
    R_df = df2.pivot(index = 'userId', columns ='items', values = 'rating').fillna(0)
    return R_df

"""
    for each user calculates the mean of his/her ratings
"""
def meanUser(df):
    usersMean = {}
    users = df[0].drop_duplicates(keep='first').values
    for u in users:
        usersMean[u] = df[ df[0] == u ][2].mean()
    return usersMean

"""
    make a user-item matrix out of mean of ratings
"""
def calcMeanMatrix(usersMean,r_df):
    for user in usersMean:
        r_df.loc[user] -= usersMean[user]
    #print(r_df)
    return r_df

"""
Calculating adjusted cosine similarity of two items
ARGS:
  x = item1
  y = item2
  df = dataFrame
"""
def cosim(x,y,df):
    #Users who rated x and y
    usersOfx = df[ df[1] == x][0]
    usersOfy = df[ df[1] == y][0]
    ratingsOfx = df[ df[1] == x][2].values
    ratingsOfy = df[ df[1] == y][2].values
    xIndex = 0
    yIndex = 0

    commonUsers = []
    #Loop through lists of users of two items x,y. If we find a commonUser, it alongside the ratings to both items are added
    for ux in usersOfx.values:
        for uy in usersOfy.values:
            if ux is uy:
                commonUsers.append((ux,ratingsOfx[xIndex],ratingsOfy[yIndex]))
            yIndex += 1
        yIndex = 0
        xIndex += 1
    #If we find no commonUser, simiarity is 0 and there is no need to continue calculating cosine similarity:
    #Just returning a small float number
    if len(commonUsers) == 0:
        return 0.0001

    NUMERATOR  = 0
    norm1 = 0
    norm2 = 0
    for u,rx,ry in commonUsers:
        r_mean = df[ df[0] == u ][2].mean()
        NUMERATOR  += (rx - r_mean) * (ry - r_mean)
        norm1 += pow(rx - r_mean,2)
        norm2 += pow(ry - r_mean,2)
    return ( NUMERATOR  / np.sqrt(norm1) * np.sqrt(norm2) )

"""
    Calculating adjusted cosine similarity of two items
"""
def cosim2(item1, item2):
    numerator = item1.dot(np.transpose(item2))
    norm1 = np.linalg.norm(item1)
    norm2 = np.linalg.norm(item2)
    result = numerator / (norm1*norm2)
    return result

def itemSimilarity(r_df,df,usersRatingsMean ):
    items = df[1].drop_duplicates(keep = 'first').values
    simMatrix = np.zeros((len(items),len(items)))
    #simMatrix = pd.DataFrame(0, index=items, columns=items)
    for i in range(r_df.shape[1]):
        for j in range(i, r_df.shape[1]):
            simMatrix[i][j] = cosim2(r_df[items[i]].values,r_df[items[j]].values)
    # for i in range(len(items)):
    #     for j in range(i,len(items)):
    #         print(i)
    #         simMatrix[i][j] = cosim2(r_df[items[i]].values,r_df[items[j]].values,)
    print(simMatrix)
    return simMatrix

def arrayToDf(matrix,df):
    items = df[1].drop_duplicates(keep = 'first').values
    dfMatrix = pd.DataFrame(data=matrix, index=items, columns=items)
    return dfMatrix

"""
    calculating the set of items rated by active user (userId)
    that are most similar to item itemId. (L)
    Returns a list of tuples [(itemId, rating), ..]
"""
def mostSimilar(userId, itemId, iiMatrix, df, L):
    #Items that active user rated (L):
    userItems = df[ df[0] == userId ][1].values
    #iidf = pd.DataFrame(data=iiMatrix,index=df[1].drop_duplicates(keep="first").values)
    similarList = {}
    for item in userItems:
        tempSim = iiMatrix.loc[itemId][item]

        #only half of the matrix is full ( item-item matrix )
        #So we need to check if we're in the right triangle:
        if tempSim == 0:
            tempSim = iiMatrix.loc[item][itemId]
        if tempSim > 0:
            similarList[item] = tempSim
    sortedList = sorted(similarList.items(), key=lambda kv: kv[1], reverse=True)
    if len(sortedList) > L:
        sortedList = sortedList[:L]
    return sortedList

#Predict a rate for the itemId which is given
def predict(userId, itemId, df, r_df,iiDF):
    similars = mostSimilar(userId, itemId, iiDF, df, 10)
    NUMERATOR  = 0
    simsum = 0
    for s in (similars):
        simsum += abs(s[1])
        NUMERATOR  += s[1] * r_df.loc[userId,s[0]]
    return NUMERATOR  / simsum

"""
Returns the top K Item recommended to user
"""
def predictTopKRecommendations(userId, k, df, r_df, iiDF):
    #Items that active user rated (L):
    recs = []
    userItems = df[ df[0] == userId][1].tolist()
    items = df[1].drop_duplicates(keep = 'first').values
    for item in items:
        if( item not in userItems ):
            recs.append((predict(userId, item, df, r_df, iiDF),item))
        else:
            continue
    #sorting the predicts by their rate
    recs.sort(key=lambda tup: tup[0], reverse=True)
    topK = recs[:k]
    return topK

#[(nan, 'B00008A6CC'), (nan, 'B0001GMT02'), (nan, 'B000CF3QTA'), (-2.0, 'B000EWJYYW'), (nan, 'B000F1QV5M'), (nan, 'B000GJXCAA'), (-4.883327471146629, 'B000HDJT4S'), (-5.0, 'B000I98ZYG'), (-5.0, 'B000ID37EA'), (nan, 'B000NWHWIS')]


df = readFile()
r_df = userItemMatrix(df)

usersRatingsMean = meanUser(df)
uiMatrix = calcMeanMatrix(usersRatingsMean,r_df)

"""
Users:
A12LH2100CKQO

Items:
B00008A6CC
B000L4D42Q
B000WON6O6
B000HDJT4S *

* : our user did not rate this item
"""


iiMatrix = itemSimilarity(uiMatrix,df, usersRatingsMean)
iiDF = arrayToDf(iiMatrix,df)

#mostSims = mostSimilar("A12LH2100CKQO","B000HDJT4S",df=df, iiMatrix=iiDF, L=10)
#pr = predict("A12LH2100CKQO", "B000HDJT4S", df, r_df,iiDF)
t1 = time.time()
pr = predictTopKRecommendations("A12LH2100CKQO", 10, df, r_df, iiDF)
t2 = time.time()
print(pr)

print("Duration: ",t2 - t1)
#[(4.999850129100263, 'B000I98ZYG'), (4.999505488331909, 'B008CBQSKU'), (4.998953081172865, 'B0079X1VQS'), (4.997260737706318, 'B002HQUIVQ'), (4.99555894041664, 'B0014YXM9M'), (4.968261050552186, 'B0014L4ZKK'), (4.931845483767931, 'B000067SPL'), (4.9004820177043715, 'B002WE6D44'), (4.9004820177043715, 'B002WE6D44'), (4.892200110256977, 'B0058UUR6E')]

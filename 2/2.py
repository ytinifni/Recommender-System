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
    #items = self.df[1].drop_duplicates(keep = 'first').values[:10]
    #users = self.df[0].drop_duplicates(keep = 'first').values[:10]
    df2 = pd.DataFrame({'userId': df[0], 'items': df[1], 'rating': df[2]})
    R_df = df2.pivot(index = 'userId', columns ='items', values = 'rating').fillna(0)
    #print(self.R_df)
    return R_df


def meanUser(df):
    usersMean = {}
    users = df[0].drop_duplicates(keep='first').values
    for u in users:
        usersMean[u] = df[ df[0] == u ][2].mean()
    #print(usersMean)
    return usersMean

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

def cosim2(item1, item2):
    numerator = item1.dot(np.transpose(item2))
    norm1 = np.linalg.norm(item1)
    norm2 = np.linalg.norm(item2)
    result = numerator / (norm1*norm2)
    #print(result)
    return result

def itemSimilarity(r_df,usersRatingsMean ):
    items = df[1].drop_duplicates(keep = 'first').values
    simMatrix = np.zeros((len(items),len(items)))
    for i in range(r_df.shape[1]):
        for j in range(i, r_df.shape[1]):
            simMatrix[i][j] = cosim2(r_df[items[i]].values,r_df[items[j]].values,)
    # for i in range(len(items)):
    #     for j in range(i,len(items)):
    #         print(i)
    #         simMatrix[i][j] = cosim2(r_df[items[i]].values,r_df[items[j]].values,)
    print(simMatrix)
    return simMatrix

def mostSimilar(userId,itemId,df,L):
    #Items that active user rated (L):
    userItems = df[ df[0] == userId][1].tolist()

    # key, values ==> keys are the items which were similar enough
    #values are the cosine similarity between the two:
    similarList = {}
    for item in userItems:
        tempSim = cosim(str(item),itemId,df)
        #If similarity is a negative number, we ignore.
        if tempSim > 0:
            similarList[item] = tempSim
        else:
            continue

    sortedList = sorted(similarList.items(), key=lambda kv: kv[1], reverse=True)
    if len(sortedList) > L:
        sortedList = sortedList[:L]
    return sortedList
    #return similarList

#Predict a rate for the itemId which is given
def predict(userId, itemId, df, r_df):
    similars = mostSimilar(userId, itemId, df, 10)
    NUMERATOR  = 0
    simsum = 0
    for s in (similars):
        simsum += abs(s[1])
        NUMERATOR  += s[1] * r_df.loc[userId,s[0]]
    if ( simsum == 0 ):
        simsum = 1
    return NUMERATOR  / simsum

"""
Returns the top K Item recommended to user
"""
def predictTopKRecommendations(userId, k, df, r_df):
    #Items that active user rated (L):
    recs = []
    userItems = df[ df[0] == userId][1].tolist()
    items = df[1].drop_duplicates(keep = 'first').values
    for item in items[:100]:
        if( item not in userItems ):
            recs.append((predict(userId, item, df, r_df),item))
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
a = calcMeanMatrix(usersRatingsMean,r_df)
t1 = time.time()

#itemSimilarity(a,usersRatingsMean)


#print(predict("A12LH2100CKQO","B00008A6CC",df, r_df)) #7.771 TIME
#print(cosim('B000WON6O6', 'B000L4D42Q',df)) # 0.06 TIME
# print(a['B000WON6O6'].values)
# print(a['B00008A6CC'].values)
#print(cosim2(a['B000WON6O6'].values, a['B00008A6CC'].values)) # 0.06 TIME
#print(cosim2(a['B000WON6O6'].values, a['B000L4D42Q'].values)) # 0.06 TIME

#print(mostSimilar("A12LH2100CKQO","B000L4D42Q",df,10))  # 7.606 TIME
#print(predictTopKRecommendations('A12LH2100CKQO', 10,df, r_df)) # 15 min

t2 = time.time()
print(t2 - t1)
#[(4.999850129100263, 'B000I98ZYG'), (4.999505488331909, 'B008CBQSKU'), (4.998953081172865, 'B0079X1VQS'), (4.997260737706318, 'B002HQUIVQ'), (4.99555894041664, 'B0014YXM9M'), (4.968261050552186, 'B0014L4ZKK'), (4.931845483767931, 'B000067SPL'), (4.9004820177043715, 'B002WE6D44'), (4.9004820177043715, 'B002WE6D44'), (4.892200110256977, 'B0058UUR6E')]

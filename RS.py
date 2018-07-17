import numpy as np
import scipy
from numpy import linalg as la
import math
import pandas as pd
import operator
import csv
import time


"""
    TASK 2 FUNCTIONS
"""

"""
    Read the csv file and return a DataFrame
"""
def readFile(fileName):
    return pd.read_csv(fileName, header=None)


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
    for each user calculates the mean of all his/her ratings
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

"""
    Calculating adjusted cosine similarity of two items
"""
def cosim(item1, item2):
    numerator = item1.dot(np.transpose(item2))
    norm1 = np.linalg.norm(item1)
    norm2 = np.linalg.norm(item2)
    result = numerator / (norm1*norm2)
    return result

"""
    making the item-item matrix ( similarity Matrix )
"""
def itemSimilarity(r_df,df,usersRatingsMean ):
    print("\n\ncalculating similarity Matrix...\n")
    items = df[1].drop_duplicates(keep = 'first').values
    simMatrix = np.zeros((len(r_df.columns),len(r_df.columns)))

    for i in range(r_df.shape[1]):
        for j in range(i, r_df.shape[1]):
            simMatrix[i][j] = cosim(r_df[r_df.columns.values[i]].values,r_df[r_df.columns.values[j]].values)
    # for i in range(len(items)):
    #     for j in range(i,len(items)):
    #         print(i)
    #         simMatrix[i][j] = cosim2(r_df[items[i]].values,r_df[items[j]].values,)
    print(simMatrix)
    return simMatrix

def arrayToDf(matrix,r_df):
    items = r_df.columns.values
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
            #recs.append((predict(userId, item, df, r_df, iiDF),item))
            recs.append((predict(userId, item, df, r_df, iiDF),item))
        else:
            continue
    #sorting the predicts by their rate
    recs.sort(key=lambda tup: tup[0], reverse=True)
    topK = recs[:k]
    return topK

# ------------------------------------------------------------------------------

"""
    TASK 3 FUNCTIONS
"""
class MF():
    def __init__(self, df, R_df, r, alpha, lambdaa, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - df            : dataframe
        - r_df          : user-item dataframe
        - r (int)       : Rank
        - alpha (float) : learning rate
        - lambdaa (float)  : regularization parameter
        """

        self.r = r
        self.alpha = alpha
        self.lambdaa = lambdaa
        self.iterations = iterations
        self.df = df
        self.R_df = R_df

    def train(self):

        self.u = abs(np.random.normal(scale=1./self.r, size=(len(self.R_df.index), self.r)))
        self.v = abs(np.random.normal(scale=1./self.r, size=(len(self.R_df.columns), self.r)))

        self.b_u = np.zeros(len(self.R_df.index))
        self.b_i = np.zeros(len(self.R_df.columns))
        #self.b = self.df[2].mean()

        for i in range(self.iterations):
            self.sgd()
            print("Round ",i+1)
        self.a = pd.DataFrame(self.alll())
        self.a.index = self.R_df.index.values
        self.a.columns = self.R_df.columns.values
        return self.a


    #mean square error
    def mse(self):
        predicted = self.alll()
        error = 0
        for index, row in self.df.iterrows():
            userId = row[0]
            itemId = row[1]
            r = row[2]
            i = self.R_df.index.get_loc(userId)
            j = self.R_df.columns.get_loc(itemId)
            error += pow(r - predicted[i, j], 2)
        print("error = %.3f" % (np.sqrt(error)))

    # stochastic graident descent
    def sgd(self):
        for index, row in self.df.iterrows():
            userId = row[0]
            itemId = row[1]
            r = row[2]
            i = self.R_df.index.get_loc(userId)
            j = self.R_df.columns.get_loc(itemId)
            """
            to this point we have the row and col index of each nonzero rates
            """
            r_prime = self.b_u[i] + self.b_i[j] + self.u[i, :].dot(self.v[j, :].T)
            error = r - r_prime

            self.b_u[i] += self.alpha * (error - self.lambdaa * self.b_u[i])
            self.b_i[j] += self.alpha * (error - self.lambdaa * self.b_i[j])

            self.u[i,:] += self.alpha * ((error * self.v[j,:]) - (self.lambdaa*self.u[i,:]))
            self.v[j,:] += self.alpha * ((error * self.u[i,:]) - (self.lambdaa*self.v[j,:]))



    def alll(self):
        return self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.u.dot(np.transpose(self.v))




fileList = ['ratings_Electronics_50_fold1.csv', 'ratings_Electronics_50_fold2.csv', 'ratings_Electronics_50_fold3.csv', 'ratings_Electronics_50_fold4.csv']


def mergeCSVs(fileList):
    dfs = [pd.read_csv(f, header=None) for f in fileList]
    merged = pd.concat(dfs, axis=0, join='inner').sort_index()
    return merged

"""
    Binary Judgment
"""
def rel( user, rating, r_df):
    if rating > r_df.loc[user].mean():
        return 1
    else:
        return 0

"""
    calculating relevant set of items to user u
    ARGS:
        preds = dictionary of (user, Top_10_Recommendations_Items)
"""
def relevant_ARS(preds, r_df):
    print("\nCalculating relevancy for ARS...\n")
    u_plus = {}
    for user, items in preds.items():
        for item in items:
            if u_plus.get(user) == None:
                u_plus[user] = rel(user, item, r_df)
            else:
                u_plus[user] += rel(user, item, r_df)

    print(u_plus)
    return u_plus

"""
    calculating relevant set of items to user u
    ARGS:
        preds = dictionary of (user, Top_10_Recommendations_Items)
"""
def relevant_BLRS(preds, r_df):
    print("\nCalculating relevancy for BLRS...\n")
    u_plus = {}
    for user, items in preds.items():
        for item in items:
            if u_plus.get(user) == None:
                u_plus[user] = rel(user, item[0], r_df)
            else:
                u_plus[user] += rel(user, item[0], r_df)

    print(u_plus)
    return u_plus

"""
    given a dictionary of user and number of 1 in its relavant set.
    will return the percision@10 regarding the related Formula in the slides
"""
def percisionM(u_plus):
    print("\nCalculating percision@10...\n")
    numerator = 0
    for key, value in u_plus.items():
        if value > 0:
            numerator += 1
    return numerator/10


#-------------------------------------------------------------------------------
"""
Making the dataFrame and user-item dataFrame of fold 1_4 and fold 5:

NOTE:
    since the data was very huge. Time-wise and using our ordinary Laptop,
    it was not possible to see the results for the whole datas.
    Therefore, we are just using a slice of the data ( 1000 ).
"""
df5 = readFile("ratings_Electronics_50_fold5.csv")
r_df5 = userItemMatrix(df5)
df1_4 = mergeCSVs(fileList)
df1_4 = df1_4[:1000]
r_df1_4 = userItemMatrix(df1_4)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
"""
    Same procedure as above without slicing.
    If it is desirable to test the code with the whole data,
    use the dataframes below rather than the ones from above code section
"""
df5_total = readFile("ratings_Electronics_50_fold5.csv")
r_df5_total = userItemMatrix(df5)
df1_4_total = mergeCSVs(fileList)
r_df1_4_total = userItemMatrix(df1_4)
#-------------------------------------------------------------------------------

"""
 This function runs the whole task2 by calling the related functions
 Returning a dictionary consisting of (userId, topKRecommendations) key, values

 ARGS:
    df = desired dataFrame
    r_df = desired user-item dataFrame
"""
def BLRS(df, r_df):
    usersList = df[0].drop_duplicates(keep="first").values
    usersRatingsMean = meanUser(df)
    uiMatrix = calcMeanMatrix(usersRatingsMean,r_df)
    iiMatrix = itemSimilarity(uiMatrix,df, usersRatingsMean)
    iiDF = arrayToDf(iiMatrix,r_df)
    pr = {}
    # in each loop we add  Top_10_Recommendations for corresponding user
    for user in usersList:
        pr[user] = predictTopKRecommendations(user, 10, df, r_df, iiDF)
    return pr

#-------------------------------------------------------------------------------

"""
 This function runs the whole task3 by calling the related functions
 Returning

 ARGS:
    df = desired dataFrame
    r_df = desired user-item dataFrame
"""
def ARS(df, r_df):
    mf = MF(df=df, R_df=r_df, r=20, alpha=0.05, lambdaa=0.002, iterations=10)
    return mf.train()
#-------------------------------------------------------------------------------

"""
    returns a dataFrame and its user-item Matrix with 10 users
"""
def selectUsers(df, r_df):
    r_df = r_df[:10]
    users = r_df.index.values
    tempDF = pd.DataFrame()
    res = []
    for u in users:
        tempDF = df[ df[0] == u ]
        res.append(tempDF)
    return pd.concat(res)

"""
    getting the topK recommends for each user from the predicted item-user matrix
"""
def getTopK(r_df):
    users = r_df.index.values
    tmp = []
    topKs = {}
    for user in users:
        tmp = r_df.loc[user].values
        sortedTmp = sorted(tmp, reverse=True)
        sortedTmp = sortedTmp[:10]
        topKs[user] = sortedTmp
    return topKs
#-------------------------------------------------------------------------------
# selecting 10users for evaluation from user-item Matrix:
randomR_df = r_df1_4[:10]
#getting these 10 users and their item they rated from dataFrame:
randomDF = selectUsers(df1_4, r_df1_4)

#-------------------------------------------------------------------------------
"""
    calling all the necessary functions for
    evaluating ARS
"""
def eval_ARS():
    ars = ARS(randomDF, randomR_df)
    preds = getTopK(ars)
    rels = relevant_ARS(preds, r_df5)
    print("precision: ",percisionM(rels))

"""
    calling all the necessary functions for
    evaluating BLRS
"""
def eval_BLRS():
    blrs = BLRS(randomDF, randomR_df)
    rels = relevant_BLRS(blrs, r_df5)
    print("precision: ",percisionM(rels))

if __name__ == '__main__':
    eval_ARS()
    print("------------------------------------------------------------------n")
    eval_BLRS()

    """
        un-comment the codes below to run just the prediction implementations 
    """
    # Just running ARS prediction:
    #ars = print(randomDF, randomR_df)

    #just running BLRS prediciton:
    #ars = BLRS(randomDF, randomR_df)

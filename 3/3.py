import numpy as np
import scipy
from numpy import linalg as la
import math
import pandas as pd

"""

1540 unique users
48190 unique items

"""

class MF():

    def __init__(self, filename, r, alpha, lambdaa, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - filename (string) : data CSV file
        - r (int)       : Rank
        - alpha (float) : learning rate
        - lambdaa (float)  : regularization parameter
        """

        self.r = r
        self.alpha = alpha
        self.lambdaa = lambdaa
        self.iterations = iterations
        self.fileName = filename

    #Read the csv file and return a DataFrame
    def readFile(self):
        self.df = pd.read_csv(self.fileName, header=None)

    #Make the input DataFrame into a User-Item Matrix typeof( dataframe )
    def userItemMatrix(self):
        #items = self.df[1].drop_duplicates(keep = 'first').values[:10]
        #users = self.df[0].drop_duplicates(keep = 'first').values[:10]
        df2 = pd.DataFrame({'userId': self.df[0], 'items': self.df[1], 'rating': self.df[2]})
        self.R_df = df2.pivot(index = 'userId', columns ='items', values = 'rating').fillna(0)
        #print(self.R_df)

    def train(self):

        self.u = abs(np.random.normal(scale=1./self.r, size=(len(self.R_df.index), self.r)))
        self.v = abs(np.random.normal(scale=1./self.r, size=(len(self.R_df.columns), self.r)))

        self.b_u = np.zeros(len(self.R_df.index))
        self.b_i = np.zeros(len(self.R_df.columns))
        self.b = self.df[2].mean()
        print(self.b)

        for i in range(self.iterations):
            self.sgd()
            print("Round ",i)
            #if (i+1) % 10 == 0:
            #    print("Iteration: %d ; error = %.4f" % (i+1, mse))
        a = pd.DataFrame(self.alll())
        print(a)
        a.to_csv('out.csv', sep='\t', encoding='utf-8')
        self.mse()


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
        print("error = %.4f" % (np.sqrt(error)))

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
            r_prime = self.b + self.b_u[i] + self.b_i[j] + self.u[i, :].dot(self.v[j, :].T)
            error = r - r_prime

            self.b_u[i] += self.alpha * (error - self.lambdaa * self.b_u[i])
            self.b_i[j] += self.alpha * (error - self.lambdaa * self.b_i[j])

            self.u[i,:] += self.alpha * ((error * self.v[j,:]) - (self.lambdaa*self.u[i,:]))
            self.v[j,:] += self.alpha * ((error * self.u[i,:]) - (self.lambdaa*self.v[j,:]))



    def alll(self):
        return  self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.u.dot(np.transpose(self.v))




mf = MF(filename='ratings_Electronics_50.csv', r=20, alpha=0.05, lambdaa=0.002, iterations=10)
mf.readFile()
mf.userItemMatrix()
mf.train()

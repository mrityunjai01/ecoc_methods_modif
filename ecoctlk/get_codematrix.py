from CodeMatrix.CodeMatrix import ecoc_one
from sklearn.linear_model import LogisticRegression
import mnist
from sklearn.preprocessing import StandardScaler
import numpy as np

def normer(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return X

def mnist_load(totalsamp = 200):
    XALL = mnist.train_images()[:totalsamp]/255.
    XALL = XALL.reshape(-1, 784)
    XALL = normer(XALL)
    yALL = mnist.train_labels().reshape(-1,1)[:totalsamp].astype(np.int8)
    print(XALL.shape)

    return XALL,yALL


X, Y = mnist_load(totalsamp=10000)
r = int(X.shape[0] * 0.1)
trainX = X[:r*9]
trainY = Y[:r*9]
valX = X[r*9:]
valY = Y[r*9:]

matrix, _, _1, _2 = ecoc_one(trainX, trainY, valX, valY, LogisticRegression(max_iter=1000))
print(matrix.shape)
print(dir(matrix))
np.save("codematrix_ecoc_one.npy", matrix)

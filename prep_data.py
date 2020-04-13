import numpy as np
import pandas as pd

#xEnd is inclusice(last column of X)
def getData(dataFile, xStart, xEnd):
    data = pd.read_csv(dataFile, sep=",", header=None)
    xDat = data.loc[:,xStart:xEnd]
    yDat = data.loc[:, xEnd+1]
    x = xDat.to_numpy()
    y = yDat.to_numpy()
    return (x,y)

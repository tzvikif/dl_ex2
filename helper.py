import numpy as np
import matplotlib.pyplot as plt

def Multiplot(l,xlabel,ylabel,title=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for p in l:
        x = p['x']
        y = p['y']
        funcName = p['funcName']
        plt.plot(x,y,label = funcName)
        plt.legend()
        plt.title(title)
        plt.plot()
    plt.show()

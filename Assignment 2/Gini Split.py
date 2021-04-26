# AN IMPLEMENTATION OF GINI SPLIT VALUE CALCULATION FROM SCRATCH

import numpy as np


def nos(c, in1):
    in2 = np.where(c[in1[0]] == 0)
    in3 = np.where(c[in1[0]] > 0)
    return len(in2[0]), len(in3[0])


def gini_split_cal(f, c):
    mf = np.median(f)
    in1 = np.where(f <= mf)
    in2 = np.where(f > mf)
    n00, n01 = nos(c, in1)
    n10, n11 = nos(c, in2)
    if n00 == 0 and n01 == 0:
        p00 = 0.5
        p01 = 0.5
    else:
        p00 = n00/(n00+n01)
        p01 = n01/(n00+n01)
    if n10 == 0 and n11 == 0:
        p10 = 0.5
        p11 = 0.5
    else:
        p10 = n10/(n10+n11)
        p11 = n11/(n10+n11)
    gl = 1-p00*p00-p01*p01
    gr = 1-p10*p10-p11*p11

    fgs = len(in1[0])/(len(in1[0])+len(in2[0]))*gl + \
        len(in2[0])/(len(in1[0]) + len(in2[0]))*gr
    return fgs


def rank(result):
    frank = np.zeros((56, 20), dtype='int32')
    for i in range(1, 57):
        val = result[i-1, :]
        frank[i-1, :] = np.argsort(val) + 1
    return frank


# code to edit csv
gini_split = np.zeros((56, 25))
fileloc = "output/"
for i in range(1, 57):
    print(i)
    fname = 'data/'+str(i)+'.csv'
    data = np.genfromtxt(fname, delimiter=',')
    for j in range(0, 20):
        gini_split[i-1, j] = gini_split_cal(data[:, j], data[:, -1])
    frank = rank(gini_split[:, :20])
    gini_split[:, 20:] = frank[:, :5]
fname = fileloc+'gini_split.csv'
np.savetxt(fname, gini_split, delimiter=',', fmt='%f')

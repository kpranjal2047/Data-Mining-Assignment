import numpy as np
from math import log2


def entropy(class0, class1):
    if(class0 == 0 or class1 == 0):
        return 0
    class0, class1 = class0/(class0+class1), class1/(class0+class1)
    return -(class0 * log2(class0) + class1 * log2(class1))


def nos(c, in1):
    in2 = np.where(c[in1[0]] == 0)
    in3 = np.where(c[in1[0]] > 0)
    return len(in2[0]), len(in3[0])


def info_gain_cal(f, c):
    mf = np.median(f)
    in1 = np.where(f <= mf)
    in2 = np.where(f > mf)
    n00, n01 = nos(c, in1)
    n10, n11 = nos(c, in2)

    class_0 = len(in1[0]) / (len(in1[0])+len(in2[0]))
    class_1 = len(in2[0]) / (len(in1[0])+len(in2[0]))

    s_entropy = entropy(n00+n10, n01+n11)

    if (n00 == 0 or n01 == 0):
        s1_entropy = 0
    else:
        s1_entropy = entropy(n00, n01)
    if(n10 == 0 or n11 == 0):
        s2_entropy = 0
    else:
        s2_entropy = entropy(n10, n11)

    gain = s_entropy - (class_0 * s1_entropy + class_1 * s2_entropy)
    return gain


def rank(result):
    frank = np.zeros((56, 20), dtype='int32')
    for i in range(1, 57):
        val = result[i-1, :]
        frank[i-1, :] = np.argsort(val)[::-1] + 1
    return frank


# code to edit csv
info_gain = np.zeros((56, 25))
fileloc = "output/"
for i in range(1, 57):
    print(i)
    fname = 'data/'+str(i)+'.csv'
    data = np.genfromtxt(fname, delimiter=',')
    for j in range(0, 20):
        info_gain[i-1, j] = info_gain_cal(data[:, j], data[:, -1])
    frank = rank(info_gain[:, :20])
    info_gain[:, 20:] = frank[:, :5]
fname = fileloc+'info_gain.csv'
np.savetxt(fname, info_gain, delimiter=',', fmt='%f')

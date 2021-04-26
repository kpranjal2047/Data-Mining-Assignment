# AN IMPLEMENTATION OF CHI-SQUARE VALUE CALCULATION FROM SCRATCH
import numpy as np


def nos(c, in1):
    in2 = np.where(c[in1[0]] == 0)
    in3 = np.where(c[in1[0]] > 0)
    return len(in2[0]), len(in3[0])


def chaid_cal(f, c):
    mf = np.mean(f)
    in1 = np.where(f <= mf)
    in2 = np.where(f > mf)
    n00, n01 = nos(c, in1)
    n10, n11 = nos(c, in2)

    cl_0 = n00+n01
    cl_1 = n10+n11
    tot = cl_0+cl_1
    val_0 = n00+n10
    val_1 = n01+n11

    if val_0 == 0 or val_1 == 0 or cl_0 == 0 or cl_1 == 0:
        return float('inf')

    exp_00 = cl_0*val_0/tot
    exp_01 = cl_0*val_1/tot
    exp_10 = cl_1*val_0/tot
    exp_11 = cl_1*val_1/tot

    chaid_value = ((n00-exp_00)**2)/exp_00 + \
        ((n01-exp_01)**2)/exp_01 + \
        ((n10-exp_10)**2)/exp_10 + \
        ((n11-exp_11)**2)/exp_11

    return chaid_value


def rank(result):
    frank = np.zeros((56, 20), dtype='int32')
    for i in range(1, 57):
        val = result[i-1, :]
        frank[i-1, :] = np.argsort(val)[::-1] + 1
    return frank


# code to edit csv
chaidval = np.zeros((56, 25))
fileloc = "output/"
for i in range(1, 57):
    print(i)
    fname = 'data/'+str(i)+'.csv'
    data = np.genfromtxt(fname, delimiter=',')
    for j in range(0, 20):
        chaidval[i-1, j] = chaid_cal(data[:, j], data[:, -1])
    frank = rank(chaidval[:, :20])
    chaidval[:, 20:] = frank[:, :5]
fname = fileloc+'chi-square.csv'
np.savetxt(fname, chaidval, delimiter=',', fmt='%f')

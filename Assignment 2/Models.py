# A DECSION TREE FROM SCRATCH

import numpy as np


def preprocess(data):
    for i in range(20):
        col = data[:, i]
        med = np.median(col)
        f_0 = np.where(col <= med)[0]
        f_1 = np.where(col > med)[0]
        col[f_0] = 0
        col[f_1] = 1
        data[:, i] = col
    data[np.where(data[:, -1] > 0)[0], -1] = 1
    return data


def create(data, features):
    if len(features) > 0:
        dic = {'value': None, 0: None, 1: None}
        class_0 = np.where(data[:, -1] == 0)[0]
        class_1 = np.where(data[:, -1] == 1)[0]

        if len(class_0) == 0:
            dic['value'] = 1
            return dic
        if len(class_1) == 0:
            dic['value'] = 0
            return dic

        f_0 = np.where(data[:, features[0]-1] == 0)[0]
        f_1 = np.where(data[:, features[0]-1] == 1)[0]

        dic[0] = create(data[f_0, :], features[1:])
        dic[1] = create(data[f_1, :], features[1:])
        return dic

    elif len(features) == 0:
        dic = {'value': None, 0: None, 1: None}
        class_0 = np.where(data[:, -1] == 0)[0]
        class_1 = np.where(data[:, -1] == 1)[0]
        dic['value'] = 1 if len(class_1) > len(class_0) else 0
        return dic

    else:
        return None


def fit(data, features):
    return create(data, features)


def predict(model, data, features):
    if model['value'] is not None:
        return model['value']
    if data[features[0]-1] == 0:
        return predict(model[0], data, features[1:])
    if data[features[0]-1] == 1:
        return predict(model[1], data, features[1:])


def calculate_accuracy(prediction, actual):
    total = len(actual)
    correct = np.where(prediction == actual)[0]
    return len(correct)/total


def calculate_f1_score(prediction, actual):
    p0 = np.where(prediction == 0)[0]
    p1 = np.where(prediction == 1)[0]
    a0 = np.where(actual == 0)[0]
    a1 = np.where(actual == 1)[0]

    tp = np.intersect1d(p1, a1)
    fp = np.intersect1d(p1, a0)
    fn = np.intersect1d(p0, a1)

    try:
        f1 = 2*len(tp) / (2*len(tp) + len(fp) + len(fn))
    except ZeroDivisionError:
        f1 = 0
    return f1


if __name__ == '__main__':
    accuracies = np.zeros((56, 5))
    f1scores = np.zeros((56, 5))

    outfilename = ['chi-square', 'gain_ratio',
                   'gini_split', 'info_gain', 'mis_error']

    for fileno, f in enumerate(outfilename):
        print(f'\nBuilding models based on {f}')
        features = np.genfromtxt(
            f'output/{f}.csv', delimiter=',')[:, 20:].astype('int32')

        for i in range(1, 57):
            print(f'    {i}')
            data = np.genfromtxt(f'data/{i}.csv', delimiter=',')
            data = preprocess(data)
            feat = features[i-1, :]

            accuracy = []
            f1 = []

            kfoldsize = data.shape[0] // 10
            for k in range(10):
                data_test = data[k*kfoldsize:(k+1)*kfoldsize, :]
                data_train = np.concatenate((data[0:k*kfoldsize, :], data[(k+1)*kfoldsize:, :]))
                model = fit(data_train, feat)
                prediction = np.zeros((data_test.shape[0]), dtype='int32')
                for j in range(data_test.shape[0]):
                    prediction[j] = predict(model, data_test[j, :20], feat)
                accuracy.append(calculate_accuracy(prediction, data_test[:, -1]))
                f1.append(calculate_f1_score(prediction, data_test[:, -1]))

            accuracies[i-1, fileno] = np.mean(accuracy)
            f1scores[i-1, fileno] = np.mean(f1)

    fname = 'output/accuracies.csv'
    np.savetxt(fname, accuracies, delimiter=',', fmt='%f')

    fname = 'output/f1_scores.csv'
    np.savetxt(fname, f1scores, delimiter=',', fmt='%f')

import matplotlib.pyplot as plt
import numpy as np

f = 'output/chi-square.csv'
rank = np.genfromtxt(f, delimiter=',')[:, 20:]
plt.figure(figsize=(14, 7))
plt.title('Chi-Square')
plt.xticks(np.arange(1, 57), rotation=60)
plt.yticks(np.arange(1, 21))
for i in range(1, 57):
    plt.scatter([i]*5, rank[i-1, :], color='black')
plt.grid()
plt.savefig('output/chi-square.png')
plt.show()
plt.close()

f = 'output/gain_ratio.csv'
rank = np.genfromtxt(f, delimiter=',')[:, 20:]
plt.figure(figsize=(14, 7))
plt.title('Gain Ratio')
plt.xticks(np.arange(1, 57), rotation=60)
plt.yticks(np.arange(1, 21))
for i in range(1, 57):
    plt.scatter([i]*5, rank[i-1, :], color='black')
plt.grid()
plt.savefig('output/gain_ratio.png')
plt.show()
plt.close()

f = 'output/gini_split.csv'
rank = np.genfromtxt(f, delimiter=',')[:, 20:]
plt.figure(figsize=(14, 7))
plt.title('Gini Split')
plt.xticks(np.arange(1, 57), rotation=60)
plt.yticks(np.arange(1, 21))
for i in range(1, 57):
    plt.scatter([i]*5, rank[i-1, :], color='black')
plt.grid()
plt.savefig('output/gini_split.png')
plt.show()
plt.close()

f = 'output/info_gain.csv'
rank = np.genfromtxt(f, delimiter=',')[:, 20:]
plt.figure(figsize=(14, 7))
plt.title('Information Gain')
plt.xticks(np.arange(1, 57), rotation=60)
plt.yticks(np.arange(1, 21))
for i in range(1, 57):
    plt.scatter([i]*5, rank[i-1, :], color='black')
plt.grid()
plt.savefig('output/info_gain.png')
plt.show()
plt.close()

f = 'output/mis_error.csv'
rank = np.genfromtxt(f, delimiter=',')[:, 20:]
plt.figure(figsize=(14, 7))
plt.title('Misclassification Error')
plt.xticks(np.arange(1, 57), rotation=60)
plt.yticks(np.arange(1, 21))
for i in range(1, 57):
    plt.scatter([i]*5, rank[i-1, :], color='black')
plt.grid()
plt.savefig('output/mis_error.png')
plt.show()
plt.close()

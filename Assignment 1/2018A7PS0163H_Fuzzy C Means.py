import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def calculate_U(data_matrix, centers, exponent):
	'''
	Calculates and returns membership matrix U
	'''
	U = []
	for data in data_matrix:
		U_i = []
		for cur_center in centers:
			num = np.linalg.norm(data - cur_center)
			U_ij = []
			for center in centers:
				den = np.linalg.norm(data - center)
				U_ij.append((1 if num == 0 else float('+inf')) if den == 0 else num/den)				
			U_ij = np.array(U_ij) ** (2 / (exponent-1))
			U_ij = 1 / np.sum(U_ij)
			U_i.append(U_ij)
		U.append(U_i)
	return np.array(U)


def fuzzycmean(data_matrix, num_clusters, exponent, num_iteration=20, epsilon=0.1):
	'''
	Function to apply Fuzzy CMeans algorithm
	'''

	# Selecting non-overlapping datapoints to be used as cluster centers
	unique_rows = np.unique(data_matrix, axis=0)
	init_clusters = np.random.choice(len(unique_rows), num_clusters, replace=False)
	centers = unique_rows[init_clusters, :]
	U = None

	# Applying Fuzzy CMeans Algorithm
	for _ in range(num_iteration):
		U = calculate_U(data_matrix, centers, exponent)
		U_m = U ** exponent
		U_m_sum = np.sum(U_m, axis=0).reshape((-1, 1))

		# Error Handling -- If somehow, a cluster center does not contain any point
		if 0 in U_m_sum:
			return fuzzycmean(data_matrix, num_clusters, exponent, num_iteration, epsilon)

		new_centers = np.matmul(U_m.T, data_matrix) / U_m_sum

		# Checking if cluster centers moved below a certain limit
		center_diff = np.sum([np.linalg.norm(centers[i] - new_centers[i]) for i in range(num_clusters)])
		if center_diff < epsilon:
			break
		centers = new_centers

	return centers, U


if __name__ == '__main__':

	result_sil = []
	result_db = []

	for i in range(1, 57):
		print(f'Working on {i}.csv')
		mat = np.genfromtxt(f'data/{i}.csv', delimiter=',')

		# Separating data and label
		X = mat[:, :-1]
		Y = mat[:, -1]

		sil = []
		db = []

		# Applying Fuzzy CMeans with different number of clusters
		for j in range(2, 11):
			print(f'    Working on {j} clusters')
			centers, U = fuzzycmean(X, num_clusters=j, exponent=2)
			membership = U.argmax(axis=1)
			sil.append(silhouette_score(X, membership))
			db.append(davies_bouldin_score(X, membership))

		result_sil.append(sil)
		result_db.append(db)

	result_sil = np.array(result_sil)
	result_db = np.array(result_db)

	# Saving result to csv file
	np.savetxt('fuzzy_silhouette.csv', result_sil, delimiter=',')
	np.savetxt('fuzzy_dbindex.csv', result_db, delimiter=',')

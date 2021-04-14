import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def distance(data_matrix, centers):
	'''
	Computes distance from all data points to all cluster centers
	'''
	dist = []
	for data in data_matrix:
		dist.append([np.linalg.norm(data - center) for center in centers])
	return np.array(dist)


def update_centers(data_matrix, membership, num_clusters):
	'''
	Returns new cluster centers based on membership matrix
	'''
	centers = [np.mean(data_matrix[membership == j, :], axis=0) for j in range(num_clusters)]
	return np.array(centers)


def kmeans(data_matrix, num_clusters, num_iteration=20, epsilon=0.1):
	'''
	Function to apply KMeans algorithm
	'''

	# Selecting non-overlapping datapoints to be used as cluster centers
	unique_rows = np.unique(data_matrix, axis=0)
	init_clusters = np.random.choice(len(unique_rows), num_clusters, replace=False)
	centers = unique_rows[init_clusters, :]
	membership = None

	# Applying KMeans Algorithm
	for _ in range(num_iteration):
		dist = distance(data_matrix, centers)
		membership = dist.argmin(axis=1)

		# Error Handling -- If somehow, a cluster center does not contain any point
		if len(np.unique(membership)) < num_clusters:
			return kmeans(data_matrix, num_clusters, num_iteration, epsilon)

		new_centers = update_centers(data_matrix, membership, num_clusters)

		# Checking if cluster centers moved below a certain limit
		center_diff = np.sum([np.linalg.norm(centers[i] - new_centers[i]) for i in range(num_clusters)])
		if center_diff < epsilon:
			break
		centers = new_centers

	return centers, membership


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

		# Applying KMeans with different number of clusters
		for j in range(2, 11):
			print(f'    Working on {j} clusters')
			centers, membership = kmeans(X, num_clusters=j)
			sil.append(silhouette_score(X, membership))
			db.append(davies_bouldin_score(X, membership))

		result_sil.append(sil)
		result_db.append(db)

	result_sil = np.array(result_sil)
	result_db = np.array(result_db)

	# Saving result to csv file
	np.savetxt('kmeans_silhouette.csv', result_sil, delimiter=',')
	np.savetxt('kmeans_dbindex.csv', result_db, delimiter=',')

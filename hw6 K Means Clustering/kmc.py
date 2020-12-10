import numpy as np
import numpy.matlib
import pandas as pd
import random
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler

def calc_distance(x1, x2):
    return(sum((x1 - x2) ** 2)) ** 0.5

class K_Means:

    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter


    def init_centroids(self, data):
        init_centroids = np.random.choice(range(0, len(data)), self.k, replace=False)
        centroids = []

        for i in init_centroids:
            centroids.append(data.loc[i])

        centroids = np.array(centroids)

        return centroids

    # Assign clusters based on closest centroid
    def assign_centroids(self, centroids, data):
        assigned_centroids = []

        for i in range(0, len(data)):
            distances = []
            for j in range(0, len(centroids)):
                distances.append(calc_distance(centroids[j,:], data[i, :]))

            assigned_centroids.append(np.argmin(distances))
        assigned_centroids = np.array(assigned_centroids)
        return assigned_centroids


    def calc_centroids(self, labels, data):
        new_centroids = []

        # concatenate the cluster labels with the datapoints
        cluster_df = pd.concat([pd.DataFrame(data), pd.DataFrame(labels, columns=['cluster'])], axis=1)

        for c in set(labels):
            current_cluster = cluster_df[cluster_df['cluster'] == c][cluster_df.columns[:-1]]
            cluster_mean = current_cluster.mean(axis=0)
            new_centroids.append(cluster_mean)

        new_centroids = np.array(new_centroids)
        return new_centroids


    def calc_sse(self, labels, data):
        sum_squares = []

        cluster_df = pd.concat([pd.DataFrame(data), pd.DataFrame(labels, columns=['cluster'])], axis=1)

        for c in set(labels):
            current_cluster = cluster_df[cluster_df['cluster'] == c][cluster_df.columns[:-1]]
            cluster_mean = current_cluster.mean(axis=0)
            mean_repmat = np.matlib.repmat(cluster_mean, current_cluster.shape[0], 1)
            sum_squares.append(np.sum(np.sum((current_cluster - mean_repmat) ** 2)))
        return sum_squares


    def kmeans(self, data):
        # choose a random set of initial centroids
        centroids = self.init_centroids(data)
        data = data.to_numpy()
        # create a list of cluster labels for the data
        labels = self.assign_centroids(centroids, data)
        # iterate and update labels
        for i in range(self.max_iter):
            self.centroids = centroids
            centroids = self.calc_centroids(labels, data)

            labels = self.assign_centroids(centroids, data)
            # if centroids does not updata, stop iteration since it is optimized

            if np.array_equal(centroids, self.centroids):
                break
        variance = self.calc_sse(labels, data)
        # sse = np.mean(variance)
        cluster_df = pd.concat([pd.DataFrame(data, columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']), pd.DataFrame(labels, columns=['Cluster'])], axis=1)

        """
        colors=['orange', 'blue', 'green']
        for i in range(len(data)):
            plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(labels[i])])
            plt.scatter(centroids[:,0], centroids[:,1], marker='*', c='r', s=150)
        plt.show()
        """

        return cluster_df, variance


def main():
    """
    Main file to run from the command line.
    """

    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("dataset",
                        default="iris.csv",
                        help="dataset")

    args = parser.parse_args()
    # load the data
    data = pd.read_csv(args.dataset)

    label = data['variety']
    xFeat = data.drop(columns=['variety'])
    k = args.k
    model = K_Means(k, 10)

    xFeat = pd.DataFrame(StandardScaler().fit_transform(xFeat))

    result, variance = model.kmeans(xFeat)

    sse = np.mean(variance)




    print(result)
    print("The sum squared error per cluster is: ", variance)
    print("The mean sum squared error is: ", sse, "for k = ", k)


if __name__ == "__main__":
    main()

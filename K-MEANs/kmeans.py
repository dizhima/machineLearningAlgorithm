"""
K-means clustering.
"""

import numpy as np
import random
from matplotlib import pyplot as plt


def analyze_kmeans():
    """
    Top-level wrapper to iterate over a bunch of values of k and plot the
    distortions and misclassification rates.
    """
    X = np.genfromtxt("digit.txt")
    y = np.genfromtxt("labels.txt", dtype=int)
    distortions = []
    errs = []
    ks = range(1, 11)
    for k in ks:
        distortion, err = analyze_one_k(X, y, k)
        distortions.append(distortion)
        errs.append(err)
    fig, ax = plt.subplots(2, figsize=(8, 6))
    ax[0].plot(ks, distortions, marker=".")
    ax[0].set_ylabel("Distortion")
    ax[1].plot(ks, errs, marker=".")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Mistake rate")
    ax[0].set_title("k-means performance")
    fig.savefig("kmeans.png")


def analyze_one_k(X, y, k):
    """
    Run the k-means analysis for a single value of k. Return the distortion and
    the mistake rate.
    """
    print ("Running k-means with k={0}",format(k))
    clust = cluster(X, y, k)
    print ("Computing classification error.")
    err = compute_mistake_rate(y, clust)
    return clust["distortion"], err


def cluster(X, y, k, n_starts=5):
    """
    Run k-means a total of n_starts times. Returns the results from the run that
    had the lowest within-group sum of squares (i.e. the lowest distortion).

    Inputs
    ------
    X is an NxD matrix of inputs.
    y is a Dx1 vector of labels.
    n_starts says how many times to randomly re-initialize k-means. You don't
        need to change this.

    Outputs
    -------
    The output is a dictionary with the following fields:
    Mu is a kxD matrix of cluster centroids
    z is an Nx1 vector assigning points to clusters. So, for instance, if z[4] =
        2, then the algorithm has assigned the 4th data point to the second
        cluster.
    distortion is the within-group sum of squares, a number.
    """
    def loop(X, i):
        """
        A single run of clustering.
        """
        Mu = initialize(X, k)
        N = X.shape[0]
        z = np.repeat(-1, N)        # So that initially all assignments change.
        while True:
            old_z = z
            z = assign(X, Mu)       # The vector of assignments z.
            Mu = update(X, z, k)    # Update the centroids
            if np.all(z == old_z):
                distortion = compute_distortion(X, Mu, z)
                return dict(Mu=Mu, z=z, distortion=distortion)

    # Main function body
    print ("Performing clustering.")
    results = [loop(X, i) for i in range(n_starts)]
    best = (min(results, key=lambda entry: entry["distortion"]))
    best["digits"] = label_clusters(y, k, best["z"])
    return best


def assign(X, Mu):
    """
    Assign each entry to the closest centroid. Return an Nx1 vector of
    assignments z.
    X is the NxD matrix of inputs.
    Mu is the kxD matrix of cluster centroids.
    """
    # TODO: Compute the assignments z.
    z = []
    for example in X:
        dist=[]
        for point in Mu:
            district=(example-point)**2
            sum=0
            for i in district:
                sum+=i
            dist.append(sum)
        z.append(dist.index(min(dist)))
    z=np.array(z)
    return z


def update(X, z, k):
    """
    Update the cluster centroids given the new assignments. Return a kxD matrix
    of cluster centroids Mu.
    X is the NxD inputs as always.
    z is the Nx1 vector of cluster assignments.
    k is the number of clusters.
    """
    # TODO: Compute the cluster centroids Mu.
    Mu=np.zeros((k,len(X[0])))
    for i in range(k):
        temp=[]
        for j in range(len(X)):
            if z[j]==i:
                temp.append(X[j])
                np.delete(X,j,axis=0)
        length=len(temp)
        mu=np.zeros(len(X[0]))
        for example in temp:
            mu+=np.array(example)
        Mu[i]=np.array(mu)
        Mu[i]=Mu[i]/length
    return Mu


def compute_distortion(X, Mu, z):
    """
    Compute the distortion (i.e. within-group sum of squares) implied by NxD
    data X, kxD centroids Mu, and Nx1 assignments z.
    """
    # TODO: Compute the within-group sum of squares (the distortion).
#     distortion=np.zeros(len(Mu))
    distortion=0

    for i in range(len(Mu)):
        for j in range(len(X)):
            if z[j]==i:#calculate distortion for this cluster
                temp=(X[j]-Mu[i])**2
                sumTemp=0
                for example in temp:
                    sumTemp+=example
                np.delete(X,j,axis=0)
                distortion+=sumTemp
    return distortion


def initialize(X, k):
    """
    Randomly initialize the kxD matrix of cluster centroids Mu. Do this by
    choosing k data points randomly from the data set X.
    """
    # TODO: Initialize Mu.
    [m,n]=X.shape
    listx=[]
    index = random.sample(range(0,m-1),k)
    for idx in index:
        listx.append(X[idx])
    Mu=np.array(listx)
    return Mu


def label_clusters(y, k, z):
    """
    Label each cluster with the digit that occurs most requently for points
    assigned to that cluster.
    Return a kx1 vector labels with the label for each cluster.
    For instance: if 20 points assigned to cluster 0 have label "3", and 40 have
    label "5", then labels[0] should be 5.

    y is the Nx1 vector of digit labels for the data X
    k is the number of clusters
    z is the Nx1 vector of cluster assignments.
    """
    # TODO: Compute the cluster labelings.
    labels = []
    classes = []#store all the label
    for Class in y:
        if Class not in classes:
            classes.append(Class)
    
    for cluster in range(k):
        score=[0]*len(classes)
        for i in range(len(z)):
            if z[i]==cluster:
                score[classes.index(y[i])]+=1
        labels.append(classes[score.index(max(score))])
    
    labels=np.array(labels)
    return labels


def compute_mistake_rate(y, clust):
    """
    Compute the mistake rate as discussed in section 3.4 of the homework.
    y is the Nx1 vector of true labels.
    clust is the output of a run of clustering. Two fields are relevant:
    "digits" is a kx1 vector giving the majority label for each cluster
    "z" is an Nx1 vector of final cluster assignments.
    """
    def zero_one_loss(xs, ys):
        return sum(xs != ys) / float(len(xs))

    y_hat = clust["digits"][clust["z"]]
    return zero_one_loss(y, y_hat)


def main():
    analyze_kmeans()


if __name__ == '__main__':
    main()


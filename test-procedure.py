# For manipulating arrays
import numpy as np
# For working with dataframes
import pandas as pd
# For Machine Learning Algorithms like clustering methods
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
# For working with directories and running external commands
import os
# For visualization
import matplotlib.pyplot as plt
# For creating affinity and Distance Matrices
from linear_algebra_conversion import *
# For graph processing
import networkx as nx 

# Dataset
datasetBankUrl = "http://cs.joensuu.fi/sipu/datasets/"


class dataset:
    name = ""
    downloadURL = ""
    destinationFolder = ""
    destinationFile = ""
    k = 0
    defaultDelim = ""
    data = ""
    targets = ""

    def __init__(self, name, url, destFolder, k, delim="\t"):

        # Sets the given parameters
        self.name = name
        self.downloadURL = url
        self.destinationFolder = destFolder
        self.k = k
        self.defaultDelim = delim

        # Checks wether the dataset exist, in case not, download the dataset
        if self.destinationFolder not in os.listdir():
            os.mkdir("./" + self.destinationFolder)
            os.system("wget -P " + self.destinationFolder + " " +
                      self.downloadURL)

        filelist = os.listdir("./" + self.destinationFolder)

        # Removing DS_Store from file lists, OSX issue
        if ".DS_Store" in filelist:
            filelist.remove(".DS_Store")
        self.destinationFile = filelist[0]
        # Loading the file to a dataframe
        dataframe = pd.read_csv("./" + self.destinationFolder +
                                "/" + self.destinationFile,
                                sep=self.defaultDelim)
        self.data = np.array(dataframe.drop(columns=[dataframe.columns[-1]],
                                            axis=1))

        self.targets = dataframe[dataframe.columns[-1]]


def bestPermution(y1, y2):
    pass


datasets = [dataset("D31", "http://cs.joensuu.fi/sipu/datasets/D31.txt",
            "D31", 31),
            dataset("Jain", "http://cs.joensuu.fi/sipu/datasets/jain.txt",
            "Jain", 2),
            dataset("Flame", "http://cs.joensuu.fi/sipu/datasets/flame.txt",
            "Flame", 2),
            dataset("PathBased", "http://cs.joensuu.fi/sipu/datasets/flame.txt",
            "PathBased", 3),
            dataset("R15", "http://cs.joensuu.fi/sipu/datasets/R15.txt",
            "R15", 15)
            ]

for dataset in datasets:
    print("Testing DataSet: ", dataset.name)

    # Making Affinity Graph and Laplacian
    AdjMat = coreMatrices.affinityMatrix(dataset.data)
    plt.imshow(AdjMat, cmap='jet', norm=plt.Normalize(0, 1))
    plt.savefig("figs/" + dataset.name + "AdjMat.png", dpi=250)
    plt.show()
    Graph = nx.from_numpy_array(AdjMat)

    LaplacianMat = coreMatrices.laplacian(dataset.data)
    plt.imshow(LaplacianMat, cmap='jet', norm=plt.Normalize(-2, 1))
    plt.savefig("figs/" + dataset.name + "Laplacian.png", dpi=250)
    plt.show()

    normalizedAdjMat = coreMatrices.normalizedAdjacencyMatrix(
        dataset.data
    )
    plt.imshow(normalizedAdjMat, cmap='jet', norm=plt.Normalize(0, 1))
    plt.savefig("figs/" + dataset.name + "normAdjMat.png", dpi=250)
    plt.show()

    normalizedLapMat = coreMatrices.noramlizedLaplacianMatrix(
        dataset.data
    )
    plt.imshow(normalizedLapMat, cmap='jet', norm=plt.Normalize(-0.01, 0))
    plt.savefig("figs/" + dataset.name + "normLapMat.png", dpi=250)
    plt.show()

    eVal, eVec = np.linalg.eigh(normalizedLapMat)
    plt.scatter(range(eVal.shape[0]), eVal)
    plt.savefig("figs/" + dataset.name + "eighs.png", dpi=250)

    plt.scatter(range(eVal[:40].shape[0]), eVal[:40])
    plt.savefig("figs/" + dataset.name + "eighs40.png", dpi=250)

    plt.scatter(range(eVal[:10].shape[0]), eVal[:10])
    plt.savefig("figs/" + dataset.name + "eighs10.png", dpi=250)

    # k-Means
    print("Running k-means algorithm")
    kmeansInstance = KMeans(n_clusters=dataset.k)
    kmeansResult = kmeansInstance.fit_predict(dataset.data)
    # insider - isualization of the data under the clustering
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1], c=kmeansResult,
                s=2)
    plt.show()

    # Spectral Clustering
    print("Running Spectral Clustering")
    spectralInstance = SpectralClustering(n_clusters=dataset.k)
    spectralResult = spectralInstance.fit_predict(dataset.data)
    plt.scatter(dataset.data[:, 0], dataset.data[:, 1], c=spectralResult,
                s=2)
    plt.show()

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
            "Jain", 2)
            ]

for dataset in datasets:
    print("Testing DataSet: ", dataset.name)

    #

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

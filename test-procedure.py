# For manipulating arrays
import numpy as np
# For working with dataframes
import pandas as pd
# For working with directories and running external commands
import os

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
        self.data = dataframe.drop(columns=[dataframe.columns[-1]],
                                   axis=1)

        self.targets = dataframe[dataframe.columns[-1]]


datasets = [dataset("D31", "http://cs.joensuu.fi/sipu/datasets/D31.txt",
            "D31", 31)
            ]

import numpy
import math


class distance:
    @staticmethod
    def norm(firstPoint, secondPoint, norm=2):
        # Converting to numpy array
        if isinstance(firstPoint, numpy.ndarray):
            firstPoint = numpy.array(firstPoint)
        if isinstance(secondPoint, numpy.ndarray):
            secondPoint = numpy.array(secondPoint)

        diff = firstPoint - secondPoint
        result = 0
        for i in diff:
            result += abs(i)**norm
        result = result ** (1/norm)
        return result

    @staticmethod
    def mahalanobis(first, second):
        pass

    @staticmethod
    def cosine(first, second):
        pass


class kernels:
    @staticmethod
    def distanceInverse(firstPoint, secondPoint):
        return 1/distance.norm(firstPoint, secondPoint)


class coreMatrices:
    @staticmethod
    def distanceMatrix(data, metric, additional_arg):
        # TODO: it should work with different metrics and ...
        result = numpy.zeros((data.shape[0], data.shape[0]),
                             dtype=numpy.float)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if metric == "norm":
                    result[i][j] = distance.norm(data[i], data[j],
                                                 additional_arg)
        return result

    @staticmethod
    def affinityMatrix(data, kernelFunction=kernels.distanceInverse):
        # TODO: it should work with different kernels
        result = numpy.zeros((data.shape[0], data.shape[0]),
                             dtype=numpy.float)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] =\
                    kernelFunction(data[i], data[j])
        for i in range(result.shape[0]):
            result[i][i] = 0
        return result

    @staticmethod
    def laplacian(data, kernelFunction=kernels.distanceInverse):
        temp = coreMatrices.affinityMatrix(data, kernelFunction)
        diag = numpy.diag(temp.sum(axis=1))
        return diag - temp

    @staticmethod
    def normalizedAdjacencyMatrix(data,
                                  kernelFunction=kernels.distanceInverse):
        temp = coreMatrices.affinityMatrix(data, kernelFunction)
        diag = temp.sum(axis=1)
        invDiag = np.diag(
            np.array([1/math.sqrt(elem) for elem in diag])
            )
        nAdj = numpy.matmul(invDiag, temp)
        nAdj = numpy.matmul(nAdj, invDiag)
        return nAdj

    @staticmethod
    def noramlizedLaplacianMatrix(data,
                                  kernelFunction=kernels.distanceInverse):
        temp = coreMatrices.normalizedAdjacencyMatrix(data)
        return numpy.identity(temp.shape[0]) - temp
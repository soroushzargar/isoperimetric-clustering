from linear_algebra_conversion import coreMatrices
import networkx


class IsoperimetricClustering:
    __n_clusters = 0
    __spanningTreeNumber = 1

    def __init__(self, n_clusters, spanningTreeNumber=0):
        self.__n_clusters = n_clusters
        self.__spanningTreeNumber = spanningTreeNumber

    def fit_predict(self, X, metric="norm", additional_arg=2):
        # Creating an affinity matrix
        distMat = coreMatrices.distanceMatrix(X, metric=metric,
                                              additional_arg=additional_arg)
        graph = networkx.from_numpy_matrix(distMat)

        mst = None
        for sp in range(spanningTreeNumber + 1):
            sp = networkx.minimum_spanning_tree(graph)
            mst = sp
            for edge in sp.edges:
                graph.remove_edge(edge[0], edge[1])
        return mst


class MinCut:
    __n_clusters = 2

    def __init__(self, k):
        self.__n_clusters = k

    def fit_predict(self, X):
        adjMat = coreMatrices.affinityMatrix(X)
        graph = networkx.from_numpy_matrix(adjMat)
        edgeCut = networkx.minimum_edge_cut(graph)
        for edge in edgeCut:
            graph.remove_edge(edge[0], edge[1])
        cc = list(networkx.connected_components(graph))
        i = 0
        target = [-1] * X.shape[0]
        for c in list(cc):
            comp = list(c)
            for elem in c:
                target[elem] = i
            i += 1
        return target

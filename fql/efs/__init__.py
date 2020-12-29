# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 17:49:57 2020

@author: jhost
"""

"""
The :mod:`sklearn.cluster` module gathers popular unsupervised clustering
algorithms.
"""

from .empirical_fuzzy_set import EmpiricalFuzzySet
#from ._affinity_propagation import affinity_propagation, AffinityPropagation
#from ._agglomerative import (ward_tree, AgglomerativeClustering,
#                             linkage_tree, FeatureAgglomeration)
#from ._kmeans import k_means, KMeans, MiniBatchKMeans
#from ._dbscan import dbscan, DBSCAN
#from ._optics import (OPTICS, cluster_optics_dbscan, compute_optics_graph,
#                      cluster_optics_xi)
#from ._bicluster import SpectralBiclustering, SpectralCoclustering
#from ._birch import Birch

__all__ = ['EmpiricalFuzzySet']
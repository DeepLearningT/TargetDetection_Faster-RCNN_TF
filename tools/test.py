# -*- coding:utf-8 -*-
import scipy.sparse

__author__ = 'tonye'
import numpy as np

overlaps = np.zeros((3, 4), dtype=np.float32)
print(overlaps)
overlaps[:, 2] = 1.0
print(overlaps)
overlaps = scipy.sparse.csr_matrix(overlaps)
print(overlaps.toarray())
print(overlaps.toarray().max(axis=1))
print(overlaps.toarray().argmax(axis=1))
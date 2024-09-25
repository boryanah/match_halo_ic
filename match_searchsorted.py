#!/usr/bin/env python

from numpy import *
from numba import jit
import numpy as np

# og
dtype = np.int32
#dtype = np.int64 # needed for huge box ph201

@jit(nopython=True)
def match(arr1, arr2, arr2_sorted=False, arr2_index=None):
    """
    For each element in arr1 return the index of the element with the
    same value in arr2, or -1 if there is no element with the same value.
    Setting arr2_sorted=True will save some time if arr2 is already sorted
    into ascending order.

    A precomputed sorting index for arr2 can be supplied using the
    arr2_index parameter. This can save time if the routine is called
    repeatedly with the same arr2 but arr2 is not already sorted.

    It is assumed that each element in arr1 only occurs once in arr2.
    """

    #arr1_n = arr1
    #arr2_n = arr2

    # Sort arr2 into ascending order if necessary
    tmp1 = arr1
    if arr2_sorted:
        tmp2 = arr2
    else:
        if arr2_index is None:
            idx = argsort(arr2)
            tmp2 = arr2[idx]
        else:
            # Use supplied sorting index
            idx = arr2_index
            tmp2 = arr2[arr2_index]

    # Find where elements of arr1 are in arr2
    ptr  = searchsorted(tmp2, tmp1)

    # Make sure all elements in ptr are valid indexes into tmp2
    # (any out of range entries won't match so they'll get set to -1
    # in the next bit)
    ptr[ptr>=len(tmp2)] = 0
    ptr[ptr<0]          = 0

    # Return -1 where no match is found
    ptr[where(tmp2[ptr] != tmp1)[0]] = -1
    
    # Put ptr back into original order
    if arr2_sorted:
        ptr = where(ptr>= 0, arange(len(tmp2))[ptr], -1)
    else:
        ind = arange(len(arr2))[idx]
        ptr = where(ptr>= 0, ind[ptr], -1)

    return ptr



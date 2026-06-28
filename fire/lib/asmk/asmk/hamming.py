import numpy as np
import numba

@numba.njit(fastmath=True)
def c_count_bits(n):
    n = (n & 0x55555555) + ((n >> 1) & 0x55555555)
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
    n = (n & 0x0f0f0f0f) + ((n >> 4) & 0x0f0f0f0f)
    n = (n & 0x00ff00ff) + ((n >> 8) & 0x00ff00ff)
    n = (n & 0x0000ffff) + ((n >> 16) & 0x0000ffff)
    return n

@numba.njit(fastmath=True)
def c_binarize_and_pack_uint32(arr, threshold):
    tmp = np.uint32(0)
    for i in range(arr.shape[0]):
        tmp = (tmp << np.uint32(1)) + np.uint32(arr[i] > threshold)
    return tmp

@numba.njit(fastmath=True)
def c_hamming_dist_uint32_arr(n1, n2, normalization):
    length = n1.shape[0]
    if normalization == 0.0:
        normalization = float(length * 32)
    s = 0
    for i in range(length):
        s += c_count_bits(n1[i] ^ n2[i])
    return float(s) / normalization

@numba.njit(fastmath=True)
def binarize_and_pack(arr, threshold=0):
    dim_orig = arr.shape[0]
    dim = int(np.ceil(dim_orig / 32.0))
    result = np.zeros(dim, dtype=np.uint32)
    
    offset = 0
    for i in range(dim - 1):
        result[i] = c_binarize_and_pack_uint32(arr[offset:offset+32], threshold)
        offset += 32
        
    tmp = c_binarize_and_pack_uint32(arr[offset:], threshold)
    result[dim - 1] = tmp << np.uint32(offset + 32 - dim_orig)
    return result

@numba.njit(fastmath=True)
def binarize_and_pack_2D(arr, threshold=0):
    dim0 = arr.shape[0]
    dim1_orig = arr.shape[1]
    dim1 = int(np.ceil(dim1_orig / 32.0))
    result = np.zeros((dim0, dim1), dtype=np.uint32)
    
    for i in range(dim0):
        offset = 0
        for j in range(dim1 - 1):
            result[i, j] = c_binarize_and_pack_uint32(arr[i, offset:offset+32], threshold)
            offset += 32
            
        tmp = c_binarize_and_pack_uint32(arr[i, offset:], threshold)
        result[i, dim1 - 1] = tmp << np.uint32(offset + 32 - dim1_orig)
    return result

@numba.njit(fastmath=True)
def hamming_dist_packed(n1, n2, normalization=0.0):
    return c_hamming_dist_uint32_arr(n1, n2, normalization)

@numba.njit(parallel=True, fastmath=True)
def hamming_cdist_packed(arr1, arr2, normalization=0.0):
    dim0 = arr1.shape[0]
    dim1 = arr2.shape[0]
    result = np.zeros((dim0, dim1), dtype=np.float32)
    for i in numba.prange(dim0):
        for j in range(dim1):
            result[i, j] = c_hamming_dist_uint32_arr(arr1[i], arr2[j], normalization)
    return result

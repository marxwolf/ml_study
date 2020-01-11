import numpy as np

def get_top_k_index(arr, k_rate):
    """
    Get top k element index in 1-D or 2-D ndarray
    """
    arr = np.absolute(arr)
    shape = list(arr.shape)
    if len(shape) == 1:
        if k_rate > 1:
            k = k_rate
        else:
            k = int(shape[0] * k_rate)
        if k < 1:
            k = 1
        index = np.argpartition(arr, -k)[-k:]
    elif len(shape) == 2:
        """
        flatten to get global top-k
        """
        if k_rate > 1:
            k = k_rate
        else:
            k = int(shape[0] * shape[1] * k_rate)
        if k < 1:
            k = 1
        flatten_arr = arr.flatten()
        flatten_idx = np.argpartition(flatten_arr, -k)[-k:]
        row_len = shape[0]
        column_len = shape[1]
        index = [divmod(i, column_len) for i in flatten_idx]

        """
        To get local top-k each column
        """
        # if k_rate > 1:
        #     k = k_rate
        # else:
        #     k = int(shape[0] * k_rate)
        # re_arr = arr.T
        # index = []
        # for i, column_data in enumerate(re_arr):
        #     local_index = [(j, i) for j in np.argpartition(column_data, -k)[-k:]]
        #     index.extend(local_index)

    else:
        raise "Not Implemented."
    return index

def get_top_k_array(arr, k_rate):
    top_k_array = np.zeros_like(arr)
    index = get_top_k_index(arr, k_rate)
    for i in index:
        top_k_array[i] = arr[i]
    return top_k_array
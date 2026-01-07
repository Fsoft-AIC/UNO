import numpy as np

def bitmasks2bboxes(bitmasks):
    bitmasks_array = bitmasks
    boxes = np.zeros((bitmasks_array.shape[0], 4), dtype=np.float32)
    x_any = np.any(bitmasks_array, axis=1)
    y_any = np.any(bitmasks_array, axis=2)
    for idx in range(bitmasks_array.shape[0]):
        x = np.where(x_any[idx, :])[0]
        y = np.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = np.array((x[0], y[0], x[-1], y[-1]),
                                     dtype=np.float32)
    return boxes
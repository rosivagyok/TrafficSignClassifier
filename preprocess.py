import cv2 
import numpy as np

def preprocess_features(X, equalize_hist=True):

    # convert RGB color space to YUV color space.
    # YUV color space holds chrominance elements, better to make model lighting invariant
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    # adjust image contrast, by equalizing bins in 0-255 monochrome hist. range
    if equalize_hist:
        # for each sample in training data equalize hist. of 8-bit greyscale images
        X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])

    X = np.float32(X)

    # standardize/normalize features by using: Standard Score(X) = (X-mean)/standard deviation(X) -> assumes normal dist.
    # subtract mean of data from data
    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + np.finfo('float32').eps)

    return X
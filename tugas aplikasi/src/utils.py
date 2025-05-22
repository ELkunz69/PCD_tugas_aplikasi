import numpy as np

def combine_features(cnn_features, shape_features):
    shape_features_norm = shape_features / np.linalg.norm(shape_features)
    combined = np.concatenate([cnn_features, shape_features_norm])
    return combined

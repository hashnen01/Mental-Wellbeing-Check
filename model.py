import pickle

import config


_model = None
_scaler = None
_feature_names = None


def load_model():
    """Load the trained model, scaler, and feature list from disk."""
    global _model, _scaler, _feature_names

    with open(config.MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(config.SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    with open(config.FEATURE_LIST_PATH, "rb") as f:
        _feature_names = pickle.load(f)


def get_model():
    """Return the trained model (lazy load on first call)."""
    if _model is None:
        load_model()
    return _model


def get_scaler():
    """Return the fitted scaler."""
    if _scaler is None:
        load_model()
    return _scaler


def get_feature_names():
    """Return the ordered list of feature names the model expects."""
    if _feature_names is None:
        load_model()
    return _feature_names

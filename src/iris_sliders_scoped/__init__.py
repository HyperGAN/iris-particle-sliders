"""Explicit cached Iris scoped-v1 APIs; original Iris routes remain unchanged."""
PROFILE = 'iris-explicit-cached-scoped-v1'

def open_training(*args, **kwargs):
    from .training import Training
    return Training(*args, **kwargs)

def open_inference(*args, **kwargs):
    from .inference import Inference
    return Inference(*args, **kwargs)

def save_prepared_cache(*args, **kwargs):
    from .cache import save_prepared_cache as save
    return save(*args, **kwargs)

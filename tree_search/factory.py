import copy
def safe_deepcopy_env(obj):
    """
        Perform a deep copy of an environment but without copying its viewer.
    """
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        if k not in ['viewer']:
            setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result

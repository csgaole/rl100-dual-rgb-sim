import numpy as np

def true_terminal(done, info):
    """Old Gym: preserve bootstrap at TimeLimit truncation, stop GAE at all done boundaries."""
    return bool(done) and not bool(np.asarray(info.get('TimeLimit.truncated', False)).any())

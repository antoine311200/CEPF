import numpy as np

class Constraint(np.ndarray):
    """A constraint represented as a numpy array with a target value."""
    def __new__(cls, values: np.ndarray, target: float) -> "Constraint":
        obj = np.asarray(values).view(cls)
        obj.target = target
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.target = getattr(obj, 'target', None)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)
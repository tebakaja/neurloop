import numpy as np
from typing import Tuple


def splitting_dataset(
  sequences: np.array,
  labels:    np.array
) -> Tuple[np.array]:
  train_size      = int(len(sequences) * 0.8)

  X_train, X_test = sequences[:train_size], sequences[train_size:]
  y_train, y_test = labels[:train_size], labels[train_size:]

  return X_train, X_test, y_train, y_test
import numpy as np
from typing import Tuple
from pandas import DataFrame


def sliding_window(
  dataframe: DataFrame,
  seq_len:   int = 60
) -> Tuple[np.array]:
  labels    = []
  sequences = []

  for i in range(len(dataframe) - seq_len):
    label    = dataframe.iloc[seq_len + i].values[0]
    sequence = dataframe.iloc[i:(seq_len + i)].values

    labels.append(label)
    sequences.append(sequence)

  return np.array(sequences), np.array(labels)
  
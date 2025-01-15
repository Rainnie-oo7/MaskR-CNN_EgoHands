import torch
import numpy as np

fin_list = []
tensor_boxes = torch.empty((0, 4), dtype=torch.float32)
print(tensor_boxes)
print(type(tensor_boxes))
print(tensor_boxes.size(0))
for i in range(4):
    fin_list.append(tensor_boxes)
print(fin_list)
print(len(fin_list))
print(sum(fin_list))



""""
a = [
    [1, 2, 3, 4, 5],
    [6],
    [7],
    [8, 9, 10, 11, 12]
]

# Finde die maximale Länge
max_len = max(len(row) for row in a)

# Fülle jede Zeile auf
result = [row + [[] for _ in range(max_len - len(row))] for row in a]

# Ausgabe
print(result)


a = [
    [1, 2, 3, 4, 5],
    [6],
    [7],
    [8, 9, 10, 11, 12]
]

# Finde die maximale Länge
max_len = max(len(row) for row in a)

# Ersetze jede Zeile durch leere Listen
result = [[[] for _ in range(max_len)] for _ in a]

# Ausgabe
print(result)
"""
from pyparsing import Empty

##v Wechsle alle (Einträge der) Zeilen, die kürźer sind, mit [] aus. dabei werden alle Einträge negliert und
"""
a = [
    [1, 2, 3, 4, 5],
    [6],
    [7],
    [8, 9, 10, 11, 12]
]

# Finde die maximale Länge
max_len = max(len(row) for row in a)

# Ersetze nur die kürzeren Zeilen mit `[]`
result = [row if len(row) == max_len else [[] for _ in range(max_len)] for row in a]

# Ausgabe
print(result)
"""
##Gebe den Index der längsten Zeile aus:
# a = [
#     [1, 2, 3, 4, 5],
#     [6],
#     [7],
#     [8, 9, 10, 11, []]
# ]

# Finde die maximale Länge
# max_len = max(len(row) for row in a)

# for i, row in enumerate(a):
#     max_len = max(len(row) for row in a)
#     if len(row) == max_len:
#         max_index = i
#         # max_index = next(i for i, row in enumerate(a) )
#     print(max_len, max_index)
# Ersetze nur die kürzeren Zeilen mit `[]`
# result = [row if len(row) == max_len else [[] for _ in range(max_len)] for row in a]
# Ausgabe
# print(result)
####################################
# import torch    #funz net TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.
# import numpy as np
# empty_arrays = np.empty(4, dtype=object)
# # Leere Listen zuweisen
# for i in range(4):
#     empty_arrays[i] = []
#
# print(empty_arrays)
# print(torch.tensor(empty_arrays))
# print(torch.tensor(empty_arrays))
#
# empty_arrays = [[] for _ in range(4)]
# print(empty_arrays)
#############################################
# Instead of NaN, initialize with zeros or another valid value
# import torch
# empty_tensor = torch.zeros(4)  # Tensor of shape (4,) filled with 0
# print(empty_tensor)  # tensor([0., 0., 0., 0.])

import torch
from torchvision import tv_tensors

# empty_list = [[] for _ in range(4)]
# print(empty_list)
# print(empty_list)
# t = torch.tensor(empty_list)
# print(t)
# print(t)
# a = torch.tensor([[], [], [], []])
# tensor_boxesml = tv_tensors.BoundingBoxes(a, format="XYXY", canvas_size=[720, 1280])
# tensor = tensor_boxesml.unsqueeze(0)
# print(tensor)
# print(tensor)
#############################################
# import torch
#
# # Tensor mit numerischen Platzhaltern (z. B. NaN)
# empty_tensor = torch.full((4,), float('nan'))  # Ein Tensor mit Shape (4,) gefüllt mit NaN
# print(empty_tensor)
# print(empty_tensor)
# print(empty_tensor)
#############################################


from torch.nn import LogSoftmax
import torch
from torch.nn import CrossEntropyLoss
# Example of target with class indices
loss = CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input)
print(target)
# print(loss(input, target))
output = loss(input, target)
print(output.item())
exit()
output.backward()

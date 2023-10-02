import numpy as np
import pomegranate.hmm as hmm
import pomegranate.distributions as distributions
from Data.Readers.brown_corpus_reader import BCReader
import Omission.utils as omitter
from Models.hmmlearn_wrapper import hmmlearn_wrapper
from Evaluations import utils as evaluations
import torch
from torch.masked import MaskedTensor

tensor1 = torch.randn(3, 1)
tensor2 = torch.randn(2, 1)
tensor3 = torch.randn(3, 1)
tensor4 = torch.randn(2, 1)
tensor5 = torch.randn(3, 1)
tensor6 = torch.randn(2, 1)

mask1 = torch.rand(3, 1) > 0.5
mask2 = torch.rand(2, 1) > 0.5
mask3 = torch.rand(3, 1) > 0.5
mask4 = torch.rand(2, 1) > 0.5
mask5 = torch.rand(3, 1) > 0.5
mask6 = torch.rand(2, 1) > 0.5

masked_tensor1 = MaskedTensor(tensor1, mask1)
masked_tensor2 = MaskedTensor(tensor2, mask2)
masked_tensor3 = MaskedTensor(tensor3, mask3)
masked_tensor4 = MaskedTensor(tensor4, mask4)
masked_tensor5 = MaskedTensor(tensor5, mask5)
masked_tensor6 = MaskedTensor(tensor6, mask6)



tensors = []
tensors.append(masked_tensor1)
tensors.append(masked_tensor2)
tensors.append(masked_tensor3)
tensors.append(masked_tensor4)
tensors.append(masked_tensor5)
tensors.append(masked_tensor6)


#new_tensor= torch.cat([torch.unsqueeze(masked_tensor, dim=0) for masked_tensor in tensors if masked_tensor.shape[0]==3], dim=0)


tensor_dict = {}
for tensor in tensors:
    if tensor.shape[0] not in tensor_dict:
        tensor_dict[tensor.shape[0]] = torch.unsqueeze(tensor, dim=0)
    else:
        tensor_dict[tensor.shape[0]] = torch.cat((tensor_dict[tensor.shape[0]], torch.unsqueeze(tensor, dim=0)))
values = list(tensor_dict.values())
print(values[0].shape)
print(values[1].shape)

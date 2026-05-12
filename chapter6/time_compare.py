import torch
import timeit

M = torch.rand(1000, 1000)
print(timeit.timeit(lambda: M.mm(M).mm(M), number=5000))

N = torch.rand(1000, 1000).cuda()
print(timeit.timeit(lambda: N.mm(N).mm(N), number=5000))

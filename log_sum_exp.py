import torch

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    return x.exp() / x.exp().sum(dim=-1, keepdims=True)


#x = torch.randn(10)
x = torch.randn((2, 2))
a = torch.softmax(x, dim=-1)
b = naive_softmax(x)

print('a', a)
print('b', b)
print('allclose', torch.allclose(a, b, atol=1e-6)) # 判断两个张量是否足够接近
# |a - b| <= atol + rtol * |b|, and rtol=1e-05. NOTE 
# 如此就是判断这两个张量的“距离”了。

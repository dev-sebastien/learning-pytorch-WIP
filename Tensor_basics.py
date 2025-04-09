import torch
import numpy as np

# Initializing tensors:
tensor1 = torch.empty(2,2)
print(tensor1)

tensor2 = torch.rand(2)
print("tensor 2: ", tensor2)

tensor3 = torch.ones(3,3, dtype = torch.float16)
print("tensor3: ", tensor3)
print("tensor3 type: ", tensor3.dtype)
print("tensor3 size: ", tensor3.size())


#tensor form data:
tensor4 = torch.tensor([1,2,3])
print("tensor4: ", tensor4)


#Basic operations: torch.div .mul etc
x = torch.rand(2,2)
y = torch.rand(2,2)
print("x and y: ", x, y)

w = x + y
print("w: ", w)

w1 = torch.add(x,y)
print("w1: ", w1)

y.add_(x) #inplace modification -> _
print("y modified:\n", y)

#Slicing operations:
h = torch.rand(4,3)
print("h:\n", h)
print("h sliced:\n", h[:,:1])
print("1 item of h \n", h[2,2].item())

#Reshape tensor:
j = torch.rand(4,4)
print("h:\n", j)
i = j.view(16) #with size 16 and thus 1 dim -> num of elements need to add up
print("i: \n", i)
#without specifying use -1 and then how many numbers per dimension, the dim will be determined automatically.
o = j.view(-1, 8)
print("o: \n", o)


#converting from numpy to torch and other way around
ten = torch.ones(6)
print("ten: \n", ten)
ton = ten.numpy()
print("ton: \n", ton)

ten.add_(1) #add one to each element
print("ten add 1: \n", ten)

ton = ten.numpy()
print("ton check?: \n", ton)
#the add 1 is also added to this because ten and ton point to the 
#same memory location. This happens when the tensor is on the gpu.

pen = np.ones(2)
print("pen: \n", pen)
pin = torch.from_numpy(pen)
print("pin: \n", pin)

print(torch.cuda.is_available())

#requires_grad = True to let pytorch know that gradients will be calculated later for this tensor.
k = torch.ones(5, requires_grad=True) #by default =False
print("k: \n", k)
#Tutorial on the Autograd package to calculate gradients:

import torch

x = torch.rand(4, requires_grad=True)
print("x: \n", x)

y = x + 1
print("y: \n", y) # y: tensor([1.7261, 1.1256, 1.8013, 1.7745], grad_fn=<AddBackward0>

z = y*y*2
z = z.mean()
print("z: \n", z) #z: tensor([4.2659, 2.7876, 4.4741, 4.0453], grad_fn=<MulBackward0>)

z.backward() # dz/dx
print("gradients of x: ", x.grad)


#when we want the gradient of a matrix we need a gradient argument
x = torch.rand(3, requires_grad=True)
print("x: \n", x)

y = x + 1
print("y: \n", y) # y: tensor([1.7261, 1.1256, 1.8013, 1.7745], grad_fn=<AddBackward0>

z = y*y*2
print("z: \n", z) #z: tensor([4.2659, 2.7876, 4.4741, 4.0453], grad_fn=<MulBackward0>)

v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32) 

z.backward(v) # Vector jacobian product
print("gradients of x: ", x.grad)

#Prevent pytorch from tracking the history: 3 options
p = torch.rand(3, requires_grad=True)
print("p: ", p)

#1) p.requires_grad_(False)
#2) p.detach()
#3) with torch.no_grad(): 


#When we call the backwards function the gradients for the tensor will be accumulated into .grad attribute

weights = torch.ones(4, requires_grad = True)

for epoch in range(2):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() #so gradients don't accumulate


optimizer = torch.optim.SGD(weights, lr = 0.01)
optimizer.step()
optimizer.zero_grad()
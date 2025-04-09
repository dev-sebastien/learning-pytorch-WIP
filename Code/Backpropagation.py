import torch

# 1) Forward pass: Compute Loss
# 2) Compute local gradients
# 3) Backward pass: Compute dLoss/dWeights using the chain rule

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

#Forward pass:
y_hat = w * x
loss = (y_hat - y)**2
print("loss: ", loss)

#Backward pass:
loss.backward()
print("w.grad: ", w.grad)

#Update weights and do next forward and backward passes ...


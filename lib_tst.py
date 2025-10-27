# test 

import numpy as np
import torch 
import diffusers
import transformers

array = np.array([1, 2, 3])
print(array)

print("Hello, World!")

tensor = torch.tensor([1, 2, 3])
print(tensor)

# test cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor_cuda = tensor.to(device)
    print(tensor_cuda)

    print("CUDA is available. Tensor moved to GPU.")
else:
    print("CUDA is not available.")


print("Torch:", torch.__version__)
print("Diffusers:", diffusers.__version__)
print("Transformers:", transformers.__version__)

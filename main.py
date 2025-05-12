# Print confirmation message
print("\n✅ main.py is being executed")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from experiment import run_experiment
run_experiment(1)
# run_experiment(1)
# run_experiment((0, 1), runs=2)
# run_experiment([1, 2], runs=2)

# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name())

# import os
# import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# print("CUDA available:", torch.cuda.is_available())
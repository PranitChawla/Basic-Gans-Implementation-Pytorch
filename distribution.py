import torch
from torch.autograd import Variable
import numpy as np
data_mean=100
data_stddev=1.25

matplotlib_is_available = True
try:
	from matplotlib import pyplot as plt
except ImportError:
	print("Will skip plotting; matplotlib is not available.")
	matplotlib_is_available = False


def get_distribution_sampler(mu,sigma):
	return lambda n: torch.Tensor(np.random.normal(mu,sigma,(1,n)))
d_sampler=get_distribution_sampler(data_mean,data_stddev)
d_real_data=Variable(d_sampler(100000))

def get_generator_input_sampler():
	return lambda m,n: torch.rand(m,n)

gi_sampler=get_generator_input_sampler()
d_gen_input=Variable(gi_sampler(1,10000000))

print (d_real_data)
plt.hist(d_gen_input, bins=50)
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Histogram of Generated Distribution')
plt.grid(True)
plt.show()
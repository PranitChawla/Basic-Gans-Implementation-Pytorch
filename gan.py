import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable
data_mean=4
data_stddev=1.25

print(torch.cuda.is_available())
(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)
matplotlib_is_available = True
try:
	from matplotlib import pyplot as plt
except ImportError:
	print("Will skip plotting; matplotlib is not available.")
	matplotlib_is_available = False


def get_moments(d):
    # Return the first 4 moments of the data provided
	mean = torch.mean(d)
	diffs = d - mean
	var = torch.mean(torch.pow(diffs, 2.0))
	std = torch.pow(var, 0.5)
	zscores = diffs / std
	skews = torch.mean(torch.pow(zscores, 3.0))
	kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # excess kurtosis, should be 0 for Gaussian
	final = torch.cat((mean.reshape(1,), std.reshape(1,), skews.reshape(1,), kurtoses.reshape(1,)))
	return final

def get_distribution_sampler(mu,sigma):
	return lambda n: torch.Tensor(np.random.normal(mu,sigma,(1,n)))
def get_generator_input_sampler():
	return lambda m,n: torch.rand(m,n)
def extract(v):
    return v.data.storage().tolist()
class Generator (nn.Module):
	def __init__ (self,input_size,hidden_size,output_size,f):
		super(Generator,self).__init__()
		self.map1=nn.Linear(input_size,hidden_size)
		self.map2=nn.Linear(hidden_size,hidden_size)
		self.map3=nn.Linear(hidden_size,output_size)
		self.f=f
	def forward (self,x):
		x=self.map1(x)
		x=self.f(x)
		x=self.map2(x)
		x=self.f(x)
		x=self.map3(x)
		return x
class Discriminator (nn.Module):
	def __init__(self,input_size,hidden_size,output_size,f):
		super(Discriminator,self).__init__()
		self.map1=nn.Linear(input_size,hidden_size)
		self.map2=nn.Linear(hidden_size,hidden_size)
		self.map3=nn.Linear(hidden_size,output_size)
		self.f=f
	def forward (self,x):
		x=self.map1(x)
		x=self.f(x)
		x=self.map2(x)
		x=self.f(x)
		x=self.map3(x)
		x=self.f(x)
		return x
def train ():
	g_input_size=1
	g_hidden_size=5
	g_output_size=1
	d_input_size=500
	d_hidden_size=10
	d_output_size=1
	minibatch_size=d_input_size
	d_learning_rate=0.001
	g_learning_rate=0.001
	sgd_momentum=0.9
	num_epochs=5000
	print_interval=100
	d_steps=20
	g_steps=20
	dfe,dre,ge=0,0,0
	d_real_data,d_fake_data,g_fake_data=None,None,None
	discriminator_activation_function=torch.sigmoid
	generator_activation_function=torch.tanh
	d_sampler=get_distribution_sampler(data_mean,data_stddev)
	gi_sampler=get_generator_input_sampler()
	G=Generator(input_size=g_input_size,hidden_size=g_hidden_size,output_size=g_output_size,f=generator_activation_function)
	D=Discriminator(input_size=d_input_func(d_input_size),hidden_size=d_hidden_size,output_size=d_output_size,f=discriminator_activation_function)
	criterion=nn.BCELoss()
	d_optimizer=optim.SGD(D.parameters(),lr=d_learning_rate,momentum=sgd_momentum)
	g_optimizer=optim.SGD(G.parameters(),lr=g_learning_rate,momentum=sgd_momentum)
	# print (d_sampler,gi_sampler)
	for epoch in range(num_epochs):
		for d_index in range (d_steps):
			D.zero_grad()
			d_real_data=Variable(d_sampler(d_input_size))
			# print ("input size ",d_input_func(d_input_size))
			d_real_decision=D(preprocess(d_real_data))
			# print (d_real_decision)
			d_real_error=criterion(d_real_decision,Variable(torch.ones([1,1])))
			d_real_error.backward()

			d_gen_input=Variable(gi_sampler(minibatch_size,g_input_size))
			d_fake_data=G(d_gen_input).detach()
			d_fake_decision=D(preprocess(d_fake_data.t()))
			d_fake_error=criterion(d_fake_decision,Variable(torch.zeros([1,1])))
			d_fake_error.backward()
			d_optimizer.step()

			dre,dfe=extract(d_real_error)[0],extract(d_fake_error)[0]

		for g_index in range (g_steps):
			G.zero_grad()
			gen_input=Variable(gi_sampler(minibatch_size,g_input_size))
			g_fake_data=G(gen_input)
			dg_fake_decision=D(preprocess(g_fake_data.t()))
			g_error=criterion(dg_fake_decision,Variable(torch.ones([1,1])))
			g_error.backward()
			g_optimizer.step()
			ge=extract(g_error)[0]

		if epoch%print_interval == 0 :
			print (epoch,dfe,dre,ge)

	if matplotlib_is_available:
		print("Plotting the generated distribution...")
		values = extract(g_fake_data)
		print(" Values: %s" % (str(values)))
		plt.hist(values, bins=50)
		plt.xlabel('Value')
		plt.ylabel('Count')
		plt.title('Histogram of Generated Distribution')
		plt.grid(True)
		plt.show()

train()
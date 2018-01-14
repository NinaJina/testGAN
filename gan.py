import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import io

from sampler import sampler,generate_lut

num_epochs = 3000
d_steps = 
g_steps = 

d_input_dim = 2
d_hidden_size = 6
d_output_size = 2

g_input_dim = 2
g_hidden_size = 6
g_output_size = 2


class Generator(nn.Modules):
	def __init__(self,input_size, hidden_size, output_size):
		super(Generator,self).__init__()
		self.map1 = nn.Linear(input_size,hidden_size)
		self.map2 = nn.Linear(hidden_size,hidden_size)
		self.map3 = nn.Linear(hidden_size,output_size)

	def forward(x):
		x = self.map1(x)
		x = F.elu(x)
		x = self.map2(x)
		x = F.sigmoid(x)
		x = self.map3(x)
		return x

class Discriminator(nn.Modules):
	def __init__(self,input_size, hidden_size, output_size):
		super(Generator,self).__init__()
		self.map1 = nn.Linear(input_size,hidden_size)
		self.map2 = nn.Linear(hidden_size,hidden_size)
		self.map3 = nn.Linear(hidden_size,output_size)

	def forward(x):
		x = self.map1(x)
		x = F.elu(x)
		x = self.map2(x)
		x = F.elu(x)
		x = self.map3(x)
		x = F.sigmoid(x)
		return x




discriminator = Discriminator(d_input_dim,d_hidden_size,d_output_size)
generator = Generator(g_input_dim,g_hidden_size,g_output_size)
criterion = nn.MSELoss()
d_optimizer = optim.Adim(discriminator.parameters(),lr=1e-2,betas=1e-1)
g_optimizer = optim.Adim(generator.parameters(),lr=1e-2,betas=1e-1)

img = io.imread('batman.jpg',True)
get_point = generate_lut(img)

for epoch in range(num_epochs):
	for d_step in range(d_steps):
		# Compute Discriminator loss and update
		# Sample from the True distribution
		d_inputs = get_true_sample(get_point,batch_size,d_input_dim)
		d_outputs = discriminator(d_inputs)
		dloss_true_sample = criterion(d_outputs,Variable(torch.ones(batch_size)))

		# Sample from the generated distribution
		g_inputs = get_generator_inputs(batch_size,g_input_dim)
		g_outputs = generator(g_inputs)
		d_fake_outputs = discriminator(g_outputs)
		dloss_fake_sample = criterion(d_fake_outputs,Variable(torch.zeros(batch_size)))

		# Compute total loss and update weights
		dloss = dloss_true_sample + dloss_fake_sample
		dloss.backward()
		d_optimizer.step()

	for g_step in range(g_steps):
		g_inputs = get_generator_inputs(batch_size,g_input_dim)
		g_outputs = generator(g_inputs)
		d_decisions = discriminator(g_outputs)
		gloss = criterion(d_decisions,Variable(torch.ones(batch_size)))

		gloss.backward()
		g_optimizer.step()


def get_true_sample(get_point,batch_size,d_input_dim = 2):
	return sampler(get_point,batch_size)

def get_generator_inputs(batch_size,g_input_dim):
	pass



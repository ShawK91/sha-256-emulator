import cPickle
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, torch
import torch.nn.functional as F
from scipy.special import expit
import fastrand, math


#Polynet bundle
class Polynet(nn.Module):
    def __init__(self, input_size, h1, h2, output_size, output_activation):
        super(Polynet, self).__init__()

        self.input_size = input_size; self.h1 = h1; self.h2 = h2; self.output_size = output_size
        if output_activation == 'sigmoid': self.output_activation = expit
        elif output_activation == 'tanh': self.output_activation = np.tanh
        else: self.output_activation = None

        #Weights
        self.w1 = np.mat(np.random.normal(0,1, (h1, input_size)))
        self.w_poly = np.mat(np.ones((h2, h1)))
        self.w2 = np.mat(np.random.normal(0, 1, (output_size, h2)))

        self.param_dict = {'w1': self.w1,
                           'w_poly': self.w_poly,
                           'w2': self.w2}
        self.gd_net = GD_polynet(input_size, h1, h2, output_size, output_activation)


    def forward(self, input):
        batch_size = input.shape[1]

        first_out = np.dot(self.w1, input) #Linear transform
        first_out = np.multiply(first_out, (first_out > 0.01)) #First dense layer with thresholding activation (Relu except 0.01 translated to 0.01)

        #Polynomial Operation
        poly_out = self.poly_op(first_out, batch_size) #Polynomial dot product
        output = np.dot(self.w2, poly_out) #Output dense layer
        if self.output_activation != None: output = self.output_activation(output)
        return output

    def poly_op(self, inp, batch_size):
        poly_out = np.mat(np.zeros((self.h2, batch_size)))
        for i, node in enumerate(self.w_poly):
            batch_poly = self.batch_copy(node, batch_size, axis=0)
            node_act = np.sum(np.power(inp, np.transpose(batch_poly)), axis=0)
            poly_out[i,:] = node_act

        return poly_out

    def batch_copy(self, mat, batch_size, axis):
        padded_mat = np.copy(mat)
        for _ in range(batch_size - 1): padded_mat = np.concatenate((padded_mat, mat), axis=axis)
        return padded_mat

    def from_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            params[key][:] = gd_params[key].cpu().numpy()

    def to_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            gd_params[key][:] = params[key]

    def reset(self, batch_size):
        return

class GD_polynet(nn.Module):
    def __init__(self, input_size, h1, h2, output_size, output_activation):
        super(GD_polynet, self).__init__()

        self.input_size = input_size; self.output_size = output_size
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None

        #Weights
        self.w1 = Parameter(torch.rand(h1, input_size), requires_grad=1)
        self.w_poly = Parameter(torch.rand(h2, h1), requires_grad=1)
        self.w2 = Parameter(torch.rand(output_size, h2), requires_grad=1)

        #Initialize weights except for poly weights which are initialized to all 1s
        for param in self.parameters():
            #torch.nn.init.xavier_normal(param)
            #torch.nn.init.orthogonal(param)
            #torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)
        #self.w_poly = Parameter(torch.ones(h2, h1), requires_grad=1) +
        self.w_poly.data += 1.0


    def forward(self, input):
        first_out = F.threshold(self.w1.mm(input), 0.01, 0.01) #First dense layer with thresholding activation (Relu except 0 translated to 0.1)

        #Polynomial operation
        poly1 = torch.t(first_out).pow(self.w_poly)
        poly_out = torch.sum(poly1, 1).unsqueeze(1)

        #Output dense layer
        output = self.w2.mm(poly_out)
        if self.output_activation != None: output = self.output_activation(output)
        return output

    #TODO Batch Process for GD_Polynet
    def reset(self, batch_size):
        return


#MMU Bundle
class MMU:
    def __init__(self, num_input, num_hnodes, num_memory, num_output, output_activation, mean = 0, std = 1):
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes; self.num_mem = num_memory
        self.output_activation = output_activation

        #Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std, (num_hnodes, num_input)))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std, (num_hnodes, num_output)))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std, (num_hnodes, num_memory)))

        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, (num_hnodes, num_input)))
        self.w_rec_inp = np.mat(np.random.normal(mean, std, (num_hnodes, num_output)))

        #Read gate
        self.w_readgate = np.mat(np.random.normal(mean, std, (num_memory, num_input)))
        self.w_rec_readgate = np.mat(np.random.normal(mean, std, (num_memory, num_output)))
        self.w_mem_readgate = np.mat(np.random.normal(mean, std, (num_memory, num_memory)))

        #Memory Decoder
        self.w_decoder = np.mat(np.random.normal(mean, std, (num_hnodes, num_memory)))

        #Memory write gate
        self.w_writegate = np.mat(np.random.normal(mean, std, (num_memory, num_input)))
        self.w_rec_writegate = np.mat(np.random.normal(mean, std, (num_memory, num_output)))
        self.w_mem_writegate = np.mat(np.random.normal(mean, std, (num_memory, num_memory)))

        # Memory Encoder
        self.w_encoder= np.mat(np.random.normal(mean, std, (num_memory, num_hnodes)))

        #Output weights
        self.w_hid_out= np.mat(np.random.normal(mean, std, (num_output, num_hnodes)))

        #Biases
        self.w_input_gate_bias = np.mat(np.zeros((num_hnodes, 1)))
        self.w_block_input_bias = np.mat(np.zeros((num_hnodes, 1)))
        self.w_readgate_bias = np.mat(np.zeros((num_memory, 1)))
        self.w_writegate_bias = np.mat(np.zeros((num_memory, 1)))

        #Adaptive components (plastic with network running)
        self.output = np.mat(np.zeros((num_output, 1)))
        self.memory = np.mat(np.zeros((num_memory, 1)))

        self.param_dict = {'w_inpgate': self.w_inpgate,
                           'w_rec_inpgate': self.w_rec_inpgate,
                           'w_mem_inpgate': self.w_mem_inpgate,
                           'w_inp': self.w_inp,
                           'w_rec_inp': self.w_rec_inp,
                            'w_readgate': self.w_readgate,
                            'w_rec_readgate': self.w_rec_readgate,
                            'w_mem_readgate': self.w_mem_readgate,
                           'w_decoder' : self.w_decoder,
                            'w_writegate': self.w_writegate,
                            'w_rec_writegate': self.w_rec_writegate,
                            'w_mem_writegate': self.w_mem_writegate,
                           'w_encoder': self.w_encoder,
                           'w_hid_out': self.w_hid_out,
                            'w_input_gate_bias': self.w_input_gate_bias,
                           'w_block_input_bias': self.w_block_input_bias,
                            'w_readgate_bias': self.w_readgate_bias,
                           'w_writegate_bias': self.w_writegate_bias}

        self.gd_net = GD_MMU(num_input, num_hnodes, num_memory, num_output, output_activation) #Gradient Descent Net

    def forward(self, input): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)
        #Input gate
        input_gate_out = expit(np.dot(self.w_inpgate, input)+ np.dot(self.w_rec_inpgate, self.output) + np.dot(self.w_mem_inpgate, self.memory) + self.w_input_gate_bias)

        #Input processing
        block_input_out = expit(np.dot(self.w_inp, input) + np.dot(self.w_rec_inp, self.output) + self.w_block_input_bias)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #Read Gate
        read_gate_out = expit(np.dot(self.w_readgate, input) + np.dot(self.w_rec_readgate, self.output) + np.dot(self.w_mem_readgate, self.memory) + self.w_readgate_bias)

        #Compute hidden activation - processing hidden output for this iteration of net run
        decoded_mem = np.dot(self.w_decoder, np.multiply(read_gate_out, self.memory))
        hidden_act =  decoded_mem + input_out

        #Write gate (memory cell)
        write_gate_out = expit(np.dot(self.w_writegate, input)+ np.dot(self.w_rec_writegate, self.output) + np.dot(self.w_mem_writegate, self.memory) + self.w_writegate_bias)

        #Write to memory Cell - Update memory
        encoded_update = np.dot(self.w_encoder, hidden_act)
        self.memory += np.multiply(write_gate_out, encoded_update)

        #Compute final output
        self.output = np.dot(self.w_hid_out, hidden_act)
        if self.output_activation == 'tanh': self.output = np.tanh(self.output)
        if self.output_activation == 'sigmoid': self.output = expit(self.output)
        return self.output

    def reset(self, batch_size):
        #Adaptive components (plastic with network running)
        self.output = np.mat(np.zeros((self.num_output, batch_size)))
        self.memory = np.mat(np.zeros((self.num_mem, batch_size)))

    def from_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            params[key][:] = gd_params[key].cpu().numpy()

    def to_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            gd_params[key][:] = params[key]

class GD_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, output_activation):
        super(GD_MMU, self).__init__()

        self.input_size = input_size; self.hidden_size = hidden_size; self.memory_size = memory_size; self.output_size = output_size
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None

        #Input gate
        self.w_inpgate = Parameter(torch.rand(hidden_size, input_size), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand( hidden_size, output_size), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        #Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(hidden_size, output_size), requires_grad=1)

        #Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Memory Decoder
        self.w_decoder = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        #Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Memory Encoder
        self.w_encoder = Parameter(torch.rand(memory_size, hidden_size), requires_grad=1)

        #Output weights
        self.w_hid_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        #Biases
        self.w_input_gate_bias = Parameter(torch.zeros(hidden_size, 1), requires_grad=1)
        self.w_block_input_bias = Parameter(torch.zeros(hidden_size, 1), requires_grad=1)
        self.w_readgate_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)
        self.w_writegate_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)

        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, 1), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            #torch.nn.init.xavier_normal(param)
            #torch.nn.init.orthogonal(param)
            #torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, batch_size), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    def graph_compute(self, input, rec_output, mem):
        #Block Input
        block_inp = F.sigmoid(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))# + self.w_block_input_bias)

        #Input gate
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(mem) + self.w_rec_inpgate.mm(rec_output))# + self.w_input_gate_bias)

        #Input out
        inp_out = block_inp * inp_gate

        #Read gate
        read_gate_out = F.sigmoid(self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(mem))# + self.w_readgate_bias) * mem

        #Compute hidden activation
        decoded_mem = self.w_decoder.mm(read_gate_out * mem)
        hidden_act = inp_out + decoded_mem

        #Write gate
        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(mem) + self.w_rec_writegate.mm(rec_output))# + self.w_writegate_bias)

        #Update memory
        encoded_update = F.tanh(self.w_encoder.mm(hidden_act))
        mem = mem + write_gate_out * encoded_update

        output = self.w_hid_out.mm(hidden_act)
        if self.output_activation != None: output = self.output_activation(output)

        return output, mem

    def bgraph_compute(self, input, rec_output, mem):
        #Block Input
        block_inp = F.sigmoid(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))# + self.w_block_input_bias)

        #Input gate
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(mem) + self.w_rec_inpgate.mm(rec_output))# + self.w_input_gate_bias)

        #Input out
        inp_out = block_inp * inp_gate

        #Read gate
        read_gate_out = F.sigmoid(self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(mem))# + self.w_readgate_bias) * mem

        #Compute hidden activation
        hidden_act = inp_out + read_gate_out * mem

        #Write gate
        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(mem) + self.w_rec_writegate.mm(rec_output))# + self.w_writegate_bias)

        #Update memory
        mem = mem + write_gate_out * F.tanh(hidden_act)

        output = self.w_hid_out.mm(hidden_act)
        if self.output_activation != None: output = self.output_activation(output)

        return output, mem


    def forward(self, input):
        self.out, self.mem = self.graph_compute(input, self.out, self.mem)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True


#FF Bundle
class FF:
    def __init__(self, num_input, num_hnodes, num_output, output_activation, mean=0, std=1):
        self.num_input = num_input;
        self.num_output = num_output;
        self.num_hnodes = num_hnodes;
        self.output_activation = output_activation

        # Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, (num_hnodes, num_input)))

        # Output weights
        self.w_hid_out = np.mat(np.random.normal(mean, std, (num_output, num_hnodes)))

        # Biases
        self.w_inp_bias = np.mat(np.zeros((num_hnodes, 1)))
        self.w_output_bias = np.mat(np.zeros((num_output, 1)))



        self.param_dict = {'w_inp': self.w_inp,
                           'w_hid_out': self.w_hid_out,
                           'w_inp_bias': self.w_inp_bias,
                           'w_output_bias': self.w_output_bias}

        self.gd_net = GD_FF(num_input, num_hnodes, num_output, output_activation)  # Gradient Descent Net

    def forward(self, input):  # Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        # Hidden activations
        hidden_act = expit(np.dot(self.w_inp, input) + self.w_inp_bias)

        # Compute final output
        self.output = np.dot(self.w_hid_out, hidden_act) + self.w_output_bias
        if self.output_activation == 'tanh': self.output = np.tanh(self.output)
        if self.output_activation == 'sigmoid': self.output = expit(self.output)
        return self.output

    def reset(self, batch_size):
        return

    def from_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            params[key][:] = gd_params[key].cpu().numpy()

    def to_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            gd_params[key][:] = params[key]

class bGD_FF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_activation):
        super(GD_FF, self).__init__()

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.output_size = output_size
        if output_activation == 'sigmoid':
            self.output_activation = F.sigmoid
        elif output_activation == 'tanh':
            self.output_activation = F.tanh
        else:
            self.output_activation = None


        # Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size), requires_grad=1)

        # Output weights
        self.w_hid_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)


        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def reset(self, batch_size):
        return

    def graph_compute(self, input):
        # Compute hidden activations
        hidden_act = F.sigmoid(self.w_inp.mm(input))  # + self.w_block_input_bias)

        #Compute Output
        output = self.w_hid_out.mm(hidden_act)
        if self.output_activation != None: output = self.output_activation(output)

        return output

    def forward(self, input):
        self.out = self.graph_compute(input)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True


#Neuroevolution SSNE
class SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters;
        self.population_size = self.parameters.pop_size;
        self.num_elitists = int(self.parameters.elite_fraction * parameters.pop_size)
        if self.num_elitists < 1: self.num_elitists = 1

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1, gene2):
        keys = list(gene1.param_dict.keys())

        # References to the variable tensors
        W1 = gene1.param_dict
        W2 = gene2.param_dict
        num_variables = len(W1)
        if num_variables != len(W2): print 'Warning: Genes for crossover might be incompatible'

        # Crossover opertation [Indexed by column, not rows]
        num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = fastrand.pcg32bounded(num_variables)  # Choose which tensor to perturb
            receiver_choice = random.random()  # Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = fastrand.pcg32bounded(W1[keys[tensor_choice]].shape[-1])  #
                W1[keys[tensor_choice]][:, ind_cr] = W2[keys[tensor_choice]][:, ind_cr]
                #W1[keys[tensor_choice]][ind_cr, :] = W2[keys[tensor_choice]][ind_cr, :]
            else:
                ind_cr = fastrand.pcg32bounded(W2[keys[tensor_choice]].shape[-1])  #
                W2[keys[tensor_choice]][:, ind_cr] = W1[keys[tensor_choice]][:, ind_cr]
                #W2[keys[tensor_choice]][ind_cr, :] = W1[keys[tensor_choice]][ind_cr, :]

    def mutate_inplace(self, gene):
        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        # References to the variable keys
        keys = list(gene.param_dict.keys())
        W = gene.param_dict
        num_structures = len(keys)
        ssne_probabilities = np.random.uniform(0,1,num_structures)*2

        for ssne_prob, key in zip(ssne_probabilities, keys): #For each structure apart from poly
            if random.random()<ssne_prob:
                num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                for _ in range(num_mutations):
                    ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                    ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                    random_num = random.random()

                    if random_num < super_mut_prob:  # Super Mutation probability
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[key][ind_dim1, ind_dim2])
                    elif random_num < reset_prob:  # Reset probability
                        W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)

                    else:  # mutauion even normal
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][ind_dim1, ind_dim2])

                    # Regularization hard limit
                    W[key][ind_dim1, ind_dim2] = self.regularize_weight(W[key][ind_dim1, ind_dim2], self.parameters.weight_magnitude_limit)

    def trial_mutate_inplace(self, hive):
        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_prob = 0.05

        for drone in hive.all_drones:
            # References to the variable keys
            keys = list(drone.param_dict.keys())
            W = drone.param_dict
            num_structures = len(keys)
            ssne_probabilities = np.random.uniform(0,1,num_structures)*2

            for ssne_prob, key in zip(ssne_probabilities, keys): #For each structure
                if random.random()<ssne_prob:

                    mut_matrix = scipy_rand(W[key].shape[0], W[key].shape[1], density=num_mutation_frac, data_rvs=np.random.randn).A * mut_strength
                    W[key] += np.multiply(mut_matrix, W[key])


                    # num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                    # for _ in range(num_mutations):
                    #     ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                    #     ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                    #     random_num = random.random()
                    #
                    #     if random_num < super_mut_prob:  # Super Mutation probability
                    #         W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                    #                                                                       W[key][
                    #                                                                           ind_dim1, ind_dim2])
                    #     elif random_num < reset_prob:  # Reset probability
                    #         W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)
                    #
                    #     else:  # mutauion even normal
                    #         W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][
                    #                                                                           ind_dim1, ind_dim2])
                    #
                    #     # Regularization hard limit
                    #     W[key][ind_dim1, ind_dim2] = self.regularize_weight(
                    #         W[key][ind_dim1, ind_dim2])

    def copy_individual(self, master, replacee):  # Replace the replacee individual with master
            keys = master.param_dict.keys()
            for key in keys:
                replacee.param_dict[key][:] = master.param_dict[key]

    def reset_genome(self, gene):
        keys = gene.param_dict
        for key in keys:
            dim = gene.param_dict[key].shape
            gene.param_dict[key][:] = np.mat(np.random.uniform(-1, 1, (dim[0], dim[1])))

    def epoch(self, all_hives, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        #Extinction step (Resets all the offsprings genes; preserves the elitists)
        if random.random() < self.parameters.extinction_prob: #An extinction event
            print
            print "######################Extinction Event Triggered#######################"
            print
            for i in offsprings:
                if random.random() < self.parameters.extinction_magnituide and not (i in elitist_index):  # Extinction probabilities
                    self.reset_genome(all_hives[i])

        # Figure out unselected candidates
        unselects = []; new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=all_hives[i], replacee=all_hives[replacee])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            self.copy_individual(master=all_hives[off_i], replacee=all_hives[i])
            self.copy_individual(master=all_hives[off_j], replacee=all_hives[j])
            self.crossover_inplace(all_hives[i], all_hives[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.parameters.crossover_prob: self.crossover_inplace(all_hives[i], all_hives[j])

        # Mutate all genes in the population except the new elitists plus homozenize
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.parameters.mutation_prob: self.mutate_inplace(all_hives[i])


#Functions
def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))



import numpy as np, os
import lib_shaw as mod
from random import randint
from torch.autograd import Variable
import torch
from torch.utils import data as util
import hashlib
from random import randint




class Tracker(): #Tracker
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        for update, var in zip(updates, self.all_tracker):
            var[0].append(update)

        #Constrain size of convolution
        if len(self.all_tracker[0][0]) > 100: #Assume all variable are updated uniformly
            for var in self.all_tracker:
                var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            var[1] = sum(var[0])/float(len(var[0]))

        if generation % 10 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')

class Parameters:
    def __init__(self):
        self.pop_size = 50
        self.load_checkpoint = False
        self.total_gens = 5000

        #NN specifics
        self.num_hnodes = 100
        self.num_mem = 100
        self.output_activation = None
        self.sample_size = 500


        #SSNE stuff
        self.elite_fraction = 0.04
        self.crossover_prob = 0.05
        self.mutation_prob = 0.9
        self.extinction_prob = 0.004 #Probability of extinction event
        self.extinction_magnituide = 0.5 #Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 100000
        self.mut_distribution = 0 #1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s

        # Train data
        self.batch_size = 1000
        self.train_size = 10000
        self.valid_size = 2000
        self.inp_len = 3
        self.hash_len = 16


        #Dependents
        self.num_output = 16
        self.num_input = 16
        self.save_foldername = 'R_Copy/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)


class Task_Copy:
    def __init__(self, parameters):
        self.params = parameters
        self.ssne = mod.SSNE(parameters)
        self.train_x, self.train_y = self.get_data(self.params.train_size) #Get training and test data
        self.test_x, self.test_y = self.get_data(self.params.valid_size)

        #Initialize population
        if self.params.load_checkpoint: self.pop = mod.unpickle(self.params.save_foldername + 'population')
        else:
            self.pop = []
            #for _ in range(parameters.pop_size): self.pop.append(mod.Polynet(self.params.num_input, self.params.h1, self.params.h2, self.params.num_output, self.params.output_activation))
            #for _ in range(parameters.pop_size): self.pop.append(mod.FF(self.params.num_input, self.params.h1, self.params.num_output, self.params.output_activation))
            for _ in range(parameters.pop_size): self.pop.append(mod.MMU(self.params.num_input, self.params.num_hnodes, self.params.num_mem, self.params.num_output, self.params.output_activation))

        #Run backprop
        self.run_bprop(self.pop[0].gd_net)
        self.pop[0].from_gdnet()

    def run_bprop(self, model):


        if True: #GD optimizer choices
            #criterion = torch.nn.L1Loss(False)
            criterion = torch.nn.SmoothL1Loss(False)
            #criterion = torch.nn.KLDivLoss()
            #criterion = torch.nn.CrossEntropyLoss()
            #criterion = torch.nn.MSELoss()
            # criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum = 0.5, nesterov = True)
            # optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.005, momentum=0.1)

        #Set up training
        all_train_x = torch.Tensor(self.train_x).cuda()
        all_train_y = torch.Tensor(self.train_y).cuda()
        train_dataset = util.TensorDataset(all_train_x, all_train_y)
        train_loader = util.DataLoader(train_dataset, batch_size=self.params.batch_size, shuffle=True)
        model.cuda()

        for epoch in range(1, 5000):

            epoch_loss = 0.0
            for data in train_loader:  # Each Batch
                net_inputs, targets = data

                recall_input = torch.Tensor(torch.zeros((16, self.params.batch_size)))
                recall_input[-1,:] = 1
                model.reset(self.params.batch_size)  # Reset memory and recurrent out for the model

                #Run sequence of input to copy
                for i in range(self.params.inp_len):  # For the length of the sequence
                    net_inp = Variable(torch.t(net_inputs[:,:,i]), requires_grad=True)
                    net_out = model.forward(net_inp)

                #Check if network is able to copy
                for j in range(self.params.hash_len):
                    net_inp = Variable(recall_input, requires_grad=True).cuda()
                    net_out = model.forward(net_inp)

                    target_T = Variable(torch.t(targets[:,:,j]))
                    #loss = criterion(torch.t(net_out), torch.t(target_T))
                    loss = criterion(net_out, target_T)
                    loss.backward(retain_variables=True)
                    epoch_loss += loss.cpu().data.numpy()[0]

                #print target_T.cpu().data.numpy()[0], net_out.cpu().data.numpy()[0], loss.cpu().data.numpy()[0]#, model.w1.grad.data.cpu().numpy()[0], model.w_poly.grad.data.cpu().numpy()[0]
                optimizer.step()  # Perform the gradient updates to weights for the entire set of collected gradients
                optimizer.zero_grad()


            if epoch % 10 == 0:
                print 'Epoch: ', epoch, ' Loss: ', epoch_loss,
                print self.evaluate(model, self.test_x, self.test_y)



    def str_one_hot(self, str):
        one_hot = np.zeros((16, len(str)))
        for i, ch in enumerate(str):
            one_hot[int(ch, 16),i] = 1

        return one_hot

    def get_data(self, volume):
        raw_x = []; raw_y = []
        for _ in range(volume):
            raw_x.append(str(randint(100, 999)))
            raw_y.append(hashlib.sha1(raw_x[-1]).hexdigest())

        #Convert both train_x and train_y to one hot
        train_x = []; train_y = []
        for x, y in zip(raw_x, raw_y):
            train_x.append(self.str_one_hot(x))
            train_y.append(self.str_one_hot(y))

        train_x = np.array(train_x); train_y = np.array(train_y)

        return train_x, train_y




    def shuffle_data(self):
        permutation = np.random.permutation(len(self.train_x[1]))
        self.train_x = self.train_x[:,permutation]
        self.train_y = self.train_y[:,permutation]

    def evaluate(self, net, test_x, test_y):
        mat_correct = []
        net.reset(self.params.valid_size)
        recall_input = torch.Tensor(torch.zeros((16, self.params.valid_size)))
        recall_input[-1, :] = 1
        test_x = torch.Tensor(test_x).cuda()
        test_y = torch.Tensor(test_y).cuda()

        # Run sequence of input to copy
        for i in range(self.params.inp_len):  # For the length of the sequence + 2
            net_inp = Variable(torch.t(test_x[:, :, i]), requires_grad=True)
            net_out = net.forward(net_inp)

        # Check if network is able to copy
        for j in range(self.params.hash_len):
            net_inp = Variable(recall_input, requires_grad=True).cuda()
            net_out = net.forward(net_inp)
            target = torch.t(test_y[:, :, j])

            y = Variable(target).cpu().data.numpy(); y_max = np.argmax(y, axis=0)
            out = net_out.cpu().data.numpy(); out_max = np.argmax(out, axis=0)
            is_correct = (y_max == out_max)
            mat_correct.append(is_correct)

            #error = torch.sum(torch.abs(Variable(target) - net_out)).cpu().data.numpy()

        mat_correct = np.array(mat_correct)
        strict_metric = mat_correct.all(axis=0)


        return '%.2f'%np.average(mat_correct), '%.2f'%np.average(strict_metric)


        # net_out = net.forward(train_x)
        # is_finite = np.isfinite(net_out).all(axis=0)
        # net_picks = (net_out>0.5)
        # is_correct = np.multiply((net_picks == train_y), is_finite)
        # fitness = np.sum(is_correct)/(1.0 * len(train_x[1]))

        #print net_picks[:,10], train_y[:,10], is_correct[:,10]
        #return fitness

    def evolve(self, gen, tracker):

        #Evaluation loop
        fitnesses = []

        #self.shuffle_data()
        #data_choices = np.random.choice(len(self.train_x[1]), self.params.sample_size, replace=False)
        #train_x = self.train_x[:, data_choices]; train_y = self.train_y[:,data_choices]
        train_x = self.train_x[:,:self.params.sample_size]; train_y = self.train_y[:,0:self.params.sample_size]
        #train_x = self.train_x; train_y = self.train_y
        for net in self.pop:
            fitness = self.evaluate(net, train_x, train_y)
            fitnesses.append(fitness)

        #Get champion index and compute validation score
        best_train_fitness = max(fitnesses)
        champion_index = fitnesses.index(best_train_fitness)

        #Run simulation for champion on validation
        valid_fitness = self.evaluate(self.pop[champion_index], self.test_x, self.test_y)

        #Save champion
        if gen % 40 == 0:
            mod.pickle_object(self.pop[champion_index], self.params.save_foldername + 'champion') #Save champion
            mod.pickle_object(self.pop, self.params.save_foldername + 'population')  # Save entire colony of hives (all population)
            mod.pickle_object(tracker, self.params.save_foldername + 'tracker') #Save the tracker file
            np.savetxt(self.params.save_foldername + 'gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')


        #SSNE Epoch
        self.ssne.epoch(self.pop, fitnesses)

        return best_train_fitness, valid_fitness



if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    if parameters.load_checkpoint:
        gen_start = int(np.loadtxt(parameters.save_foldername + 'gen_tag'))
        tracker = mod.unpickle(parameters.save_foldername + 'tracker')
    else:
        tracker = Tracker(parameters, ['best_train', 'valid'], '_cifar.csv')  # Initiate tracker
        gen_start = 1
    print 'Sha-1 Code break with', parameters.num_input, 'inputs,', parameters.num_hnodes, 'hnodes', parameters.num_mem, 'memory', parameters.num_output
    sim_task = Task_Copy(parameters)
    for gen in range(gen_start, parameters.total_gens):
        best_train_fitness, validation_fitness = sim_task.evolve(gen, tracker)
        print 'Gen:', gen, 'Ep_best:', '%.2f' %best_train_fitness, ' Valid_Fit:', '%.2f' %validation_fitness, 'Cumul_valid:', '%.2f'%tracker.all_tracker[1][1]
        tracker.update([best_train_fitness, validation_fitness], gen)
















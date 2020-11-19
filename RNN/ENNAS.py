import sys
from model import RNNModel
import genotypes
import data
from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

import time
import math
import numpy as np
import torch
import copy
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import gc

import shutil
import logging
import inspect
import pickle
import argparse
import genotypes

from numpy.linalg import cholesky

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DartsWrapper:
    def __init__(self, save_path, seed, batch_size, grad_clip, config='eval'):
        if config == 'search':
            args = {'emsize':300, 'nhid':300, 'nhidlast':300, 'dropoute':0, 'wdecay':5e-7}
        elif config == 'eval':
            args = {'emsize':850, 'nhid':850, 'nhidlast':850, 'dropoute':0.1, 'wdecay':8e-7}
        args['config'] = config

        args['data'] = '/home/mzhang3/Data/DARTS/darts-master/data/penn'
        args['lr'] = 20
        args['clip'] = grad_clip
        args['batch_size'] = batch_size
        args['search_batch_size'] = 256*4
        args['small_batch_size'] = batch_size
        args['bptt'] = 35
        args['dropout'] = 0.75
        args['dropouth'] = 0.25
        args['dropoutx'] = 0.75
        args['dropouti'] = 0.2
        args['seed'] = seed
        args['nonmono'] = 5
        args['log_interval'] = 50
        args['save'] = save_path
        args['alpha'] = 0
        args['beta'] = 1e-3
        args['max_seq_length_delta'] = 20
        args['unrolled'] = True
        args['gpu'] = 0
        args['cuda'] = True
        args = AttrDict(args)
        self.args = args
        self.seed = seed

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True
        torch.cuda.manual_seed_all(args.seed)

        corpus = data.Corpus(args.data)
        self.corpus = corpus

        eval_batch_size = 10
        test_batch_size = 1

        self.train_data = batchify(corpus.train, args.batch_size, args)
        self.search_data = batchify(corpus.valid, args.search_batch_size, args)
        self.val_data = batchify(corpus.valid, eval_batch_size, args)
        self.test_data = batchify(corpus.test, test_batch_size, args)
        self.batch = 0
        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()


        ntokens = len(corpus.dictionary)

        model = RNNModel(ntokens, args.emsize, args.nhid, args.nhidlast,
                   args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute, genotype=genotypes.DARTS)

        size = 0
        for p in model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))
        logging.info('initial genotype:')
        logging.info(model.rnns[0].genotype)

        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

        self.model = model.cuda()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        #self.parallel_model = model.cuda()

    def set_model_arch(self, arch):
        for rnn in self.model.rnns:
            rnn.genotype = arch

    def train_batch(self, arch):
        args = self.args
        model = self.model
        self.set_model_arch(arch)

        corpus = self.corpus
        optimizer = self.optimizer
        total_loss = self.total_loss

        # Turn on training mode which enables dropout.
        ntokens = len(corpus.dictionary)
        i = self.steps % (self.train_data.size(0) - 1 - 1)
        batch = self.batch

        if i == 0:
            hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
        else:
            hidden = self.hidden

        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        data, targets = get_batch(self.train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            # assuming small_batch_size = batch_size so we don't accumulate gradients
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
              loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()
        self.hidden = hidden

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        self.epochs = self.steps // (self.train_data.size(0) - 1 - 1)
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - self.start_time
            val_ppl = self.evaluate(arch)
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | val_ppl {:8.2f}'.format(
                self.epochs, batch % (len(self.train_data) // args.bptt), len(self.train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), val_ppl))
            total_loss = 0
            self.start_time = time.time()
        self.batch += 1
        self.steps += seq_len

    def evaluate(self, arch, n_batches=10):
        # Turn on evaluation mode which disables dropout.
        model = self.model
        #weights = self.get_weights_from_arch(arch)
        self.set_model_arch(arch)
        #self.set_model_weights(weights)
        model.eval()
        args = self.args
        total_loss = 0
        ntokens = len(self.corpus.dictionary)
        hidden = model.init_hidden(self.args.search_batch_size)
        #TODO: change this to take seed so that same minibatch can be used when desired
        #batches = np.random.choice(np.arange(self.search_data.size(0) -1), n_batches, replace=False)
        for i in range(0, self.search_data.size(0) - 1, args.bptt):
            data, targets = get_batch(self.search_data, i, args, evaluation=True)
            targets = targets.view(-1)

            log_prob, hidden = model(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

            total_loss += loss * len(data)

            hidden = repackage_hidden(hidden)
        try:
            ppl = total_loss / len(self.search_data)
        except Exception as e:
            ppl = 100000
        return ppl
        # TODO: add resume functionality

    def save(self):
        save_checkpoint(self.model, self.optimizer, self.epochs, self.args.save)

        
    def novelty_fitness(self,arch,store_arch,k):

        def dis_arch(arch1,arch2):
            dis=genotypes.STEPS
            n_nodes=genotypes.STEPS

            for i in range(n_nodes):
                if arch1[2*i,]==arch2[2*i,] and arch1[2*i+1,]==arch2[2*i+1,]:
                    dis=dis-1
            dis=dis/8
            return dis     
        store_arch=store_arch[:-100,]

        dis=np.zeros((store_arch.shape[0],))
        for i in range(store_arch.shape[0]):
            dis[i]=dis_arch(arch[0,],store_arch[i,])
        sort_dis=np.sort(dis)
        novelty_dis=np.mean(sort_dis[0:k])
        
        return novelty_dis               
        
    def sample_arch(self,node_id,store_arch):
        n_nodes = genotypes.STEPS
        n_ops = len(genotypes.PRIMITIVES)
                
        def limite_range(arch,n_ops,n_nodes):
            arch=np.int_(arch)
            for i in range(n_nodes):               
                arch[2*i,]=np.max((np.min((arch[2*i,],n_ops-1)),1))###############################why not conyain none
                arch[2*i+1,]=np.max((np.min((arch[2*i+1,],(i))),0))

            return arch    
            
        def get_performance(self,selec_arch):
 
            n_nodes = genotypes.STEPS
            n_ops = len(genotypes.PRIMITIVES)       
            arch = []
            performance=np.zeros((1))
           # selec_arch=np.zeros((2*n_nodes,))
            for i in range(n_nodes):
                op = np.int(selec_arch[2*i,])
                node_in = np.int(selec_arch[2*i+1,])
                arch.append((genotypes.PRIMITIVES[op], node_in))
            concat = range(1,9)
            genotype = genotypes.Genotype(recurrent=arch, concat=concat)
            performance[0,]=self.evaluate(genotype)   
            return performance[0,]

                 
        if node_id>999:
            alfa=0.01
            n=2
            sigma=1
            mu=np.zeros((1,2*n_nodes))
            Sigma=np.eye(2*n_nodes)
            R=cholesky(Sigma)       
            yita=np.dot(np.random.randn(n,2*n_nodes),R)+mu
            n_yita=np.empty((n,2*n_nodes))
            n_yita1=np.empty((n,2*n_nodes))#############################store the performance###whether take the reward into consideration
                        
            index0=np.random.randint(1000)
            test_arch=store_arch[index0,]
         
            for i in range(n):
                test_i=test_arch+yita[i,]
                n_f=self.novelty_fitness(np.int_(np.round((test_i))),np.int_(np.round(store_arch)),10)
                n_yita[i,]=n_f*yita[i,]
               # select_i=limite_range(test_i,n_ops,n_nodes)###whether take the reward into consideration
               # n_yita1[i,]=get_performance(self,select_i)###whether take the reward into consideration
                
                
            selec_arch=test_arch+alfa*(1/(n*sigma))*sum(n_yita)
           # selec_arch=test_arch+alfa*(1/(n*sigma))*(0.5*sum(n_yita)+0.5*sum(n_yita1))######whether take the reward into consideration
            store_arch[index0,]=selec_arch             
            selec_arch=np.int_(np.round(selec_arch))            
            selec_arch=limite_range(selec_arch,n_ops,n_nodes)
            
            arch = []
            for i in range(n_nodes):               
                op=np.int(selec_arch[2*i,])
                node_in=np.int(selec_arch[2*i+1,])
                arch.append((genotypes.PRIMITIVES[np.int(op)], node_in))
            index=index0
           
                                                   
        else:     
            n_nodes = genotypes.STEPS
            n_ops = len(genotypes.PRIMITIVES)
            arch = []
            selec_arch=np.zeros((2*n_nodes,))
            for i in range(n_nodes):
                op = np.random.choice(range(1,n_ops))
                node_in = np.random.choice(range(i+1))
                arch.append((genotypes.PRIMITIVES[op], node_in))
                selec_arch[2*i,]=op
                selec_arch[2*i+1,]=node_in
            index=node_id
           
        concat = range(1,9)
        genotype = genotypes.Genotype(recurrent=arch, concat=concat)

######the operations from two previous node are different
        return index,selec_arch,genotype


        
    def sample_arch_eval(self):
        n_nodes = genotypes.STEPS
        n_ops = len(genotypes.PRIMITIVES)
        arch = []
        for i in range(n_nodes):
            op = np.random.choice(range(1,n_ops))
            node_in = np.random.choice(range(i+1))
            arch.append((genotypes.PRIMITIVES[op], node_in))
        concat = range(1,9)
        genotype = genotypes.Genotype(recurrent=arch, concat=concat)
        return genotype 
    
    

    def perturb_arch(self, arch):
        new_arch = copy.deepcopy(arch)
        p = np.arange(1,genotypes.STEPS+1)
        p = p / sum(p)
        c_ind = np.random.choice(genotypes.STEPS, p=p)
        #c_ind = np.random.choice(genotypes.STEPS)
        new_op = np.random.choice(range(1,len(genotypes.PRIMITIVES)))
        new_in = np.random.choice(range(c_ind+1))
        new_arch.recurrent[c_ind] = (genotypes.PRIMITIVES[new_op], new_in)
        #print(arch)
        #arch.recurrent[c_ind] = (arch.recurrent[c_ind][0],new_in)
        return new_arch


    def get_weights_from_arch(self, arch):
        n_nodes = genotypes.STEPS
        n_ops = len(genotypes.PRIMITIVES)
        weights = torch.zeros(sum([i+1 for i in range(n_nodes)]), n_ops)

        offset = 0
        for i in range(n_nodes):
            op = arch[i][0]
            node_in = arch[i][1]
            ind = offset + node_in
            weights[ind, op] = 5
            offset += (i+1)

        weights = torch.autograd.Variable(weights.cuda(), requires_grad=False)

        return weights





class Rung:
    def __init__(self, rung, nodes):
        self.parents = set()
        self.children = set()
        self.rung = rung
        for node in nodes:
            n = nodes[node]
            if n.rung == self.rung:
                self.parents.add(n.parent)
                self.children.add(n.node_id)

class Node:
    def __init__(self, parent, arch, node_id, rung):
        self.parent = parent
        self.arch = arch
        self.node_id = node_id
        self.rung = rung
    def to_dict(self):
        out = {'parent':self.parent, 'arch': self.arch, 'node_id': self.node_id, 'rung': self.rung}
        if hasattr(self, 'objective_val'):
            out['objective_val'] = self.objective_val
        return out

class Random_NAS:
    def __init__(self, B, model, seed, save_dir):
        self.save_dir = save_dir

        self.B = B
        self.model = model
        self.seed = seed

        self.iters = 0

        self.arms = {}
        self.node_id = 0
        
        size_arch = 2*genotypes.STEPS
               
        self.store_arch=np.empty((100,size_arch))
        

    def print_summary(self):
        logging.info(self.parents)
        objective_vals = [(n,self.arms[n].objective_val) for n in self.arms if hasattr(self.arms[n],'objective_val')]
        objective_vals = sorted(objective_vals,key=lambda x:x[1])
        best_arm = self.arms[objective_vals[0][0]]
        val_ppl = self.model.evaluate(best_arm.arch, split='valid')
        logging.info(objective_vals)
        logging.info('best valid ppl: %.2f' % val_ppl)


    def get_arch(self):
        inde,arr_arch,arch = self.model.sample_arch(self.node_id,self.store_arch)
        #arch = self.model.sample_arch_eval()
        
        self.store_arch[inde,]=arr_arch
                
        self.arms[self.node_id] = Node(self.node_id, arch, self.node_id, 0)
        self.node_id += 1
        return arch

    def save(self):
        to_save = {a: self.arms[a].to_dict() for a in self.arms}
        # Only replace file if save successful so don't lose results of last pickle save
        with open(os.path.join(self.save_dir,'results_tmp.pkl'),'wb') as f:
            pickle.dump(to_save, f)
        shutil.copyfile(os.path.join(self.save_dir, 'results_tmp.pkl'), os.path.join(self.save_dir, 'results.pkl'))

        self.model.save()

    def run(self):
        while self.iters < self.B:
            arch = self.get_arch()
            self.model.train_batch(arch)
            self.iters += 1
            if self.iters % 500 == 0:
                self.save()
        self.save()

    def get_eval_arch(self, rounds=None):
        #n_rounds = int(self.B / 7 / 1000)
        if rounds is None:
            n_rounds = max(1,int(self.B/10000))
        else:
            n_rounds = rounds
        best_rounds = []
        for r in range(n_rounds):
            sample_vals = []
            for _ in range(1000):
                arch = self.model.sample_arch_eval()
                ppl = self.model.evaluate(arch)
                if ppl>1000000:
                    ppl=1000000

                com_ppl=ppl        
                logging.info(arch)
                logging.info('objective_val: %.3f' % ppl)
                sample_vals.append((arch, com_ppl))
            sample_vals = sorted(sample_vals, key=lambda x:x[1])

            full_vals = []
            if 'split' in inspect.getargspec(self.model.evaluate).args:
                for i in range(10):
                    arch = sample_vals[i][0]
                    try:
                        ppl = self.model.evaluate(arch, split='valid')
                        com_ppl=ppl
                    except Exception as e:
                        ppl = 1000000                        
                    full_vals.append((arch, ppl))
                full_vals = sorted(full_vals, key=lambda x:x[1])
                logging.info('best arch: %s, best arch valid performance: %.3f' % (' '.join([str(i) for i in full_vals[0][0]]), full_vals[0][1]))
                best_rounds.append(full_vals[0])
            else:
                best_rounds.append(sample_vals[0])
        return best_rounds  
  
    
    def EA_arch_search(self,num_pop,num_ite,num_cross,num_mutation):

        def get_init_pop(self,num_pop,n_nodes):
            pop=np.empty((num_pop,2*n_nodes))
            fitness=np.zeros((num_pop,))
            for m in range(num_pop):                         
                num_ops = len(genotypes.PRIMITIVES)
                arch = []
                for i in range(n_nodes):
                    op = np.random.choice(range(1,num_ops))
                    node_in = np.random.choice(range(i+1))                   
                    arch.append((genotypes.PRIMITIVES[op], node_in))
                    pop[m,2*i]=op
                    pop[m,2*i+1]=node_in
                concat = range(1,9)                               
                genotype = genotypes.Genotype(recurrent=arch, concat=concat)
                fitness[m,]=self.model.evaluate(genotype)          
            return pop,fitness


        def corssover(self,pop,fitness,num_cross):
            index=np.argsort(fitness)
            pop_select=pop[index[0:num_cross],]


            inde_cross=np.arange(num_cross)
            np.random.shuffle(inde_cross)
            pop_select=pop_select[inde_cross,]
            pop_cross=np.empty((num_cross,pop.shape[1]))

            for i in range(np.int(num_cross/2)):
                cross1=pop_select[2*i,]
                cross2=pop_select[2*i+1,]

                cross_points=np.arange(2*genotypes.STEPS)
                np.random.shuffle(cross_points)
                cross_points=cross_points[0:2]
                cross_points=np.sort(cross_points)
                p1=2*cross_points[0]
                p2=2*cross_points[1]

                cross1_=cross1
                cross2_=cross2

                cross1_[p1:p2]=cross2[p1:p2]
                cross2_[p1:p2]=cross1[p1:p2]

                pop_cross[2*i,]= cross1_       
                pop_cross[2*i+1,]= cross2_   

            return pop_cross


        def mutation(self,pop,fitness,num_mutation):
            index=np.argsort(fitness)
            pop_select=pop[index[0:num_mutation],]
            pop_mutation=np.empty((num_mutation,pop.shape[1]))
            num_ops = len(genotypes.PRIMITIVES)


            for i in range(num_mutation):
                pop_mutation[i,]=pop_select[i,]

                for j in range(pop.shape[1]):
                    if np.random.rand()<0.2:#################genes with mutation probability 0.2np.random.choice(range(1,n_ops))
                        if j%2==0:
                            pop_mutation[i,j]=np.random.choice(range(1,num_ops))
                        else:
                            pop_mutation[i,j]=np.random.choice(range(np.int(j/2)+1))    
            return pop_mutation


        def get_fitness(self,pop,n_nodes):
            num_pop=pop.shape[0]
            fitness=np.zeros((num_pop))
            for m in range(num_pop):
                arch = []
                for i in range(n_nodes):
                    op = np.int(pop[m,2*i])
                    node_in = np.int(pop[m,2*i+1])
                    arch.append((genotypes.PRIMITIVES[np.int(op)], node_in))
                concat = range(1,9)                               
                genotype = genotypes.Genotype(recurrent=arch, concat=concat)
                fitness[m,]=self.model.evaluate(genotype)       
            return fitness

        k = sum(1 for i in range(genotypes.STEPS) for n in range(2+i))
        num_ops = len(genotypes.PRIMITIVES)
        n_nodes = genotypes.STEPS

        pop,fitness=get_init_pop(self,num_pop,n_nodes)

        for it in range(num_ite):
            pop_cross=corssover(self,pop,fitness,num_cross)
            
            fitness_cross=get_fitness(self,pop_cross,n_nodes)
            pop_mutate=mutation(self,pop,fitness,num_mutation)
            fitness_mutate=get_fitness(self,pop_mutate,n_nodes) 
            pop_comb=np.concatenate((pop,pop_cross,pop_mutate),axis=0)
            fitness_comb=np.concatenate((fitness,fitness_cross,fitness_mutate),axis=0)
            index=np.argsort(fitness_comb)
            pop_comb=pop_comb[index,]
            pop=pop_comb[0:num_pop,]
            fitness=fitness_comb[0:num_pop,]

        index=np.argsort(fitness)
        indi_final=pop[index[0],]
        sele_arch=[]
        
        for i in range(genotypes.STEPS):
            op = indi_final[2*i]
            node_in = indi_final[2*i+1]         
            sele_arch.append((genotypes.PRIMITIVES[np.int(op)], np.int(node_in)))
        concat = range(1,9)                               
        genotype = genotypes.Genotype(recurrent=sele_arch, concat=concat) 

        return genotype
   
    
parser = argparse.ArgumentParser(description='Args for SHA with weight sharing')
parser.add_argument('--benchmark', dest='benchmark', type=str, default='ptb')
parser.add_argument('--seed', dest='seed', type=int, default=300)
parser.add_argument('--epochs', dest='epochs', type=int, default=300)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
parser.add_argument('--grad_clip', dest='grad_clip', type=float, default=0.25)
parser.add_argument('--save_dir', dest='save_dir', type=str, default=None)
parser.add_argument('--eval_only', dest='eval_only', type=int, default=0)
# PTB only argument. config=search uses proxy network for shared weights while
# config=eval uses proxyless network for shared weights.
parser.add_argument('--config', dest='config', type=str, default="search")
# CIFAR-10 only argument.  Use either 16 or 24 for the settings for random search
# with weight-sharing used in our experiments.
parser.add_argument('--init_channels', dest='init_channels', type=int, default=16)
args = parser.parse_args()

# Fill in with root output path
root_dir = '/home/mzhang3/Data/randomNAS_RNN_OWN/results'
if args.save_dir is None:
    save_dir = os.path.join(root_dir, '%s/random/trial%d' % (args.benchmark, args.seed))
else:
    save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.eval_only:
    assert args.save_dir is not None

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info(args)

if args.benchmark=='ptb':
    data_size = 929589
    time_steps = 35
else:
    data_size = 25000
    time_steps = 1
B = int(args.epochs * data_size / args.batch_size / time_steps)
if args.benchmark=='ptb':
    #from benchmarks.ptb.darts.darts_wrapper_discrete import DartsWrapper
    model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, config=args.config)
elif args.benchmark=='cnn':
    #from benchmarks.cnn.darts.darts_wrapper_discrete import DartsWrapper
    model = DartsWrapper(save_dir, args.seed, args.batch_size, args.grad_clip, args.epochs, init_channels=args.init_channels)

searcher = Random_NAS(B, model, args.seed, save_dir)
logging.info('budget: %d' % (searcher.B))
if not args.eval_only:
    searcher.run()
    #archs1 = searcher.get_eval_arch()#####using random search for architecture selection
    archs1 = searcher.EA_arch_search(num_pop=100,num_ite=60,num_cross=60,num_mutation=40)##
    archs2 = searcher.EA_arch_search(num_pop=100,num_ite=60,num_cross=60,num_mutation=40)#####using EA for architecture selection
else:
    np.random.seed(args.seed+1)
    archs = searcher.get_eval_arch(2)
logging.info(archs1)
logging.info(archs2)

arch1 = ' '.join([str(a) for a in archs1[0][0]])
arch2 = ' '.join([str(a) for a in archs2[0][0]])


print(arch1)
print(arch2)

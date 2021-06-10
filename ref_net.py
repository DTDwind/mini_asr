#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, sys, random
import time, os, itertools, shutil, importlib
from eend.pytorch_backend.KaldiDataLoader import test_dataset_loader

from torch.cuda.amp import autocast, GradScaler

class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, ilen=None, label=None, n_speakers=None):
        return self.module(x, ilen, label, n_speakers)


class DairizationNet(nn.Module):
    def __init__(self, model, optimizer, loss_fn, gpu, num_speakers=None, use_attractor=False, use_triplet=False, attractor_loss_ratio=1, **kwargs): # Dynamic import model
        super(DairizationNet, self).__init__();

        if num_speakers is not None:
            self.num_speakers = num_speakers
        else:
            self.num_speakers = None

        self.gpu = gpu
        self.use_attractor = use_attractor
        self.process_counter = 0

        DairizationNetModel = importlib.import_module('eend.pytorch_backend.models.'+model).__getattribute__('MainModel')
        self.__S__          = DairizationNetModel(**kwargs);

        LossFunction        = importlib.import_module('eend.pytorch_backend.loss.'+loss_fn).__getattribute__('LossFunction')
        self.__L__          = LossFunction(**kwargs);

        self.use_triplet = use_triplet
        
        if use_triplet:
            triplet = 'triplet'
            triplet_LossFunction        = importlib.import_module('eend.pytorch_backend.loss.'+triplet).__getattribute__('LossFunction')
            self.__TL__          = triplet_LossFunction(**kwargs);

        if use_attractor:
            print('use EDA.')
            self.eda_lin = nn.Linear(256*2, 256)

            attractor_model = 'attractor'
            attractor_loss  = 'attractor_loss'

            self.attractor_loss_ratio = attractor_loss_ratio

            AttractorsModel     = importlib.import_module('eend.pytorch_backend.models.'+attractor_model).__getattribute__('AttractorModel')
            self.__atractors__  = AttractorsModel(**kwargs);

            AttractorsLossFunction     = importlib.import_module('eend.pytorch_backend.loss.'+attractor_loss).__getattribute__('LossFunction')
            self.__AL__  = AttractorsLossFunction(**kwargs);

        else:
            self.lin = nn.Linear(256*2, num_speakers)

        self.sigmoid = nn.Sigmoid()
        

    def forward(self, data, label=None, ilen=None, n_speakers=None, max_n_speaker=15):
        emb, ilen    = self.__S__.forward(data, ilen) # [B,T,D]
        # print('network emb: '+str(emb))
        # quit(0)
        if self.use_triplet:
            if label != None:
                Mloss = self.__TL__.forward(emb, label, ilen, n_speakers) 

        if self.use_attractor:
            emb = self.eda_lin(emb)

            # TODO
            # if shuffle:
            #     xp = cuda.get_array_module(emb[0])
            #     orders = [xp.arange(e.shape[0]) for e in emb]
            #     for order in orders:
            #         xp.random.shuffle(order)
            #     attractors, probs = self.eda.estimate([e[order] for e, order in zip(emb, orders)])

            if label == None: # eval
                zeros = torch.zeros(emb.size()[0], max_n_speaker, emb.size()[2]).cuda() # [B, max_S, D]
            else:
                zeros = torch.zeros(label.size()[0], label.size()[2]+1, emb.size()[2]).cuda() # [B, S+1, D]
            
            
            # train:[B, S+1, D]  inference: atractors[B, max_S, D] attractors_prob[B, max_S, 1]
            # TODO: shuffle
                # for order in orders:
                # xp.random.shuffle(order)
            shuffle = False
            if shuffle:
                orders = [numpy.arange(frame_num) for frame_num in ilen]
                # for order in orders:
                #     numpy.random.shuffle(order)
                # [e[order] for e, order in zip(emb, orders)]
                # atractors, attractors_prob = self.__atractors__.forward(emb, zeros, ilen) 
            else:
                atractors, attractors_prob = self.__atractors__.forward(emb, zeros, ilen) 

        
        if label == None: # eval
            if self.num_speakers is not None and self.use_attractor:

                attractors_prob = attractors_prob.squeeze()

                toptop_tensor, indices = torch.topk(attractors_prob, self.num_speakers)
                atractors = atractors[: , indices, :]

                atractors = atractors.permute(0, 2, 1)

                pred = torch.matmul(emb, atractors)

                logit_pred = self.sigmoid(pred)
            
                return logit_pred

            elif self.num_speakers is not None: # No eda
                
                # torch.set_printoptions(edgeitems=600)
                # print("QAQ emb: "+str(emb))
                pred       = self.lin(emb)
                logit_pred = self.sigmoid(pred)

                return logit_pred

            else: # dont know spk num
                threshold = 0.5
                attractors_prob = attractors_prob.squeeze()
                num_speakers = len(attractors_prob[attractors_prob > threshold])
                toptop_tensor, indices = torch.topk(attractors_prob, num_speakers)
                atractors = atractors[: , indices, :]
                atractors = atractors.permute(0, 2, 1)
                pred = torch.matmul(emb, atractors)
                logit_pred = self.sigmoid(pred)

                return logit_pred

        if self.use_attractor:

            # The final attractor does not correspond to a speaker so remove it
            atractors = atractors[:,:-1,:] 
            atractors = atractors.permute(0, 2, 1) # Transpose # [B, S, D]
        
            pred = torch.matmul(emb, atractors) # [B, T, S]

        else:
            pred = self.lin(emb)
        
        if self.use_attractor:
            dloss, _ = self.__L__.forward(pred, label, ilen, n_speakers) # label:[B, T, S]
            aloss = self.__AL__.forward(attractors_prob, ilen, n_speakers)
            if self.use_triplet:
                loss = 0.5*dloss + 0.5*aloss*self.attractor_loss_ratio + 0.3*Mloss
                # loss = dloss + aloss*self.attractor_loss_ratio + 0.1 *Mloss
            else:
                # loss = 0.5*dloss + 0.5*aloss*self.attractor_loss_ratio
                loss = dloss + aloss*self.attractor_loss_ratio
        
        else:
            dloss, _ = self.__L__.forward(pred, label, ilen, n_speakers) # label:[B, T, S]
            loss = dloss
            aloss = dloss
        
        return loss, dloss, aloss

class ModelTrainer(object):

    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__  = speaker_model

        Optimizer = importlib.import_module('eend.pytorch_backend.optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module('eend.pytorch_backend.scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler() 

        self.gpu = gpu

        self.mixedprec = mixedprec

        assert self.lr_step in ['epoch', 'iteration']

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose):

        self.__model__.train();

        stepsize = loader.batch_size;
        counter = 0;
        index   = 0;
        loss    = 0;
        d_loss = 0;
        a_loss = 0;
        top1    = 0     # EER or accuracy

        tstart = time.time()
        
        for xs, ys, ilen, n_speakers, reco_batch in loader:
            print('reco_batch: '+str(reco_batch))
            self.__model__.zero_grad();

            if self.mixedprec:
                with autocast():
                    print('mixedprec')
                    nloss, prec1 = self.__model__(xs, ys, ilen, n_speakers)
                self.scaler.scale(nloss).backward();
                self.scaler.step(self.__optimizer__);
                self.scaler.update();       

            else:
                nloss, dloss, aloss = self.__model__(xs, ys, ilen, n_speakers)
                nloss.backward();
                self.__optimizer__.step();

            loss    += nloss.detach().cpu();
            d_loss  += dloss.detach().cpu();
            a_loss  += aloss.detach().cpu();
            counter += 1;
            index   += stepsize;
        
            telapsed = time.time() - tstart
            tstart = time.time()
            
            if verbose:
                sys.stdout.write("\rProcessing (%d) "%(index));
                sys.stdout.write('gpu:%s, dloss: %f, aloss: %f'%( str(self.gpu), (d_loss/counter), (a_loss/counter) ))
                sys.stdout.write("\rDairizationNet xs: %s , gpu: %s "%( str(xs.size()), str(self.gpu) ) );
                sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
                sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n");
        
        # return (loss/counter, top1/counter);
        return (loss/counter, 0);


    ## Evaluate
    def evaluate(self, test_data_dir, subsampling, **kwargs):
        
        self.__model__.eval();
        
        tstart      = time.time()

        ## Define test data loader
        test_dataset = test_dataset_loader(test_data_dir, subsampling)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        diarization_result = {}

        print('evaluate...')
        for idx, data in enumerate(test_loader):
            feat = data[0].cuda()
            ilen = data[1]
            reco = data[2][0]
            # print('Q ilen: '+str(ilen))
            # print('Q reco: '+str(reco))
            pred = self.__model__(feat, None, ilen, [2])
            telapsed = time.time() - tstart

            diarization_result[reco] = pred.detach().cpu().numpy()
            print("Reading %d of %d: %.2f Hz"%(idx, test_loader.__len__(), idx/telapsed ));
            print('')
        

        return (diarization_result);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.__model__.module.state_dict(), path);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict();
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu);

        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);


import sys, time, os, socket
import numpy
import torch
import glob
import zipfile
import datetime
import random
# from tuneThreshold import *
import torch.distributed as dist
import torch.multiprocessing as mp

from eend.pytorch_backend.KaldiDataLoader import get_data_loader
from eend.pytorch_backend.DairizationNet import *

# ## ===== ===== ===== ===== ===== ===== ===== ===== test
# from eend.pytorch_backend.loss.pit import *
# ## ===== ===== ===== ===== ===== ===== ===== =====

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Trainer script
# ## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    seed = args.seed

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

    ## Load models
    s = DairizationNet(**vars(args));

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port
        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)
        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)
        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU %d'%args.gpu)

    else:
        s = WrappedModel(s).cuda(args.gpu)

    it = 1
    
    ## Write args to scorefile   
    scorefile   = open(args.result_save_path+"/scores.txt", "a+");

    ## Initialise trainer and data loader
    trainLoader, sampler = get_data_loader(args.train_data_dir, args);
    trainer     = ModelTrainer(s, **vars(args))
    
    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print("Model %s loaded from previous state!"%modelfiles[-1]);
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print("Model %s loaded!"%args.initial_model);

    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Core training script
    for it in range(it,args.epoch+1):
        
        if args.distributed:
            pass

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]
        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training epoch %d on GPU %d with LR %f "%(it,args.gpu,max(clr)));
        sampler.set_epoch(it)
        loss, traineer = trainer.train_network(trainLoader, verbose=(args.verbose));

        if it % args.test_interval == 0 and args.gpu == 0:
            # TODO: dev set eval and early stop
            trainer.saveParameters(args.model_save_path+"/model%09d.model"%it);

            print(time.strftime("%Y-%m-%d %H:%M:%S"), "TrainEER/TAcc %2.2f, TrainLoss %f"%( traineer, loss));
            scorefile.write("IT %d, TrainEER/TAcc %2.2f, TrainLoss %f\n"%(it, traineer, loss));
            scorefile.flush()

    scorefile.close();

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====
def train(args):
    args.model_save_path     = args.exp_dir+"/model"
    args.result_save_path    = args.exp_dir+"/result"

    if not(os.path.exists(args.model_save_path)):
        os.makedirs(args.model_save_path)
            
    if not(os.path.exists(args.result_save_path)):
        os.makedirs(args.result_save_path)

    n_gpus = torch.cuda.device_count()

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs Available:', torch.cuda.device_count())

    # TODO: save train config for inference
    # # save the config
    # numpy.save(args.model_save_path+'/args.npy', vars(args))

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)

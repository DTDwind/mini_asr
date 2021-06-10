#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#

import yamlargparse

def main():
    parser = yamlargparse.ArgumentParser(description='ASR training')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    # environment options
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('train_data_dir',
                        help='kaldi-style data dir used for training.')
    parser.add_argument('dev_data_dir',
                        help='kaldi-style data dir used for validation.')
    parser.add_argument('exp_dir',
                        help='output directory which model file will be saved in.')
    parser.add_argument('--backend', default='pytorch',
                        choices=['chainer', 'pytorch'],
                        help='backend framework')
    parser.add_argument('--model_save_dir', default="",
                        help='which will set by system')
    parser.add_argument('--result_save-_path', default="",
                        help='which will set by system')
    parser.add_argument('--verbose', default=0, type=int,
                        help='verbose message')

    # training options
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--max-epochs', default=20, type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--max-frames', default=2000, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--label-delay', default=0, type=int,
                        help='number of frames delayed from original labels'
                            ' for uni-directional rnn to see in the future')
    parser.add_argument('--batchsize', default=1, type=int,
                        help='number of utterances in one batch')
    parser.add_argument('--epoch',      type=int,   default=500,    help='Maximum number of epochs');
    parser.add_argument('--loss_fn',      type=str,   default='pit',    help='Loss function');
    parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs');
    parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');

    # model setting
    parser.add_argument('--model', default='blstm',
                        help='Type of model (Transformer or BLSTM)')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--hidden-size', default=128, type=int,
                        help='number of lstm output nodes')
    parser.add_argument('--emb-lstm-layer', default=1, type=int,
                        help='number of lstm layers')
    parser.add_argument('--dc-loss-ratio', default=0.5, type=float)
    parser.add_argument('--embedding-layers', default=2, type=int)
    parser.add_argument('--embedding-size', default=256, type=int)
    parser.add_argument('--noam-scale', default=1.0, type=float)
    parser.add_argument('--noam-warmup-steps', default=25000, type=float)
    parser.add_argument('--transformer-encoder-n-heads', default=4, type=int)
    parser.add_argument('--transformer-encoder-n-layers', default=2, type=int)
    parser.add_argument('--transformer-encoder-dropout', default=0.1, type=float)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    
    # inferene options
    parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
    parser.add_argument('--num-speakers', default=None, type=int)
    parser.add_argument('--spk-threshold', default=2, type=int)
    
    
    # feature options
    parser.add_argument('--subsampling', default=1, type=int)
    parser.add_argument('--feat_dim', default=83, type=int)
    parser.add_argument('--frame-size', default=1024, type=int)
    parser.add_argument('--frame-shift', default=256, type=int)
    parser.add_argument('--sampling-rate', default=16000, type=int)
    parser.add_argument('--feat_type', default='',
                        choices=['', 'log', 'logmel', 'logmel23', 'logmel23_mn',
                                'logmel23_mvn', 'logmel23_swn'],
                        help='input transform')

    ## Optimizer
    parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
    parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
    parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
    parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
    parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

    ## Parallel distributed
    parser.add_argument('--distributed', default=1, type=int)
    parser.add_argument('--port', default='1314', type=str)
    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

    # Triplet
    parser.add_argument('--use-triplet', default=False, help='triplet')
    parser.add_argument('--margin', default=0, help='triplet margin')
                                


    # attractor
    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument('--use-attractor', action='store_true',
                                help='Enable encoder-decoder attractor mode')
    
    attractor_args.add_argument('--shuffle', action='store_true',
                                help='Shuffle the order in time-axis before input to the network')
    attractor_args.add_argument('--attractor-loss-ratio', default=1.0, type=float,
                                help='weighting parameter')
    attractor_args.add_argument('--attractor-encoder-dropout', default=0.1, type=float)
    attractor_args.add_argument('--attractor-decoder-dropout', default=0.1, type=float)
    

    args = parser.parse_args()

    # system_info.print_system_info()
    print(args)
    
    if args.backend == 'pytorch':
        from mini_asr.bin.trainASRnet import train
        train(args)
    else:
        raise ValueError()

if __name__ == '__main__':
    main()
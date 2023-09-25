import os
import math
import time
import datetime
from functools import reduce
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
#import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.dim = args.dim
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.dir = args.dir
        # if args.dir:
        #     self.dir = args.dir
        # elif args.load == '.':
        #     if args.save == '.': args.save = now
        #     self.dir = '../experiment/' + args.save
        # else:
        #     self.dir = '../experiment/' + args.load
        #     if not os.path.exists(self.dir):
        #         args.load = '.'
        #     else:
        #         self.log = torch.load(self.dir + '/psnr_log.pt')
        #         print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            # args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        # _make_dir(self.dir + '/model')
        # _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
            import sys; f.write('Commandline\n' + ' '.join(map(lambda a: a.replace('=','="',1)+'"' if ('=' in a and any(c in a for c in r' {}')) else a, sys.argv)) + '\n')

    def save(self, trainer, epoch, is_best=False, loss_log=None):
        trainer.model.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        # trainer.loss.plot_loss(self.dir, epoch)

        #if self.dim == 2:
        #    self.plot_psnr(epoch)
        #    torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            {'optimizer_state': trainer.optimizer.state_dict(),\
             'scheduler_state': trainer.scheduler.state_dict(),\
             'epoch': epoch},
            os.path.join(self.dir, 'opt.pt')
        )
        if loss_log is not None:
            torch.save(loss_log, os.path.join(self.dir, 'loss_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, idx, save_list):
        tag = ['gt', 'pd']
        for t, v in zip(tag, save_list):
            np.save('%s/%s%04d.npy'%(self.dir, t, idx), to_channellast(v, self.dim))

def to_channellast(arr, dim):
    ndim=len(arr.shape)
    return np.transpose(arr, list(range(ndim-dim-1))+list(range(ndim-dim,ndim))+[ndim-dim-1])

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    '''
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(torch.nn.functional.relu(mse)+1.0e-12)

def make_optimizer(args, model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    opt = args.optimizer.lower()

    if opt == 'sgd':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif opt == 'adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif opt == 'adamw':
        optimizer_function = optim.AdamW
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif opt == 'rmsprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.wd
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    sch = args.scheduler.lower()
    if sch == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay_step,
            gamma=args.lr_decay_factor
        )
    elif sch == 'plateau':
        scheduler = lrs.ReduceLROnPlateau(my_optimizer,
            mode='min', patience=args.lr_decay_patience, min_lr=args.lr_min,
            factor=args.lr_decay_factor,verbose=True)
    elif sch == 'onecycle':
        scheduler = lrs.OneCycleLR(my_optimizer, max_lr=args.lr, 
          final_div_factor=int(args.lr/args.lr_min), total_steps=args.lr_decay_step)
    elif sch == 'cosine':
        scheduler = lrs.CosineAnnealingLR(my_optimizer, args.lr_decay_step,
          eta_min=args.lr_min, verbose=True)
    elif sch.find('step') >= 0:
        raise 'ERROR not implemented'
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.lr_decay_factor
        )
    else:
        raise ValueError(f'Unknown LR scheduler {sch}')
    return scheduler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import subprocess
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def make_trainer(args, loader, model, loss, checkpoint):
    try:
        module = import_module('NPS.'+args.trainer)
    except:
        module = import_module(args.trainer)
    return module.make_trainer(args, loader, model, loss, checkpoint)

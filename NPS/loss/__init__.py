import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class WtMSE(nn.modules.loss._Loss):
    def forward(self, input, target):
        return torch.mean((input - target)**2 * (target**2))

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            try:
                weight, loss_type = loss.split('*')
            except:
                weight, loss_type = 1, loss
            loss_type = loss_type.strip()
            if loss_type in ('MSE', 'L2'):
                loss_function = nn.MSELoss()
            elif loss_type == 'wt_MSE':
                loss_function = WtMSE()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.startswith('L1_wt'):
                wt = float(loss_type.replace('L1_wt',''))
                class l1_asym(nn.Module):
                    def forward(self, x, y):
                        return torch.mean(torch.abs(x-y)*(1+wt*y))
                loss_function = l1_asym()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            elif loss_type == 'BCE':
                loss_function = nn.BCELoss()
            elif loss_type[:6] == 'noise_':
                # continue
                if not args.noise:
                    continue
                if loss_type[6:] == 'MSE':
                    loss_function = nn.MSELoss()
                elif loss_type[6:] == 'L1':
                    loss_function = nn.L1Loss()
                # module = import_module('loss.noise')
                # loss_function = getattr(module, 'noise_loss')(
                #     args, loss_type[6:],
                # )
            else:
                raise ValueError('ERROR unknown loss ' + loss_type)
           
            self.loss.append({
                'type': loss_type, 'noise':loss_type[:6] == 'noise_',
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        # if args.load != '.':
        #    try:
        #        self.load(ckp.dir, cpu=args.cpu)
        #    except:
        #        print('Loss not loaded')

    # def forward(self, g_full, gen, ims, ome, sca):
    def forward(self, *args):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['noise']:
                    loss = l['function'](ome, sca)
                else:
                    loss = l['function'](*args)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                # self.log[-1, i] += self.loss[i - 1]['function'].loss
                pass

        loss_sum = sum(losses)
        # if len(self.loss) > 1:
        #     self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        # print(f'debug start log {self.log} len loss {len(self.loss)} {torch.zeros(1, len(self.loss))}')
        if len(self.log) > 0:
            if len(self.loss) > self.log.shape[1]: #added new loss
                print('WARNING: new loss since previous save?')
                self.log = torch.nn.functional.pad(self.log, (0, len(self.loss)-self.log.shape[1]))
            elif len(self.loss) < self.log.shape[1]:
                print('WARNING: removed loss since previous save?')
                self.log = self.loss[:,:len(self.loss)]
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.3g} ]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        # axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            # plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.plot(self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        # torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        # self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        # for l in self.get_loss_module():
        #     if hasattr(l, 'scheduler'):
        #         for _ in range(len(self.log)): l.scheduler.step()


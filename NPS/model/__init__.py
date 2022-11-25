import os
from importlib import import_module

import torch
import torch.nn as nn
# from torch.autograd import Variable
from NPS_common.ema import ExponentialMovingAverage

# def make_model(args, ckpt, file=''):
#     if not file:
#         file = args.model#.lower()
#     try:
#         module = import_module('NPS.model.' + file)
#     except:
#         module = import_module(file)
#     model = module.make_model(args)
#     return model

class NPSModel(nn.Module):
    def __init__(self, configs):
        super(NPSModel, self).__init__()
        self.dim= configs.dim
        self.periodic = configs.periodic
        self.in_feats = configs.n_colors
        self.out_feats = configs.n_colors_out
        self.total_length = configs.total_length
        self.input_length = configs.input_length
        self.frame_shape = configs.frame_shape
        self.ngram = configs.ngram
        self.register_buffer('conserved', torch.Tensor(configs.conserved).float()[None,:]) # persistent=False # add this argument when upgrading pytorch

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.args = args
        self.dim= args.dim
        # self.self_ensemble = args.self_ensemble
        # self.chop = args.chop
        # self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        # module = import_module('NPS.model.' + args.model.lower())
        try:
            module = import_module('NPS.model.' + args.model)
        except:
            module = import_module(args.model)
        self.model = module.make_model(args).to(self.device)
        # if args.precision == 'half': self.model.half()
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=args.ema_decay)

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        if args.freeze:
            if args.freeze.startswith('stoch'):
                for name, param in self.get_model().deter.noise_func.named_parameters():
                    param.requires_grad = False
            elif args.freeze.startswith('deter'):
                for name, param in self.get_model().deter.named_parameters():
                    if 'noise_func' not in name:
                        param.requires_grad = False
            else:
                print(f'WARNING: Unknown freeze option {args.freeze}')
            for name, param in self.get_model().named_parameters():
                print('after freezing:', name, param.shape, param.requires_grad)


    def forward(self, *x, **kwx):
        # target = self.get_model()
        # if self.self_ensemble and not self.training:
        #     if self.chop:
        #         forward_function = self.forward_chop
        #     else:
        #         forward_function = self.model.forward

        #     return self.forward_x8(x, forward_function)
        # elif self.chop and not self.training:
        #     return self.forward_chop(x)
        # else:
        #     return self.model(*x, **kwx)
        return self.model(*x, **kwx)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model_latest.pt')
        )
        torch.save(
            self.ema.state_dict(),
            os.path.join(apath, 'model_ema.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='', resume='latest', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        fname = os.path.join(apath, 'model_ema.pt')
        if os.path.exists(fname):
            self.ema.load_state_dict(torch.load(fname, **kwargs))
            print('Loaded EMA from ', fname)
        else:
                print(f'No ema found {fname}')
        fname = os.path.join(apath, f'model_{resume}.pt')
        if resume in ('latest', 'best', 'ema'):
            if os.path.exists(fname):
                self.get_model().load_state_dict(
                  torch.load(fname, **kwargs) if resume!='ema' else dict(zip([n for n,_ in self.get_model().named_parameters()], torch.load(fname, **kwargs)['shadow_params'])),
                    strict=False)
                print('Loaded from ', fname)
            else:
                print(f'No saved model found {fname}')
        elif resume == 'pre':
            print(f'Loading model from pretrained {pre_train}')
            self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
            )
        else:
            print('Loading from numbered ', fname)
            self.get_model().load_state_dict(
                torch.load(fname, **kwargs),
                strict=False
            )


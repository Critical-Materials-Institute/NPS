#!/bin/env python
import torch
import os, sys; sys.path.append(os.path.join(sys.path[0], '..'))

from NPS import utility
# from NPS import data
from NPS.data import Data
from NPS import model
from NPS import loss
from NPS.option import args

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
if not checkpoint.ok:
    exit()
loader = Data(args)
model = model.Model(args, checkpoint)
if args.print_model:
    print(model)
    try:
        from torchinfo import summary
        summary(model)
    except:
        from NPS.utility import count_parameters
        print(f'Total params: {count_parameters(model)} parameters')
loss = loss.Loss(args, checkpoint) if not args.predict_only else None
trainer = utility.make_trainer(args, loader, model, loss, checkpoint)

# job
if args.mode == 'train':
    trainer.train()
    # trainer.test()
elif args.mode == 'predict':
    # model.eval()
    trainer.predict()
elif args.mode == 'valid':
    # model.eval()
    trainer.validate()
elif args.mode == 'trace':
    from importlib import import_module
    m = import_module(args.export_wrapper)
    wrapper = m.export_wrapper(model, args, data=loader)
    wrapper.export(args.export_file)
    print('*'*100, f'\n written model to {args.export_file}\n','*'*100)
else:
    raise ValueError(f'Unknown job mode {args.mode}')

checkpoint.done()


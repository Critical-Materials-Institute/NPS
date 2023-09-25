import os, sys, time
import math
# from decimal import Decimal

from . import utility

import torch
# from torch.autograd import Variable
#from tqdm import tqdm
import numpy as np
from NPS_common.utils import a1line
from .data.data_augment import data_augment_operator

def make_trainer(args, loader, model, loss, checkpoint):
    return Trainer(args, loader, model, loss, checkpoint)

def get_frame(x, sl):
    if isinstance(x, (list, tuple)):
        return x[sl]
    else:
        return x[:, sl]

class Trainer():
    def __init__(self, args, loader, model, loss, ckp):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.args = args
        self.dim = args.dim

        self.ckp = ckp
        self.loader_train = loader.train
        self.loader_test = loader.test
        self.model = model
        self.loss = loss
        self.augment_op = data_augment_operator(args)
        self.optimizer = utility.make_optimizer(args, self.model) if args.mode == 'train' else None
        self.scheduler = utility.make_scheduler(args, self.optimizer) if args.mode == 'train' else None
        # if self.scheduler is not None: self.scheduler.last_epoch = len(self.loss.log)+1
        # self.epoch_start = len(self.loss.log)+1 if self.loss is not None else 0
        self.epoch_start = 0
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        # self.nsample_train = args.batch*args.n_GPUs
        self.loss_valid_log, self.loss_train_log = [], []
        self.loss_valid_min = np.inf
        self.ich = 1 if args.channel_first else -1

        misc_file =  os.path.join(ckp.dir, 'opt.pt')
        if self.optimizer is not None and os.path.exists(misc_file): #self.args.load != '.':
            try:
                print(f'Loading optimizer from {misc_file}')
                ckpt = torch.load(misc_file)
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
                # print(f'epoch st {self.epoch_start} log {self.loss.log}')
                self.scheduler.load_state_dict(ckpt['scheduler_state'])
                self.epoch_start = ckpt['epoch']
                print(f'Restored optimizer, scheduler, epoch')
                self.loss_train_log, self.loss_valid_log = torch.load(os.path.join(ckp.dir, 'loss_log.pt'))
                print(f'Restored loss log (train/valid)')
                if len(self.loss_valid_log): self.loss_valid_min = np.min(self.loss_valid_log)
            except:
                print(f'Failed to load optimizer from {misc_file}')
            # if args.scheduler != 'plateau':
            #     for _ in range(len(ckp.log)): self.scheduler.step()

        self.ckp.write_log('Commandline ' + ' '.join(sys.argv))
        self.preprocess_data(loader.loader_func)
        self.train_generator = self.get_training_batch()
        if args.epoch_size == -1: args.epoch_size = len(self.loader_train)
        # if args.n_traj_out == -1: args.n_traj_out = len(self.loader_test) * args.batch 

    def preprocess_data(self, loader_func):
        # loader_train/test is just dataset, not loader yet
        args = self.args
        if args.model_preprocess_data:
            print('Preprocessing data with model before dataloader')
            proc = self.model.get_model().preprocess_dataset
            if self.loader_train is not None:
                self.loader_train = loader_func(proc(self.loader_train, args), batch_size=args.batch, shuffle=True)
            self.loader_test = loader_func(proc(self.loader_test, args), batch_size=args.batch_test, shuffle=False, drop_last=False)


    def get_training_batch(self):
        while True:
            for batch in self.loader_train:
                yield(self.augment_op(batch))
                # if self.pointgroup.nops > 1:
                #     batch = self.pointgroup.random_op()(batch)
                # yield batch

    def to_device(self, x):
        if isinstance(x, (list, tuple)):
            return [ix.to(self.device) for ix in x]
        else:
            return x.to(self.device)

    def train(self):
        # while not self.terminate():
        #     self.train_epoch()
        #     self.test()
        for epoch in range(self.epoch_start+1, self.args.nepoch+1):
            t0 = time.time()
            loss_epoch = 0
            loss_epoch_item = []
            self.model.train()
            for i in range(1, self.args.epoch_size+1):
                x = self.to_device(next(self.train_generator))
                # print(f'debug x in train {x.shape} epo {epoch} {i} opt.epoch_size {opt.epoch_size} opt.niter {opt.niter}')
                loss = self.train_batch(x, epoch=epoch)
                # single loss, or (loss, (loss_items))
                if isinstance(loss, tuple):
                    if len(loss[1]): loss_epoch_item.append(loss[1])
                    loss = loss[0]
                # print(f'debug got loss {loss}')
                loss_epoch += loss
                if i % self.args.print_freq == 0:
                    print(f'  {i} loss_batch {loss:7.3e} {a1line(loss_epoch_item[-1]) if loss_epoch_item else ""} averaged {loss_epoch/i:7.3e}')
            loss_epoch /= self.args.epoch_size
            if loss_epoch_item: loss_epoch_item = np.mean(np.array(loss_epoch_item), 0)
            tmp = time.time(); t_train = tmp-t0; t0 = tmp
            with torch.no_grad():
                self.model.eval()
                loss_valid = self.evaluate(False, epoch=epoch)
                print(f'{epoch} {time.strftime("%H:%M:%S", time.localtime())} Train_loss: {loss_epoch:7.3e} {a1line(loss_epoch_item)} Valid_loss,mse,mae: {a1line(loss_valid)} train/test_time {t_train:6.2e} {time.time()-t0:6.2e}')
            self.loss_train_log.append(loss_epoch)
            self.loss_valid_log.append(loss_valid[0])
            is_best = False
            if self.loss_valid_min > loss_valid[0]:
                self.loss_valid_min = loss_valid[0]
                is_best = True
            if self.args.scheduler == 'plateau':
                self.scheduler.step(loss_valid[0])
            else:
                self.scheduler.step()

            # save the model
            if True:
                self.ckp.save(self, epoch, is_best=is_best, loss_log=(self.loss_train_log, self.loss_valid_log))
            # if epoch % 1 == 0:
                # torch.save({
                #     'model': model,
                #     'optimizer': optimizer,
                #     'opt': opt},
                #     '%/model.pt' % (opt.log_dir))

    def evaluate(self, predict_only, epoch=0):
        t0 = time.time()
        args = self.args
        n_in  = args.n_in_test
        n_out = args.n_out_test
        mse_detail = []
        mae_detail = []
        losses = []
        loss_item = None
        self.model.eval()
        n_pd = 0
        pd_all = []
        gt_in = []
        gt_all = []
        sp = 3 if args.channel_first else 2
        with torch.no_grad():
            for i, x in enumerate(self.loader_test):
                if isinstance(x, (tuple, list)): cell = tuple(x[0].cell_shape[:args.dim].cpu().numpy()) if args.dim>0 else tuple()
                if predict_only and (n_pd >= args.n_traj_out):
                    break
                # pd = self.evaluate_batch(x.to(self.device)).detach().cpu()
                # print("before eval", x, [ix.x.shape for ix in get_frame(x,slice(n_in,n_in+n_out))])
                pd, loss_item = self.evaluate_batch(self.to_device(x), predict_only)
                # print("after eval",  x, [ix.x.shape for ix in get_frame(x,slice(n_in,n_in+n_out))])
                pd = pd.detach().cpu()
                # if isinstance(x, (tuple, list)):
                #     pd = pd.reshape((len(x[0].ptr)-1,)+cell + (n_out,-1,)).
                n_pd += len(pd)
                pd_all.append(pd)
                gt_in.append(torch.stack([ix.x.cpu().reshape((len(x[0].ptr)-1,)+cell + (-1,)) for ix in get_frame(x,slice(None,n_in))],1) if isinstance(x, (tuple, list)) else get_frame(x,slice(None,n_in)))
                if not predict_only:
                    gt = torch.stack([ix.x.cpu().reshape((len(x[0].ptr)-1,)+cell + (-1,)) for ix in get_frame(x,slice(n_in,n_in+n_out))],1) if isinstance(x, (tuple, list)) else get_frame(x,slice(n_in,n_in+n_out))
                    gt_all.append(gt)
                    # print("pd.shape, gt.shape, gt_in.shape", pd.shape, gt.shape, gt_in[-1].shape, x, [ix.x.shape for ix in get_frame(x,slice(n_in,n_in+n_out))])
                    mse_detail.append(torch.mean((pd-gt)**2 , axis=tuple(range(sp,sp+args.dim))) if args.dim>0 else (pd-gt)**2)
                    mae_detail.append(torch.mean(torch.abs(pd-gt) , axis=tuple(range(sp,sp+args.dim))) if args.dim>0 else (pd-gt).abs())
                    losses.append([self.loss(pd, gt)] if (loss_item is None) or (loss_item==[]) else loss_item)
            if losses: losses = tuple(np.mean(losses, 0))
            # validation loss from model directly
            if (not predict_only) and args.loss_from_model:
                losses_model = []
                losses_model_item = []
                self.model.train() # so that model.training is True for loss computation mode. TBD: pass is_train explicitly
                for i, x in enumerate(self.loader_test):
                    loss = self.train_batch(self.to_device(x), epoch=epoch, is_train=False)
                    # single loss, or (loss, (loss_items))
                    if isinstance(loss, tuple):
                        if len(loss[1]): losses_model_item.append(loss[1])
                        loss = loss[0]
                    losses_model.append(loss)
                losses = (np.mean(losses_model, 0),) + losses

        # print(f'debug pd {pd_all[-1].shape} {len(pd_all)}')
        pd_all = torch.cat(pd_all)
        gt_in = torch.cat(gt_in)
        pd_out = torch.cat((gt_in, pd_all), 1) if args.gt_in_out else pd_all
        if args.n_traj_out > 0:
            pd_out = pd_out[:args.n_traj_out]
        np.save(f'{args.dir}/{args.file_out}pd.npy', utility.to_channellast(pd_out, self.dim) if self.args.channel_first else pd_out)
        if predict_only:
            print(f'Predicted data of size {pd_all.shape} time {time.time()-t0:7.3e}')
            return
        else:
            gt_all = torch.cat(gt_all)
            gt_out = torch.cat((gt_in, gt_all), 1) if args.gt_in_out else gt_all
            if args.n_traj_out > 0:
                gt_out = gt_out[:args.n_traj_out]
            np.save(f'{args.dir}/{args.file_out}gt.npy', utility.to_channellast(gt_out, self.dim) if self.args.channel_first else gt_out)
            mse_detail = np.concatenate(mse_detail, 0)
            mae_detail = np.concatenate(mae_detail, 0)
            mse, mae = np.mean(mse_detail), np.mean(mae_detail)
            print(f'valid per time/channel/seq: mse {a1line(np.mean(mse_detail,axis=(0,2)))} / {a1line(np.mean(mse_detail,axis=(0,1)))} / {a1line(np.mean(mse_detail,axis=(1,2)))}')
            print(f'valid per time/channel/seq: mae {a1line(np.mean(mae_detail,axis=(0,2)))} / {a1line(np.mean(mae_detail,axis=(0,1)))} / {a1line(np.mean(mae_detail,axis=(1,2)))}')
            try:
                self.model.get_model().analyze(pd_all.reshape(-1, pd_all.shape[-1]), gt_all.reshape(-1, gt_all.shape[-1]))
            except:
                pass
            if args.mode == 'valid':
                print(f'{epoch} {time.strftime("%H:%M:%S", time.localtime())} Valid_loss,mse,mae: {a1line(tuple(losses) + (mse, mae))}')
            return tuple(losses) + (mse, mae)

    def validate(self):
        return self.evaluate(False)

    def predict(self):
        return self.evaluate(True)

    def model_y_loss(self, x_in, target, loss_from_model=False, reset=True):
        """The model may either directly return a prediction, or prediction, loss and itemized losses"""
        y = self.model(x_in, reset=reset, target=target if loss_from_model else None, criterion=self.loss, mask=None)
        if target is None:
            return y, 0, []
        if isinstance(y, tuple):
            y, loss_step_item = y
            loss_step = self.step_loss(loss_step_item)
            loss_step_item = [x.item() for x in loss_step_item if len(loss_step_item)>1]
        else:
            loss_step = self.loss(y, target)
            loss_step_item = []
        return y, loss_step, loss_step_item

    def step_loss(self, loss_step_item):
        if self.args.loss_wt:
            return sum([loss_step_item[i]*self.args.loss_wt[i] for i in range(len(loss_step_item))])
        else:
            return sum(loss_step_item)

    def training_callback(self, *x, **kwx):
        self.optimizer.step()
        self.model.ema.update()

    def train_batch(self, x, epoch=1, is_train=True):
        args = self.args
        n_in  = args.n_in
        n_out = args.n_out
        RNN = args.RNN
        loss = 0
        loss_item = []
        if is_train: self.optimizer.zero_grad()
        for ei in range(n_in-1):
            target = get_frame(x, ei+1)
            tgt = target if isinstance(x, (tuple, list)) else (target[:, :self.args.nfeat_out] if args.channel_first else target[..., :self.args.nfeat_out])
            y, step_loss, step_loss_item = self.model_y_loss(get_frame(x, ei), tgt, reset=(ei==0), loss_from_model=args.loss_from_model)
            if step_loss_item: loss_item.append(step_loss_item)
            loss += step_loss
        x_in = get_frame(x,n_in-1)
        for di in range(n_out):
            target = get_frame(x,n_in+di)
            tgt = target if isinstance(x, (tuple, list)) else (target[:, :self.args.nfeat_out] if args.channel_first else target[..., :self.args.nfeat_out])
            y, step_loss, step_loss_item = self.model_y_loss(x_in, tgt, reset=(not RNN) or ((n_in==1) and (di==0)), loss_from_model=args.loss_from_model)
            if step_loss_item: loss_item.append(step_loss_item)
            loss += step_loss
            if (di < n_out-1) and args.nfeat_in>args.nfeat_out and (not isinstance(x, (tuple, list))):
                y = torch.cat([y, target[:, args.nfeat_out:] if args.channel_first else target[..., args.nfeat_out:]], self.ich)
            if di < n_out-1: x_in = self.get_scheduled_input(y, target, epoch)
        if is_train: loss.backward()
        if is_train: self.training_callback()
        return loss.item() / n_out, np.mean(loss_item, 0) if loss_item else []

    def evaluate_batch(self, x, predict_only=False):
        args = self.args
        n_in  = args.n_in_test
        n_out = args.n_out_test
        RNN = self.args.RNN
        traj = []
        loss = 0
        loss_item = []
        for ei in range(n_in-1):
            target = None if predict_only else get_frame(x, ei+1)
            tgt = None if predict_only else (target if isinstance(x, (tuple, list)) else (target[:, :self.args.nfeat_out] if args.channel_first else target[..., :self.args.nfeat_out]))
            y, step_loss, step_loss_item = self.model_y_loss(get_frame(x,ei), tgt, reset=(ei==0), loss_from_model=args.loss_from_model)
            if not predict_only:
                if step_loss_item: loss_item.append(step_loss_item)
                loss += step_loss
        x_in = get_frame(x,n_in-1)
        if isinstance(x, (tuple, list)): cell = tuple(x[0].cell_shape[:args.dim].cpu().numpy()) if args.dim>0 else tuple()
        for di in range(n_out):
            target = None if predict_only else get_frame(x,n_in+di)
            tgt = None if predict_only else (target if isinstance(x, (tuple, list)) else (target[:, :self.args.nfeat_out] if args.channel_first else target[..., :self.args.nfeat_out]))
            y, step_loss, step_loss_item = self.model_y_loss(x_in, tgt, reset=(not RNN) or ((n_in==1) and (di==0)), loss_from_model=args.loss_from_model)
            if not predict_only:
                if step_loss_item: loss_item.append(step_loss_item)
                loss += step_loss
            if args.nfeat_in>args.nfeat_out and (not predict_only) and (not isinstance(x, (tuple, list))):
                y = torch.cat([y, target[:, args.nfeat_out:] if args.channel_first else target[..., args.nfeat_out:]], self.ich)
            traj.append((y.x.reshape((len(x[0].ptr)-1,)+cell+(-1,)) if isinstance(x, (tuple, list)) else y).detach())
            x_in = y
        return torch.stack(traj, 1), np.mean(loss_item, 0) if loss_item else []

    def get_scheduled_input(self, y, target, epoch):
        flag = self.get_real_input_flag(epoch)
        if flag == 1:
            x_in = target
        elif flag == 0:
            x_in = y
        else:
            "TBD"
        return x_in

    def get_real_input_flag(self, itr=1):
        args = self.args
        if args.rnn_scheduled_sampling == 'reverse':
            real_input_flag = reserve_schedule_sampling_exp(itr, args)
        elif args.rnn_scheduled_sampling == 'decrease':
            real_input_flag = schedule_sampling(itr, args)
        elif args.rnn_scheduled_sampling == 'GT':
            real_input_flag = 1
        elif args.rnn_scheduled_sampling == 'PD':
            real_input_flag = 0
        else:
            raise ValueError(f'unknown rnn_scheduled_sampling {args.rnn_scheduled_sampling}')
        return real_input_flag



def reserve_schedule_sampling_exp(itr, args):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (args.batch, args.n_in - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (args.batch, args.n_out - 1))
    true_token = (random_flip < eta)

    ones = np.ones(  (*[x // args.patch_size for x in args.frame_shape],
                    args.patch_size ** args.dim * args.nfeat_in))
    zeros = np.zeros((*[x // args.patch_size for x in args.frame_shape],
                    args.patch_size ** args.dim * args.nfeat_in))

    real_input_flag = []
    for i in range(args.batch):
        for j in range(args.n_in + args.n_out - 2):
            if j < args.n_in - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.n_in - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch,
                                  args.n_in + args.n_out - 2,
                                  *[x // args.patch_size for x in args.frame_shape],
                                  args.patch_size ** 2 * args.nfeat_in))
    return real_input_flag


def schedule_sampling(itr, args):
    zeros = np.zeros((args.batch,
                      args.n_out - 1,
                      *[x // args.patch_size for x in args.frame_shape],
                      args.patch_size ** 2 * args.nfeat_in))
    if not args.scheduled_sampling:
        return 0.0, zeros

    eta = np.max(0, args.sampling_start_value - itr*args.sampling_changing_rate)
    random_flip = np.random.random_sample(
        (args.batch, args.n_out - 1))
    true_token = (random_flip < eta)
    ones = np.ones((*[x // args.patch_size for x in args.frame_shape],
                    args.patch_size ** args.dim * args.nfeat_in))
    zeros = np.zeros((*[x // args.patch_size for x in args.frame_shape],
                      args.patch_size ** args.dim * args.nfeat_in))
    real_input_flag = []
    for i in range(args.batch):
        for j in range(args.n_out - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch,
                                  args.n_out - 1,
                                  *[x // args.patch_size for x in args.frame_shape],
                                  args.patch_size ** 2 * args.nfeat_in))
    return real_input_flag


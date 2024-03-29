import os, sys, time
from . import utility

import torch
# from torch.autograd import Variable
#from tqdm import tqdm
import numpy as np
from NPS_common.utils import a1line
import pickle

def make_trainer(args, loader, model, loss, checkpoint):
    return TrainerNonSequential(args, loader, model, loss, checkpoint)


from NPS.trainer import Trainer
class TrainerNonSequential(Trainer):
    def evaluate(self, predict_only, epoch=0):
        t0 = time.time()
        args = self.args
        mse_detail = []
        mae_detail = []
        losses = []
        self.model.eval()
        n_pd = 0
        pd_all = []
        gt_all = []
        symbols_all = []
        with torch.no_grad():
            for i, x in enumerate(self.loader_test):
                if not predict_only:
                    gt = x[self.args.node_y].cpu()
                pd = self.evaluate_batch(x.to(self.device)).detach().cpu()
                n_pd += len(pd)
                pd_all.append(pd)
                if args.data_suffix == 'xyz':
                    symbols_all.extend(x.numbers)
                if not predict_only:
                    gt_all.append(gt)
                    mse_detail.append(torch.mean((pd-gt)**2, dim=(0,1), keepdim=True))
                    mae_detail.append(torch.mean(torch.abs(pd-gt), dim=(0,1), keepdim=True))
                    losses.append(self.loss(pd, gt))

        pd_all = torch.cat(pd_all)
        pd_all_size = pd_all.shape
        if args.n_traj_out > 0:
            pd_all = pd_all[:args.n_traj_out]
        np.save(f'{args.dir}/{args.file_out}pd.npy', pd_all)
        print(f'Predicted data of size {pd_all_size} time {time.time()-t0:7.3e}')
        if predict_only:
            return
        else:
            gt_all = torch.cat(gt_all)
            if args.n_traj_out > 0:
                gt_all = gt_all[:args.n_traj_out]
            np.save(f'{args.dir}/{args.file_out}gt.npy', utility.to_channellast(gt_all, self.dim) if self.args.channel_first else gt_all)
            mse_detail = np.concatenate(mse_detail, 0)
            mae_detail = np.concatenate(mae_detail, 0)
            mse, mae = np.mean(mse_detail), np.mean(mae_detail)
            print(f'valid per channel/seq: mse {a1line(np.mean(mse_detail,axis=(0,)))} / {a1line(np.mean(mse_detail,axis=(1,)))}')
            print(f'valid per channel/seq: mae {a1line(np.mean(mae_detail,axis=(0,)))} / {a1line(np.mean(mae_detail,axis=(1,)))}')
            if args.data_suffix == 'xyz':
                from NPS_common.io_utils import save_traj
                save_traj(f'{self.args.dir}/{args.file_out}gt.xyz', gt_all, symbols=symbols_all, CoM=True)
                save_traj(f'{self.args.dir}/{args.file_out}pd.xyz', pd_all, symbols=symbols_all, CoM=True)
            return np.mean(losses), mse, mae

    def train_batch(self, x, epoch=1, is_train=True):
        args = self.args
        if is_train: self.optimizer.zero_grad()
        # y = self.model(x)
        # if len(y)==2 and y[1] is None: y=y[0] # silly backward compatibility fix when model return x(t+1), None
        # loss = self.loss(y, x[self.args.node_y])
        y, loss, loss_item = self.model_y_loss(x, target=x[self.args.node_y], loss_from_model=args.loss_from_model)
        if is_train: loss.backward()
        if is_train: self.training_callback()
        return loss.item(), [np.mean(loss_item, 0)] if loss_item else []

    def evaluate_batch(self, x):
        # args = self.args
        y = self.model(x)
        if len(y)==2 and y[1] is None: y=y[0]
        return y



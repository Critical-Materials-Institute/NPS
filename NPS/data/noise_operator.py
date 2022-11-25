__author__ = 'Fei Zhou'

# Data augmentation with noise
import torch
import torch.nn as nn
import re


# class AdditiveNoise(nn.modules):
#     def __init__(self, ops, channels, noise):
#         super().__init__()
#         ops = re.split(';|,', ops)
#         self.op = nn.Sequential(*ops)

#     def forward(self, input, target):
#         return torch.mean((input - target)**2 * (target**2))

class noise_operator(nn.Module):
    def __init__(self, ops_str:str):
        """Example: 'add_uniform/0:1/1e-2', 'mul_uniform/1:2/1e-2', 'drop/1:6/0.3', 'add_normal/0:1/1e-3,mul_uniform/-2:9/1e-2' """
        super().__init__()
        # if (not ops_str) and (noise_simpleadd>0):
        #     ops_str = f'add_normal/:/{noise_simpleadd}'
        ops = []
        for op in filter(None, re.split(';|,', ops_str)):
            op_name, channel, noise = op.split('/')
            # channel = slice(*list(map(int, channel.split(':'))))
            channel = channel.replace(':',',');
            channel = eval(f'slice({channel})')
            try:
                noise = float(noise)
            except:
                noise = list(map(float, noise.split(':')))
            ops.append((op_name, channel, noise))
        self.ops = ops

    def __call__(self, x):
        for op_name, channel, noise in self.ops:
            if op_name == 'add_normal':
                x[...,channel].add_(noise*torch.randn_like(x[...,channel]))
            elif op_name == 'add_uniform':
                x[...,channel].add_(noise*(2*torch.rand_like(x[...,channel])-1))
            if op_name == 'mul_normal':
                x[...,channel].mul_(1+noise*torch.randn_like(x[...,channel]))
            elif op_name == 'mul_uniform':
                x[...,channel].mul_(1+noise*(2*torch.rand_like(x[...,channel])-1))
            elif op_name == 'drop':
                # noise means probability to drop, i.e. multiply by uniform(0,1)
                x[...,channel].mul_(torch.maximum(torch.rand_like(x[...,channel]), (torch.rand_like(x[...,channel])>=noise).float()))
        return x

if __name__ == '__main__':
    x = torch.randn(4,8)
    print(f'x = {x}')
    for typ_str in ('add_uniform/0:1/1e-2', 'mul_uniform/1:2/1e-2', 'drop/1:6/0.3', 'add_normal/0:1/1e-3,mul_uniform/-2:9/1e-2'):
        print(f'  noise type {typ_str}')
        noise_op = noise_operator(typ_str)
        xp = noise_op(x.clone())
        print(xp, xp-x)


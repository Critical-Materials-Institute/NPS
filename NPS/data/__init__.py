__author__ = 'Fei Zhou'

from importlib import import_module
from sklearn.model_selection import train_test_split

def load_data(args, typ, datf, num_workers=1, split='train'):
    # if typ in ('longclip', 'graph_clip', 'biased_clip', 'vartime_clip'):
    if args.data_is_time_series:
        if split == 'train':
            n_in, n_out, clip_step = args.n_in, args.n_out, args.clip_step
        elif split == 'predict':
            n_in, n_out, clip_step = args.n_in_test, 0, args.clip_step_test
        else:
            n_in, n_out, clip_step = args.n_in_test,args.n_out_test,args.clip_step_test
        nskip = args.nskip
    # if typ == 'longclip':
    #     from NPS.data.longclip import longclip
        m = getattr(import_module('NPS.data.' + typ), typ)
        ds = m(args, datf, n_in+n_out, clip_step, nskip=nskip, split=split)
        print(f'Loaded {typ} {datf} size {ds.flat.shape if hasattr(ds.flat, "shape") else len(ds.flat)} start_pos {ds.start_pos.shape} stat {ds.statistics}')
        assert ds.start_pos.size > 0
        return ds
    # elif typ == 'graph_clip':
    #     from NPS.data.graph_clip import graph_clip
    #     ds = graph_clip(args, datf, n_in+n_out, clip_step, nskip=nskip)
    #     print(f'Loaded graph_clip {datf} size {len(ds.flat)} start_pos {ds.start_pos.shape} stat {ds.statistics}')
    #     return ds
    elif typ == 'hubbard1band':
        from NPS.data.hubbard1band import load_hubbard1band
        # print(f'debug data', args.data, load_hubbard1band(args.data, cache_data=args.cache))
        return load_hubbard1band(args.data, cache_data=args.cache, filter=args.datatype[12:])
    else:
        try:
            module_name = typ
            m = import_module('NPS.data.' + module_name)
            return getattr(m, module_name)(args, datf)
        except:
            idx = typ.rfind('.')
            module_name, ds_name = typ[:idx], typ[idx+1:]
            m = import_module(typ)
            return getattr(m, ds_name)(args, datf)


class Data:
    def __init__(self, args):
        self.statistics = {}
        if args.single_dataset:
            ds = load_data(args, args.datatype, args.data, args.n_threads)
            self.train, self.test = train_test_split(ds, train_size=args.train_split, random_state=args.dataset_seed)
            self.statistics = ds.statistics
        else:
            self.train = load_data(args, args.datatype, args.data_train, args.n_threads, 'train') if (args.mode == 'train') else None
            self.test = load_data(args, args.datatype_test, args.data_test, args.n_threads, 'predict' if args.mode == 'predict' else 'valid')
            try:
                self.statistics = self.train.statistics
            except:
                pass
        if args.dataloader == 'torch':
            from torch.utils.data import DataLoader as loader
        elif args.dataloader == 'geometric':
            from torch_geometric.loader import DataLoader as loader
        else:
            raise ValueError(f'Unknown dataloader {args.dataloader}')
        self.loader_func = loader
        if args.dataloader and (not args.model_preprocess_data):
            if self.train is not None:
                self.train = loader(self.train, batch_size=args.batch, shuffle=True)
            self.test = loader(self.test, batch_size=args.batch_test, shuffle=False, drop_last=False)


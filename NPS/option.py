import argparse
import sys, re
from importlib import import_module
from NPS_common.utils import str2list

parser = argparse.ArgumentParser(description='NPS')
#################### Add model-specific arguments ####################
post_processors = []
for x in sys.argv[1:]:
    if x.startswith(('--register_args=', '--model=', '--datatype=', '--dataset=', '--core=')):
        x = ('NPS.data.' if x.startswith('--data') else '') + re.sub(r'^--\w*=', '', x)
        try:
            module = import_module(x)
        except:
            print(f'  Unable to import {x}')
            continue
        if hasattr(module, "register_args"):
            module.register_args(parser)
        else:
            print(f"  register_args not found in {x}")
        if hasattr(module, "post_process_args"):
            post_processors.append(module.post_process_args)
        else:
            print(f"  post_process_args not found in {x}")
        print(f'Registered commandline arguments from {x}')

#################### Book-keeping/misc ####################
parser.add_argument('--register_args', type=str, nargs='*', help='Module registering specific args with "def register_args(parser)" and "def post_process_args(args)"')
parser.add_argument('--jobid', type=str, default='jobid', help='job id')
parser.add_argument('--dir', type=str, default='', help='job directory')
parser.add_argument('--mode', type=str, default='train', choices=('train', 'eval', 'valid', 'predict', 'rollout', 'trace'),help='job type: train; eval|valid; predict|rollout')
# parser.add_argument('--template', default='.',
#                     help='You can set various templates in option.py')
parser.add_argument('--seed', type=int, default=54321, help='Rand seed')
parser.add_argument('--dataset_seed', type=int, default=54321, help='Rand seed for datase')
parser.add_argument('--debug', action='store_true', help='Enables debug mode')
# Hardware specifications
parser.add_argument('--n_threads', type=int, default=3, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')

#################### Logging ####################
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
# parser.add_argument('--load', type=str, default='.',
#                     help='file name to load')
parser.add_argument('--resume', type=str, default="", help='Load saved model from latest(default in training mode)|best(default in valid mode)|ema|pre(from --pre_train)')
parser.add_argument('--print_model', action='store_true', default=True,
                    help='print model')
parser.add_argument('--no-print_model', dest='print_model', action='store_false')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

#################### Data ####################
# NOTE: test simply means the dataset for validation during training or testing/prediction, i.e. non training
parser.add_argument('--data', type=str, default='', help='Directory to load dataset from.')
parser.add_argument('--dataloader', type=str, default='torch', choices=('', 'torch', 'geometric'), help='Loader')
parser.add_argument('--single_dataset', type=int, default=0, help='load from a single dataset')
parser.add_argument('--train_split', type=float, default=0.9, help='train set ratio if single_dataset')
parser.add_argument('--data_train', '--file_train', type=str, default=None, help='train dataset. Empty to default to DATA/train.npy')
parser.add_argument('--data_valid', '--file_test', type=str, default=None, help='valid dataset. Empty to default to DATA/valid.npy')
parser.add_argument('--data_predict', '--file_predict', type=str, default=None, help='predict dataset. Empty to default to DATA/test.npy')
parser.add_argument('--test_set', type=str, default='', choices=('', 'valid', 'predict'), help='Dataset for test, i.e. valid or predict.')
parser.add_argument('--datatype', '--dataset', type=str, default='longclip', help='data type')
parser.add_argument('--datatype_valid', type=str, default='', help='validation dataset type. Defaults to --datatype')
parser.add_argument('--datatype_predict', type=str, default='', help='prediction dataset type. Defaults to --datatype')
parser.add_argument('--dim', type=int, default=2, help='dimension (default 2d image, or 3d simulation')
parser.add_argument('--periodic', action='store_true', default=False, help='NPS periodic boundary condition')
parser.add_argument('--frame_shape', type=str, default='64', help='frame shape, e.g. 64,64')
parser.add_argument('--nfeat_in', type=int, default=1, help='nfeat_in')
parser.add_argument('--nfeat_out', type=int, default=-1, help='nfeat_out, default (-1) to nfeat_in')
parser.add_argument('--nfeat_out_global', type=int, default=0, help='no. of global output channels per graph')
parser.add_argument('--cache', type=bool, default=False, help='Cache whole dataset into memory')
parser.add_argument('--n_in', type=int, default=1, help='no. of input frames')
parser.add_argument('--n_in_valid', type=int, default=-1, help='n_in for validation')
parser.add_argument('--n_in_predict', type=int, default=-1, help='n_in for prediction')
parser.add_argument('--n_out', type=int, default=1, help='no. of output frames')
parser.add_argument('--n_out_valid', type=int, default=-1, help='n_out for validation')
parser.add_argument('--n_out_predict', type=int, default=-1, help='n_out for prediction')
parser.add_argument('--clip_step', type=int, default=1, help='No. of starting frames to skip when sampling the next sequence: starting frames are ::clip_step')
parser.add_argument('--nskip', type=int, default=1, help='Sampled clip are start_frame:start_frame+(n_in+n_out)*nskip:nskip')
parser.add_argument('--clip_step_valid', type=int, default=-1, help='clip_step for validation')
parser.add_argument('--clip_step_predict', type=int, default=-1, help='clip_step for prediction')
parser.add_argument('--data_slice', default='', help='Slice input data, default no slicing, Example: ":50" limits training sequences to 50 (to study training vs dataset size); "...,:1" ignores channels after 1st. THIS CHANGES DATA ITSELF')
parser.add_argument('--data_filter', default='', help='Filter with a bool function on sequences, default off, e.g. "np.mean(x)>1". THIS CHANGES DATA ITSELF')
parser.add_argument('--data_preprocess', default='', help='Preprocess data, default off, e.g. function "np.clip(x,0,1)" ; dict "{"name":"fft","tkeep": 0.1,"skeep":0.2}". THIS CHANGES DATA ITSELF')
parser.add_argument('--space_CG', action='store_true', help='Spatial coarse-graining by averaging, as specified by frame_shape')
parser.add_argument('--time_CG', type=int, default=1, help='Time coarse-graining by averaging, default 1 (off), 2 to average over 2 frames, etc')
parser.add_argument('--channel_first', action='store_true', help='Transform a channel last dataset to channel first after reading')
parser.add_argument('--data_setting', default='{}', help='Additional data settings, e.g. "{\'splitoffset\'=100}"')
parser.add_argument('--batch', '--minibatch_size', type=int, default=4, help='batch size')
parser.add_argument('--batch_valid', type=int, default=-1, help='batch size for validation')
parser.add_argument('--batch_predict', type=int, default=1, help='batch size for prediction')


#################### Model ####################
parser.add_argument('--model', type=str, default='base_evolution', help='Model module with "def make_model(args)"')
parser.add_argument('--model_preprocess_data', action='store_true', help='Call mode.preprocess_data on dataset')
# parser.add_argument('--model_setting', default='{}',
#                     help='model settings, e.g. "{\'unique_op\'=False}"')
parser.add_argument('--RNN', type=int, default=0, help='1 = recurrent (default), 0=feedforward (empty memory)')
parser.add_argument('--nfeat_hid', type=str, default='128', help='latent features. Or latent node features in GNN')
parser.add_argument('--nfeat_hid_edge', type=int, default=-1,  help='nfeat_latent of edge in GNN')
parser.add_argument('--n_mpassing', '--num_layers', type=int, default=2, help='num. of message passing')
parser.add_argument('--nlayer_mlp', type=int, default=2, help='No. of layer in MLP')
parser.add_argument('--kernel_size', default='3', help='convolutional kernel size, e.g. 3 or 3,1,1 for each layer')
parser.add_argument('--stride', type=int, default=1, help='convolutional stride')
parser.add_argument('--act', type=str, default='relu', help='Activation in MLP, e.g. relu')
# parser.add_argument('--bn', action='store_true', help='Add a batch normalization layer')
parser.add_argument('--norm_layer', type=str, default='', help='empty: no normalization layer; bn: batch_norm; ln: layerNorm; ln-1: non-trainable layerNorm')
parser.add_argument('--evaluator', type=str, default='cfd_eval', help='Select rollout method.')
parser.add_argument('--conserved', default='0', help='comma separated flags for whether the channels are conserved')
parser.add_argument('--patch_size', type=int, default=1, help='patch size for reshaping input')
# GNN specific
parser.add_argument("--n_node_type", default=2, type=int, help="No. of node types")
parser.add_argument("--node_type", default='', help="Node types")
parser.add_argument("--n_edge_type", default=1, type=int, help="No. of edge types")
parser.add_argument('--edge_cutoff', type=float, default=8.1, help='GNN edge cutof distance')
parser.add_argument('--node_y', type=str, default='node_y', help='key node_y against which to train the property in train_non_seq')
# ### Mesh
# parser.add_argument('--amr_N', type=int, default=64, help='system size, i.e. how many (fine) grids totally')
# parser.add_argument('--amr_N1', type=int, default=1, help='how many (fine) grids to bin into one, 1 to disable')
# parser.add_argument('--amr_buffer', type=int, default=1, help='how many buffer grids (must be 0 or 1)')
# parser.add_argument('--amr_eval_freq', type=int, default=1, help='Call AMR in eval every this many times (default 1)')
# parser.add_argument('--amr_threshold', type=float, default=1e-3, help='threshold to coarsen regions if values are close')

parser.add_argument('--add_c0', action='store_true', help='Add c0 (like ResNet)')
parser.add_argument('--no_putback_conv', action='store_true', help='Skip last convolution layer')
parser.add_argument('--model_deter', default='predrnn', help='deterministic model (eg. rnn, feedforward)')
parser.add_argument('--ngram', type=int, default=1, help='n-gram (no. of frames as one input)')
parser.add_argument('--model_stoch', default='simple_diffusion', help='stochastic model')
# Specifications for model_deter = feedforward
parser.add_argument('--ff_model', default='resnet', help='Which feedforward model')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
# Specifications for model_stoch
parser.add_argument('--stoch_hidden', default='10', help='number of features for stochasticity')
parser.add_argument('--stoch_kernel', default='3', help='convolutional kernel size for stochasticity, e.g. 3 or 3,1,1 for each layer')
parser.add_argument('--stoch_numNN', type=int, default=1, help='1 for nearest neighbor, 2 for 2-NN, etc')
parser.add_argument('--stoch_act', type=str, default='relu', help='activation function for stochasticity')
parser.add_argument('--stoch_onsite', default='0', help='comma separated flags for whether the channels contain onsite noise')
parser.add_argument('--stoch_skiponsite', action='store_true', help='skip onsite noise altogether')
parser.add_argument('--f1', type=float, default=1,
                    help='float1 for debug')
parser.add_argument('--s1', default='',
                    help='string1 for debug')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

#################### Training ####################
# parser.add_argument('--test_only', action='store_true',
#                     help='set this option to test the model. Sets file_test to test.npy if empty.')
parser.add_argument('--trainer', default='trainer', help='Trainer module with "def make_trainer(args, loader, model, loss, checkpoint)". Try subclass Trainer with new train_batch and evaluate_batch methods')
parser.add_argument('--nepoch', '--epochs', type=int, default=8000000, help='nepoch')
parser.add_argument('--epoch_size', type=int, default=-1, help='epoch size. -1=whole train set')
parser.add_argument('--keep_ckpt', type=int, default=-1, help='number of checkpoints to keep. -1 to default(5)')
parser.add_argument('--print_freq', '--print_every', type=int, default=20, help='how many batches to wait before logging training status')
parser.add_argument('--valid_freq', '--valid_every', type=int, default=1, help='Perform validation/checkpoint every this many epochs')
parser.add_argument('--n_training_steps', type=int, default=int(10e6), help='No. of training steps')
parser.add_argument('--data_aug', type=str, default='spg,noise', help='Data augmentation by pointgroup, cropping, noise, etc')
parser.add_argument('--pointgroup', default='1',
                    help='point group symmetry for data augmentation during training')
parser.add_argument('--slice_op', type=str, default='', help='slicing operator to train on a smaller piece')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
parser.add_argument('--noise', type=float, default=0.0, help='noise magnitude')
parser.add_argument('--noise_op', type=str, default='', help="Example: 'add_uniform/0:1/1e-2', 'mul_uniform/1:2/1e-2', 'drop/1:6/0.3', 'add_normal/0:1/1e-3,mul_uniform/-2:9/1e-2' If empty, use add/normal and --option")
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
# parser.add_argument('--split_batch', type=int, default=1,
#                     help='split the batch into smaller chunks')
# parser.add_argument('--self_ensemble', action='store_true',
#                     help='use self-ensemble method for test')
parser.add_argument('--visualization_setting', default='{}', help='settings for plotting, e.g.{cmin=-1,cmax=1,Tmin=0,Tmax=10}')
# parser.add_argument('--gan_k', type=int, default=1,
#                     help='k value for adversarial loss')
parser.add_argument('--freeze', default='', help='empty or deter|stoch to freeze certain net')
## Exponential Moving Average
parser.add_argument('--ema_decay', type=float, default=0.995, help='ExponentialMovingAverage decay')
## Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--loss_wt', type=str, default='', help='weight of itemized losses, e.g. 1e0,1e-2')
parser.add_argument('--loss_from_model', type=int, default=0, help='provide target to model so it produces (y, loss), not just y')
# parser.add_argument('--stoch_loss', default='NLL', help='')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')
parser.add_argument('--rnn_scheduled_sampling', type=str, default='GT',
  help="""PD:always use PD
  GT: always use GT
  decrease: set sampling_stop_iter sampling_start_value sampling_changing_rate
  reverse: set r_sampling_step_1 r_sampling_step_2 r_exp_alpha""")
### scheduled sampling
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)
### reverse scheduled
parser.add_argument('--r_sampling_step_1', type=float, default=25000)
parser.add_argument('--r_sampling_step_2', type=int, default=50000)
parser.add_argument('--r_exp_alpha', type=int, default=5000)

## optimizer
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_min', type=float, default=1e-7, help='Minimum learning rate')
parser.add_argument('--wd', '--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta1')
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon')
parser.add_argument('--momentum', type=float, default=0, help='momentum')
# scheduler
parser.add_argument('--scheduler', type=str, default='plateau', help='lr reduction schedule')
parser.add_argument('--lr_decay_step', type=int, default=500, help='Learning rate decay factor.')
parser.add_argument('--lr_decay_factor', type=float, default=0.3, help='Learning rate decay steps.')
parser.add_argument('--lr_decay_patience', type=int, default=4, help='Learning rate decay steps.')
# parser.add_argument('--lr_decay_type', type=str, default='step',
#                     help='learning rate decay type')
# parser.add_argument('--gamma', type=float, default=0.5,
#                     help='learning rate decay factor for step decay')
### Predict job
parser.add_argument('--traj_out', type=str, default='', help='file to save test trajectories')
parser.add_argument('--n_traj_out', '--n_rollout', type=int, default=-1, help='No. of rollout trajectories')
parser.add_argument('--predict_only', action='store_true',
                    help='set this option to test the model without GT.')
# parser.add_argument('--pred_length', type=int, default=-1,
#                     help='No. of frames in training. Default -1 means total_length-input_length')
# parser.add_argument('--pred_length_test', type=int, default=-1,
#                     help='No. of frames to predict in test. Default -1 means total_length-input_length')
# parser.add_argument('--tbegin_test', type=int, default=-1, help='Step from which to save in test. Default -1')
# parser.add_argument('--tskip_test', type=int, default=1, help='Save every tskip steps in test. Default 1')
# parser.add_argument('--testpath', type=str, default='',
#                     help='dataset directory for testing')
# parser.add_argument('--testset', type=str, default='longclip',
#                     help='dataset name for testing')

#################### Training ####################
parser.add_argument('--export_wrapper', type=str, default='', help='module for tracing & exporting model')
parser.add_argument('--export_file', type=str, default='exported.pt', help='save file of exported (traced) model')

#################### Validation/Prediction ####################
parser.add_argument('--gt_in_out', action='store_true', help='Include the GT sequence in output sequence')

args = parser.parse_args()
# from NPS import template
# template.set_template(args)

#################### Book-keeping/misc ####################
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
if args.mode == 'eval': args.mode = 'valid'
if args.mode == 'rollout': args.mode = 'predict'
# args.test_only = (args.mode == 'predict')
if not args.dir: args.dir='experiment/'+args.jobid
args.device = 'cpu' if args.cpu else 'cuda'
if not args.resume:
    args.resume = 'latest' if args.mode=='train' else 'best'

#################### Data ####################
args.periodic = bool(args.periodic)
if not args.data_train:
    args.data_train = args.data+'/train'
if not args.data_valid:
    args.data_valid = args.data+f'/valid'
if not args.data_predict:
    args.data_predict = args.data+f'/test'
if not args.datatype_valid:
    args.datatype_valid = args.datatype
if not args.datatype_predict:
    args.datatype_predict = args.datatype
args.frame_shape = tuple(str2list(args.frame_shape))
if len(args.frame_shape) == 1:
    args.frame_shape *= args.dim
else:
    assert len(args.frame_shape) == args.dim, ValueError('frame shape mismatch')
if args.nfeat_out == -1:
    args.nfeat_out = args.nfeat_in
if args.n_in_valid == -1:
    args.n_in_valid = args.n_in
if args.n_in_predict == -1:
    args.n_in_predict = args.n_in_valid
if args.n_out_valid == -1:
    args.n_out_valid = args.n_out
if args.n_out_predict == -1:
    args.n_out_predict = args.n_out_valid
if args.clip_step_valid == -1:
    args.clip_step_valid = args.clip_step
if args.clip_step_predict == -1:
    args.clip_step_predict = args.clip_step_valid
if args.batch_valid == -1:
    args.batch_valid = args.batch
if args.batch_predict == -1:
    args.batch_predict = args.batch_valid
if not args.test_set:
    args.test_set = {'predict':'predict'}.get(args.mode, 'valid')
if args.test_set == 'predict':
    args.data_test = args.data_predict
    args.datatype_test = args.datatype_predict
    args.n_in_test = args.n_in_predict
    args.n_out_test = args.n_out_predict
    args.clip_step_test = args.clip_step_predict
    args.batch_test = args.batch_predict
else:
    args.data_test = args.data_valid
    args.datatype_test = args.datatype_valid
    args.n_in_test = args.n_in_valid
    args.n_out_test = args.n_out_valid
    args.clip_step_test = args.clip_step_valid
    args.batch_test = args.batch_valid
if args.ngram > 1: assert args.ngram <= args.n_in and args.ngram <= args.n_in_test
if args.node_type:
    args.node_type = args.node_type.split(',')
    assert args.n_node_type == len(args.node_type)

#################### Model ####################
args.kernel_size = str2list(args.kernel_size)
if len(args.kernel_size)==1: args.kernel_size=args.kernel_size[0] # to maintain compatibility
args.nfeat_hid = str2list(args.nfeat_hid)
if len(args.nfeat_hid) == 1: args.nfeat_hid = args.nfeat_hid[0]
if args.nfeat_hid_edge == -1:
    args.nfeat_hid_edge = args.nfeat_hid
args.RNN = bool(args.RNN)
# assert args.RNN or (args.n_in==args.ngram), ValueError('Set n_in=ngram when disabling RNN')

#################### Training ####################
if (not args.noise_op) and (args.noise > 0):
    args.noise_op = f'add_normal/0:None/{args.noise}'
args.loss_wt = str2list(args.loss_wt, float)

#parser.add_argument('--data_range', type=str, default='1-800/801-810',
#                    help='train/test data range')



# # each data point of shape [clip_length, *frame_shape, n_colors]
# parser.add_argument('--clip_length', type=int, default=20,
#                     help='No. of frames in a video clip')
# when reading data, may read every tskip frames, and read at maximum nclip_max clips
# parser.add_argument('--nclip_max', type=int, default=-1,
#                     help='read at maximum nclip_max clips (OBSELETE, use data_slice instead)')
# parser.add_argument('--tskip', type=int, default=1,
#                     help='skip every tskip frames')
# parser.add_argument('--benchmark_noise', action='store_true',
#                     help='use noisy benchmark sets')
# parser.add_argument('--n_train', type=int, default=800,
#                     help='number of training set')
# parser.add_argument('--n_val', type=int, default=5,
#                     help='number of validation set')
# parser.add_argument('--offset_val', type=int, default=800,
#                     help='validation index offest')
# parser.add_argument('--ext', type=str, default='sep_reset',
#                     help='dataset file extension')
# parser.add_argument('--chop', action='store_true',
#                     help='enable memory-efficient forward')



# Training specifications
# Prediction specs

# Optimization specifications



# # options for residual group and feature channel reduction
# parser.add_argument('--n_resgroups', type=int, default=10,
#                     help='number of residual groups')
# parser.add_argument('--reduction', type=int, default=16,
#                     help='number of feature maps reduction')

# args.num_hidden = str2list(args.num_hidden)
# args.stoch_hidden = str2list(args.stoch_hidden)
# args.stoch_kernel = str2list(args.stoch_kernel)


# args.conserved = str2list(args.conserved)
# assert set(args.conserved).issubset(set([0,1]))
# if len(args.conserved) == 1:
#     args.conserved *= args.n_colors
# else:
#     assert len(args.conserved) == args.n_colors
# args.any_conserved = 1 in args.conserved

# args.stoch_onsite = str2list(args.stoch_onsite)
# assert set(args.conserved).issubset(set([0,1]))
# if len(args.stoch_onsite) == 1:
#     args.stoch_onsite *= args.n_colors
# else:
#     assert len(args.stoch_onsite) == args.n_colors

for p in post_processors:
    p(args)

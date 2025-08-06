import argparse
import logging
import os
import random
import sys
import time 
import copy
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn

from models.blocks_resnet import *
from models.custom_modules import *
from models.custom_models_resnet import *
from models.custom_models_vgg import *
from models.feature_quant_module import *

from utils import *
import utils
from utils import printRed

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample

import utils_distill


start_time = time.time()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (CIFAR)")
# data and model
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--arch', type=str, default='resnet20_quant', help='model architecture')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')

# training settings
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=1200, help='number of epochs for training')
parser.add_argument('--optimizer_m', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for model paramters')
parser.add_argument('--optimizer_q', type=str, default='Adam', choices=('SGD','Adam'), help='optimizer for quantizer paramters')
parser.add_argument('--lr_m', type=float, default=1e-3, help='learning rate for model parameters')
parser.add_argument('--lr_q', type=float, default=1e-5, help='learning rate for quantizer parameters')
parser.add_argument('--lr_m_end', type=float, default=0.0, help='final learning rate for model parameters (for cosine)')
parser.add_argument('--lr_q_end', type=float, default=0.0, help='final learning rate for quantizer parameters (for cosine)')
parser.add_argument('--decay_schedule_m', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--decay_schedule_q', type=str, default='150-300', help='learning rate decaying schedule (for step)')
parser.add_argument('--lr_scheduler_m', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--lr_scheduler_q', type=str, default='cosine', choices=('step','cosine'), help='type of the scheduler')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for model parameters')
parser.add_argument('--gamma', type=float, default=0.1, help='decaying factor (for step)')

# arguments for quantization
parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='do weight quantization')
parser.add_argument('--QActFlag', type=str2bool, default=True, help='do activation quantization')
parser.add_argument('--weight_levels', type=int, default=2, help='number of weight quantization levels')
parser.add_argument('--act_levels', type=int, default=2, help='number of activation quantization levels')
parser.add_argument('--baseline', type=str2bool, default=False, help='training with STE')
parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0, help='scaling factor for weights')
parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0, help='scaling factor for activations')
parser.add_argument('--use_hessian', type=str2bool, default=True, help='update scaling factor using Hessian trace')
parser.add_argument('--update_every', type=int, default=10, help='update interval in terms of epochs')
parser.add_argument('--quan_method', type=str, default='EWGS', help='training with different quantization methods')

# arguments for feature quantization
parser.add_argument('--model_type', type=str, default='student', choices=['fp', 'teacher', 'student'], help='training mode: fp model training / teacher model training / knowledge distillation')
parser.add_argument('--QFeatureFlag', type=str2bool, default=False, help='add feature quantizer to model')
parser.add_argument('--feature_levels', type=int, default=2, help='number of feature quantization levels')
parser.add_argument('--bkwd_scaling_factorF', type=float, default=0.0, help='Scaling factor for feature quantization')
parser.add_argument('--use_student_quant_params', type=str2bool, default=False, help='Enable the use of student quantization parameters during teacher quantization')
parser.add_argument('--use_adaptor', type=str2bool, default=False, help='Enable the use of adaptor(connector) for Student')
parser.add_argument('--use_adaptor_bn', type=str2bool, default=True, help='whether to use batchnorm in adaptor')
parser.add_argument('--distill_loss', type=str, default='L1', choices=['L1', 'L2', 'KL_Div'], help='Feature Distillation Loss type, L1(MAE) or L2(MSE) or KL_Div(Kullback–Leibler divergence)')

# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
parser.add_argument('--log_dir', type=str, default='./results/ResNet20_CIFAR10/W1A1/')
parser.add_argument('--vis_log_dir', type=str, default='./results/feature_analysis/')
parser.add_argument('--load_pretrain', type=str2bool, default=True, help='load pretrained full-precision model')
parser.add_argument('--pretrain_path', type=str, default='./results/CIFAR10_ResNet20/fp/checkpoint/last_checkpoint.pth', help='path for pretrained full-preicion model')

# knowledge distillation
parser.add_argument('--distill', type=str, default=None, choices=['kd', 'fd', 'crdst','hint', 'attention', 'similarity', 'correlation', 
                                                                    'vid', 'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound', 'factor', 'nst'])
parser.add_argument('--teacher_path', type=str, default='./results/CIFAR10_ResNet20/fp/checkpoint/last_checkpoint.pth', help='path for pretrained teacher model with quantizer')
parser.add_argument('--teacher_arch', type=str, default='resnet20_fp', help='teacher model architecture')
parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
parser.add_argument('--kd_gamma', type=float, default=None, help='weight for classification')
parser.add_argument('--kd_alpha', type=float, default=None, help='weight balance for KD')
parser.add_argument('--kd_beta', type=float, default=None, help='weight balance for other losses or crd loss')
parser.add_argument('--kd_theta', type=float, default=None, help='weight balance for crdSt losses')

# NCE distillation
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_t', default=0.1, type=float, help='temperature parameter for softmax') 
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
parser.add_argument('--head', default='linear', type=str, choices=['linear', 'mlp', 'pad'])

# hint layer
parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

# CKTF, called crdst in the code
parser.add_argument('--st_method', type=str, default='Last', choices=['Last', 'Smallest', 'Largest', 'First', 'Random'])

# abound, factor, fsp
parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')


args = parser.parse_args()
arg_dict = vars(args)

### make log directory
if not os.path.exists(args.log_dir):
    os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')
log_string = 'configs\n'
for k, v in arg_dict.items():
    log_string += "{}: {}\t".format(k,v)
    print("{}: {}".format(k,v), end='\t')
logging.info(log_string+'\n')
print('')

### GPU setting
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### set the seed number
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


if args.dataset == 'cifar10':
    args.num_classes = 10
    if args.distill == 'crd' or args.distill == 'crdst':
        train_dataset, test_dataset = get_cifar10_dataloaders_sample(data_folder="./dataset/data/CIFAR10/", k=args.nce_k, mode=args.mode)
        
    else:
        train_dataset, test_dataset = get_cifar10_dataloaders(data_folder="./dataset/data/CIFAR10/")

elif args.dataset == 'cifar100':
    args.num_classes = 100
    if args.distill == 'crd' or args.distill == 'crdst':
        train_dataset, test_dataset = get_cifar100_dataloaders_sample(data_folder="../data/CIFAR100/", k=args.nce_k, mode=args.mode)
    else:
        train_dataset, test_dataset = get_cifar100_dataloaders(data_folder="../data/CIFAR100/", is_instance=False)

else:
    raise NotImplementedError


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        worker_init_fn=None if args.seed is None else _init_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=100,
                                        shuffle=False,
                                        num_workers=args.num_workers)
printRed(f"dataset: {args.dataset}, num of training data (50,000): {len(train_dataset)}, number of testing data (10,000): {len(test_dataset)}")                                          


### initialize model
model_class = globals().get(args.arch)
model = model_class(args)
model.to(device)


num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))


if args.load_pretrain:
    trained_model = torch.load(args.pretrain_path, weights_only=True)
    current_dict = model.state_dict()
    printRed("Pretrained full precision weights are initialized")
    logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''
    for key in trained_model['model'].keys():
        if key in current_dict.keys():
            log_string += '{}\t'.format(key)
            current_dict[key].copy_(trained_model['model'][key])
    logging.info(log_string+'\n')
    model.load_state_dict(current_dict)

else:
    printRed("Not initialized by the pretrained full precision weights")

# initialize quantizer params
init_quant_model(model, train_loader, device, args.distill)


if args.quan_method == "EWGS" or args.baseline:
    define_quantizer_scheduler = True
else:
    define_quantizer_scheduler = False


### initialize optimizer, scheduler, loss function
if args.quan_method == "EWGS" or args.baseline:
    trainable_params_s = list(model.parameters())
    model_params_s = []
    quant_params_s = []

    for m in model.modules():
        if isinstance(m, QConv):
            model_params_s.append(m.weight)
            if m.bias is not None:
                model_params_s.append(m.bias)
            if m.quan_weight:
                quant_params_s.append(m.lW)
                quant_params_s.append(m.uW)
            if m.quan_act:
                quant_params_s.append(m.lA)
                quant_params_s.append(m.uA)
                quant_params_s.append(m.lA_t)
                quant_params_s.append(m.uA_t)
            if m.quan_act or m.quan_weight:
                quant_params_s.append(m.output_scale)
            print("QConv", m)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            model_params_s.append(m.weight)
            if m.bias is not None:
                model_params_s.append(m.bias)
            print("nn", m)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.affine:
                model_params_s.append(m.weight)
                model_params_s.append(m.bias)

    total_params_s = sum(p.numel() for p in trainable_params_s)
    model_params_count_s = sum(p.numel() for p in model_params_s)
    quant_params_count_s = sum(p.numel() for p in quant_params_s)

    print("# total student params:", total_params_s)
    print("# student model params:", model_params_count_s)
    print("# student quantizer params:", quant_params_count_s)
    logging.info("# total student params: {}".format(total_params_s))
    logging.info("# student model params: {}".format(model_params_count_s))
    logging.info("# student quantizer params: {}".format(quant_params_count_s))

    if total_params_s != (model_params_count_s + quant_params_count_s):
        raise Exception('Mismatched number of trainable parmas')
else:
    raise NotImplementedError(f"Not implement {args.quan_method}!")


if args.distill:
    args_t = copy.deepcopy(args)
    args_t.model_type = 'teacher'

    model_class_t = globals().get(args_t.teacher_arch)
    model_t = model_class_t(args_t)
    model_t.to(device)

    trainable_params_t = list(model_t.parameters())
    model_params_t = []
    quant_params_t = []

    for m in model_t.modules():
        if isinstance(m, FeatureQuantizer):
            quant_params_t.append(m.lF)
            quant_params_t.append(m.uF)
            print("FeatureQuantizer", m)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            model_params_t.append(m.weight)
            if m.bias is not None:
                model_params_t.append(m.bias)
            print("nn", m)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.affine:
                model_params_t.append(m.weight)
                model_params_t.append(m.bias)

    total_params_t = sum(p.numel() for p in trainable_params_t)
    model_params_count_t = sum(p.numel() for p in model_params_t)
    quant_params_count_t = sum(p.numel() for p in quant_params_t)

    print("# Teacher total params:", total_params_t)
    print("# Teacher model params:", model_params_count_t)
    print("# Teacher quantizer params:", quant_params_count_t)
    logging.info("# Teacher total params: {}".format(total_params_t))
    logging.info("# Teacher model params: {}".format(model_params_count_t))
    logging.info("# Teacher quantizer params: {}".format(quant_params_count_t))
    if total_params_t != (model_params_count_t + quant_params_count_t):
        raise Exception('Mismatched number of trainable parmas')

    model_t = utils.load_teacher_model(model_t, args.teacher_path)
    num_training_data = len(train_dataset)
    module_list, model_params, criterion_list = utils_distill.define_distill_module_and_loss(model, model_t, model_params_s, args, num_training_data, train_loader)


# optimizer and scheduler for model params
if args.optimizer_m == 'SGD':
    optimizer_m = torch.optim.SGD(model_params, lr=args.lr_m, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer_m == 'Adam':
    optimizer_m = torch.optim.Adam(model_params, lr=args.lr_m, weight_decay=args.weight_decay)

if args.lr_scheduler_m == "step":
    if args.decay_schedule_m is not None:
        milestones_m = list(map(lambda x: int(x), args.decay_schedule_m.split('-')))
    else:
        milestones_m = [args.epochs+1]
    scheduler_m = torch.optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=milestones_m, gamma=args.gamma)
elif args.lr_scheduler_m == "cosine":
    scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m, T_max=args.epochs, eta_min=args.lr_m_end)


# optimizer and scheduler for quantizer params
if define_quantizer_scheduler:
    if args.optimizer_q == 'SGD':
        optimizer_q = torch.optim.SGD(quant_params_s, lr=args.lr_q)
    elif args.optimizer_q == 'Adam':
        optimizer_q = torch.optim.Adam(quant_params_s, lr=args.lr_q)

    if args.lr_scheduler_q == "step":
        if args.decay_schedule_q is not None:
            milestones_q = list(map(lambda x: int(x), args.decay_schedule_q.split('-')))
        else:
            milestones_q = [args.epochs+1]
        scheduler_q = torch.optim.lr_scheduler.MultiStepLR(optimizer_q, milestones=milestones_q, gamma=args.gamma)
    elif args.lr_scheduler_q == "cosine":
        scheduler_q = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=args.epochs, eta_min=args.lr_q_end)


# for distillation
if args.distill:
    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    module_list.cuda()
    criterion_list.cuda()

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(args.log_dir)

### train
total_iter = 0
best_acc = 0
acc_last5 = []
iterations_per_epoch = len(train_loader)
lambda_dict = {}

print(f"iterations_per_epoch: {iterations_per_epoch}")
for ep in range(args.epochs):
    if args.distill:
        # set modules as train()
        for module in module_list:
            module.train()
        
        module_list[-1].eval()

        if args.distill == 'abound':
            module_list[1].eval()
        elif args.distill == 'factor':
            module_list[2].eval()

        criterion_cls = criterion_list[0]
        criterion_div = criterion_list[1]
        criterion_kd = criterion_list[2]

        model_s = module_list[0]
        model_t = module_list[-1]
    else:
        model.train()


    ### update grad scales
    if ep % args.update_every == 0 and ep != 0 and not args.baseline and args.use_hessian:
        update_grad_scales(model_s, model_t, train_loader, criterion_list, device, args)
        print("update grade scales")

    writer.add_scalar('train/model_lr', optimizer_m.param_groups[0]['lr'], ep)

    if define_quantizer_scheduler:
        writer.add_scalar('train/quant_lr', optimizer_q.param_groups[0]['lr'], ep)
        
    for i, data in enumerate(train_loader):            
        if args.distill == "crd" or args.distill == "crdst":
            images, labels, index, contrast_idx = data
            index = index.to(device)
            contrast_idx = contrast_idx.to(device)
        else:
            images, labels = data
            index = None
            contrast_idx = None
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer_m.zero_grad()
        if define_quantizer_scheduler:
            optimizer_q.zero_grad()
            
        if args.quan_method == "EWGS":
            save_dict = {"iteration": total_iter, "writer": writer, "layer_num": None, "block_num": None, "conv_num": None, "type": None}
            # for lambda
            if total_iter >= 2:
                for i in range(total_iter-1):
                    lambda_dict[f"{i}"] = {}
            lambda_dict[f"{total_iter}"] = {}
        else:
            save_dict = None
            lambda_dict = None

        if args.distill:
            flatGroupOut = True if args.distill == 'crdst' else False # for crdst
            preact = False
            if args.distill in ['abound']:
                preact = True
            
            # forward student mdoel
            feat_s, block_out_s, logit_s, quant_params, fd_map_s = model_s(images, save_dict, lambda_dict, is_feat=True, preact=preact, flatGroupOut=flatGroupOut)
            
            # forward teacher model
            with torch.no_grad():
                feat_t, block_out_t, logit_t, fd_map_t = model_t(images, is_feat=True, preact=preact,flatGroupOut=flatGroupOut, quant_params=quant_params)
                feat_t = [f.detach() for f in feat_t]

        else:
            pred = model(images, save_dict, lambda_dict)
        #======================================Backward======================================
        if args.distill:
            loss_cls = criterion_cls(logit_s, labels) # CE
            loss_div = criterion_div(logit_s, logit_t) # KL
            if args.distill == "crdst":
                loss_kd_crd, loss_kd_crdSt = utils_distill.get_loss_crdst(args, feat_s, feat_t, criterion_kd, index, contrast_idx, block_out_s, block_out_t)
                loss_total = args.kd_gamma * loss_cls + args.kd_alpha * loss_div + args.kd_beta * loss_kd_crd + args.kd_theta * loss_kd_crdSt 
            else:
                loss_fd = criterion_kd(fd_map_s, fd_map_t) # L1(MAE) or L2(MSE) or KL_Div
                loss_total = args.kd_gamma * loss_div + args.kd_alpha * loss_fd # SQAFD Loss
                
                loss_fd_value = loss_fd.item()
                loss_div_value = loss_div.item()
                loss_total_value = loss_total.item()
                writer.add_scalar('train/loss_fd', loss_fd_value, total_iter)
                writer.add_scalar('train/loss_div', loss_div_value, total_iter)
                writer.add_scalar('train/loss_total', loss_total_value, total_iter)
                
                content = f"total_iter={total_iter}, loss_fd={loss_fd_value}, loss_div={loss_div_value}, loss_total={loss_total_value}"
                with open(os.path.join(args.log_dir,'loss.txt'), "a") as w:
                    w.write(f"{content}\n")
            if i == 0:
                printRed(f"gamma: {args.kd_gamma}, alpha: {args.kd_alpha}, kd_beta: {args.kd_beta}, kd_theta: {args.kd_theta}")
                
        else:
            loss_total = criterion(pred, labels)
        
        loss = loss_total
        loss.backward()

        optimizer_m.step()
        if define_quantizer_scheduler:
            optimizer_q.step()

        writer.add_scalar('train/loss', loss.item(), total_iter)
        total_iter += 1

    scheduler_m.step()
    if define_quantizer_scheduler:
        scheduler_q.step()

    if ep % 100 == 0 and args.distill:
        qact_stats = {}

        if hasattr(model_s, 'qact') and model_s.qact is not None:
            last_batch_qact = model_s.qact.detach().clone()
        if fd_map_t is not None:
            last_batch_fd_map = fd_map_t.detach().clone()

        if last_batch_qact is not None:
            qact = last_batch_qact.cpu()
            qact_unique, qact_counts = torch.unique(qact, return_counts=True)

            qact_stats['student_qact'] = {
                    'values': qact_unique.tolist(),
                    'counts': qact_counts.tolist()
                }

        if last_batch_fd_map is not None:
            fd_map_t_detached = last_batch_fd_map.cpu()
            fd_map_t_unique, fd_map_t_counts = torch.unique(fd_map_t_detached, return_counts=True)

            qact_stats['teacher_fd_map'] = {
                    'values': fd_map_t_unique.tolist(),
                    'counts': fd_map_t_counts.tolist()
                }

        with open(os.path.join(args.log_dir, 'qact.txt'), 'a') as f:
            f.write(f"Epoch {ep}:\n")
            f.write(f"{json.dumps(qact_stats, indent=2)}\n\n")

    with torch.no_grad():
        model.eval()
        correct_classified = 0
        total = 0
        for i, data in enumerate(train_loader):
            if args.distill == "crd" or args.distill == "crdst":
                images, labels, index, contrast_idx = data
                index = index.to(device)
                contrast_idx = contrast_idx.to(device)
            else:
                images, labels = data
                index = None
                contrast_idx = None
            
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        writer.add_scalar('train/acc', test_acc, ep)

        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")
        logging.info("Current epoch: {:03d}\t Test accuracy: {}%".format(ep, test_acc))
        writer.add_scalar('test/acc', test_acc, ep)

        torch.save({
            'epoch':ep,
            'model':model.state_dict(),
            'test_acc': test_acc,
            'optimizer_m':optimizer_m.state_dict(),
            'scheduler_m':scheduler_m.state_dict(),
            'optimizer_q':optimizer_q.state_dict() if define_quantizer_scheduler else {},
            'scheduler_q':scheduler_q.state_dict() if define_quantizer_scheduler else {},
            'criterion':criterion.state_dict()
        }, os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch':ep,
                'model':model.state_dict(),
                'test_acc': test_acc,
                'optimizer_m':optimizer_m.state_dict(),
                'scheduler_m':scheduler_m.state_dict(),
                'optimizer_q':optimizer_q.state_dict() if define_quantizer_scheduler else {},
                'scheduler_q':scheduler_q.state_dict() if define_quantizer_scheduler else {},
                'criterion':criterion.state_dict()
            }, os.path.join(args.log_dir,'checkpoint/best_checkpoint.pth'))

        # for record the average acccuracy of the last 5 epochs
        if ep >= args.epochs - 5:
            acc_last5.append(test_acc)

    layer_num = 0
    for m in model.modules():
        if isinstance(m, QConv):
            layer_num += 1
            if args.QWeightFlag:
                writer.add_scalar("z_{}th_module/lW".format(layer_num), m.lW.item(), ep)
                logging.info("lW: {}".format(m.lW))
                writer.add_scalar("z_{}th_module/uW".format(layer_num), m.uW.item(), ep)
                logging.info("uW: {}".format(m.uW))
                writer.add_scalar("z_{}th_module/bkwd_scaleW".format(layer_num), m.bkwd_scaling_factorW.item(), ep)
                logging.info("grad_scaleW: {}".format(m.bkwd_scaling_factorW.item()))
            if args.QActFlag:
                writer.add_scalar("z_{}th_module/lA".format(layer_num), m.lA.item(), ep)
                logging.info("lA: {}".format(m.lA))
                writer.add_scalar("z_{}th_module/uA".format(layer_num), m.uA.item(), ep)
                logging.info("uA: {}".format(m.uA))
                writer.add_scalar("z_{}th_module/bkwd_scaleA".format(layer_num), m.bkwd_scaling_factorA.item(), ep)
                logging.info("grad_scaleA: {}".format(m.bkwd_scaling_factorA.item()))
            if args.QActFlag or args.QWeightFlag:
                writer.add_scalar("z_{}th_module/output_scale".format(layer_num), m.output_scale.item(), ep)
                logging.info("output_scale: {}".format(m.output_scale))
            logging.info('\n')

    if args.distill:
        for name, module in model_t.named_modules():
            if isinstance(module, FeatureQuantizer):
                logging.info(f"Module Name: {name}")
                logging.info(f"  lF: {module.lF.item()}")
                logging.info(f"  uF: {module.uF.item()}")
                writer.add_scalar(f"teacher_{name}/lF", module.lF.item(), ep)
                writer.add_scalar(f"teacher_{name}/uF", module.uF.item(), ep)


checkpoint_path_last = os.path.join(args.log_dir, 'checkpoint/last_checkpoint.pth')
checkpoint_path_best = os.path.join(args.log_dir, 'checkpoint/best_checkpoint.pth')
utils.test_accuracy(checkpoint_path_last, model, logging, device, test_loader)
utils.test_accuracy(checkpoint_path_best, model, logging, device, test_loader)

mean_last5 = round(np.mean(acc_last5),2)
print(f"Average accuracy of the last 5 epochs: {mean_last5}, acc_last5: {acc_last5}\n")
logging.info(f"Average accuracy of the last 5 epochs: {mean_last5}, acc_last5: {acc_last5}\n")

print(f"Total time: {(time.time()-start_time)/3600}h")
logging.info(f"Total time: {(time.time()-start_time)/3600}h")

print(f"Save to {args.log_dir}")
logging.info(f"Save to {args.log_dir}")

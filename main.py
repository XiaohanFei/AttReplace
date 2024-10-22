import copy
import torch
import datetime
import argparse

import torch.nn as nn
import torch.backends.cudnn as cudnn

from datasets import build_dataset
from loss import CosineSimilarityLoss
from train import train_model, evaluate
from models import MixerBlock, EmptyBlock, param_dict_all, param_dict_qkv

from pathlib import Path
from timm.utils import NativeScaler
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler


def get_args_parser():
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", add_help=False)
    
    # deit model parameters
    parser.add_argument("--d-model", default="deit_tiny_patch16_224", type=str, metavar="MODEL")
    parser.add_argument("--input-size", default=224, type=int, help="expected images size for model input")
    parser.add_argument("--d-weight", default="", help="path of DeiT model checkpoint")
    
    # mixer model parameters
    parser.add_argument("--replace", nargs="+", type=int, help="list for index of blocks to be replaced")
    parser.add_argument("--rep-mode", default="all", choices=["qkv", "all"], 
                        help="Choose to relace whole attention block or only qkv part")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--eval-model", default="", help="Path of model to be evaluated")
    parser.add_argument('--train', action='store_true', help='Train replaced Mixer blockes')
    
    # data parameters
    parser.add_argument("--data-path", default="./data", type=str, help="dataset path")
    parser.add_argument("--data-set", default="CIFAR", choices=["CIFAR", "IMNET", "INAT", "INAT19"],
                        type=str, help="Image Net dataset path")
    parser.add_argument("--nb-classes", default=100, type=int, help="number of classes (default:100)")
    
    # data augment parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # training parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--output-dir', default='/home/xfei22', help='path where to save, empty for no saving')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT",
                        help="Dropout rate (default: 0.)")
    parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT",
                        help="Drop path rate (default: 0.1)")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--train-mode', action='store_true')
    parser.set_defaults(train_mode=True)
    
    # Learning rate schedule parameters
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    
    return parser
    

def load_weight(model, weight):
    if weight.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            weight, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(weight, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
            
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
    # print("Missing keys:", missing_keys)
    # print("Unexpected keys:", unexpected_keys)
    return model

    
def replace_att2mixer(model, repl_blocks, model_name = "",
                      mode = "all", weighted_model = None):
    for blk_index in repl_blocks:
        
        if weighted_model == None:
            if mode == "all":
                mlp_block = MixerBlock(param_dict_all[model_name]["num_patches"], 
                                       param_dict_all[model_name]["token_hid_dim"], 
                                       param_dict_all[model_name]["channels_dim"], 
                                       param_dict_all[model_name]["channels_hid_dim"])
                mlp_block.to("cuda")
                model.blocks[blk_index] = mlp_block
            elif mode == "qkv":
                mlp_block = MixerBlock(param_dict_qkv[model_name]["num_patches"], 
                                       param_dict_qkv[model_name]["token_hid_dim"], 
                                       param_dict_qkv[model_name]["channels_dim"], 
                                       param_dict_qkv[model_name]["channels_hid_dim"])
                mlp_block.to("cuda")
                mlp_block.norm1 = nn.Identity()
                model.blocks[blk_index].attn = mlp_block
            else:
                raise NotImplementedError("Not available replace method")
            
        else:
            model.blocks[blk_index] = weighted_model.blocks[blk_index]

    return model


def cut_extra_layers(model, max_index, depth = 12):
    for index in range(max_index + 1, depth):
        model.blocks[index] = EmptyBlock()
    model.norm = nn.Identity()
    model.fc_norm = nn.Identity()
    model.head_drop = nn.Identity()
    model.head = nn.Identity()
    return model


def set_requires_grad(model, targets, mode, trainable=True):
    target_names = [f"blocks.{target}" for target in targets]
    for name, param in model.named_parameters():
        # print(name)
        if any(target in name for target in target_names):
            if mode == "qkv":
                if "mlp" in name:
                    param.requires_grad = not trainable
                else:
                    param.requires_grad = trainable
            elif mode == "all":
                param.requires_grad = trainable
            else:
                raise NotImplementedError("Not available replace method")
        else:
            param.requires_grad = not trainable


def main(args):
    print(args)

    device = torch.device(args.device)
    
    ########################################
    ### distributed training not implemented
    ########################################
    
    # fix the seed for reproducibility
    seed = args.seed
    # seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    # Load datasets
    print(f"Loading dataset {args.data_set}")
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    ### here: extra data augmentation not implemented, mix-up not implemented
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    # create structure of DeiT
    print(f"Creating DeiT model: {args.d_model}")
    model_deit = create_model(
        args.d_model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )
    model_ori = copy.deepcopy(model_deit)
    print(f"Replacing blocks: {args.replace}")
    model_repl = replace_att2mixer(model=model_deit, repl_blocks=args.replace, 
                                   mode=args.rep_mode, model_name = args.d_model)
    
    if args.eval and not args.train:
        print(f"Evaluation model: {args.eval_model}")
        model = load_weight(model_repl, args.eval_model)
        model.to(device)
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
    elif args.train and not args.eval:
        print(f"Train model: {args.d_model}, target blocks:{args.replace}")
        model = load_weight(model_repl, args.d_weight)
        model_ori = load_weight(model_ori, args.d_weight)
        weighted_model_ori = copy.deepcopy(model_ori)
        partial_model =  cut_extra_layers(model, max(args.replace))
        partial_model_ori = cut_extra_layers(model_ori, max(args.replace))
        partial_model.to(device)
        partial_model_ori.to(device)
        
        set_requires_grad(partial_model, args.replace, args.rep_mode)
        set_requires_grad(partial_model_ori, [], args.rep_mode)
        partial_model.to(device)
        partial_model_ori.to(device)
        
        ### EMA augmentation in training not implemented
        n_parameters = sum(p.numel() for p in partial_model.parameters() if p.requires_grad)
        print('number of trainable params:', n_parameters)
        
        if not args.unscale_lr:
            linear_scaled_lr = args.lr * args.batch_size / 512.0
            args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)
        criterion = CosineSimilarityLoss()
        
        current_time = datetime.datetime.now()
        output_dir = "/home/xfei22/" + \
            current_time.strftime("%Y-%m-%d-%H-%M") + "/"
        args.output_dir = Path(output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        trained_model, trained_model_dict = train_model(args, partial_model, partial_model_ori,
                                                        criterion, optimizer, loss_scaler, lr_scheduler,
                                                        data_loader_train, device, n_parameters)
        trained_model = replace_att2mixer(model=weighted_model_ori, repl_blocks=args.replace, 
                                          weighted_model= trained_model)
        # print(trained_model)
        save_path = args.output_dir / "replaced_model.pth"
        trained_model_dict["model"] = trained_model.state_dict()
        torch.save(trained_model_dict, save_path)
        
    else:
        raise ValueError("Please specify running mode (eval/train).") 
    
    # summary(partial_model, (3, 224, 224))
    # print(model.blocks[1].token_mixing.fc1.weight)
    # print(model.blocks[0].mlp.fc1.weight)
    # print(partial_model.blocks[0].mlp.fc1.weight)
    
    # TODO: take output without change model structure
    # TODO: add distribution learning


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", parents=[get_args_parser()])
    args = parser.parse_args()
    
    deit_model = "deit_tiny_patch16_224"
    deit_weight = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
    repl_index = [11]
    
    args.d_model = deit_model
    args.d_weight = deit_weight
    args.replace = repl_index
    args.rep_mode = "all"
    
    args.train = True
    args.sched = "constant"
        
    main(args)

 
# 
# batch_size = "128"
# epochs = "6"
# model = "deit_base_patch16_224"
# input_size = "224"
# stochastic_depth = "0"
# opt = "sgd"
# weight_decay = "1e-4"
# lr = "0.01"
# random_erase = "0"
# num_workers = "1"
# 
# 
# subprocess.run([
#     "python", "main.py",
#     "--model", model,
#     "--batch-size", batch_size,
#     "--epochs", epochs,
#     "--input-size", input_size,
#     "--drop-path", stochastic_depth,
#     "--opt", opt,
#     "--weight-decay", weight_decay,
#     "--lr", lr,
#     "--reprob", random_erase,
#     "--finetune", fine_tune,
#     "--data-path", data_path,
#     "--output_dir", output_dir,
#     "--num_workers", num_workers
# ])
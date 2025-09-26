import os
import json
from transformers import get_scheduler
import utils.datasets as datasets
from accelerate.logging import get_logger
import models.models as models
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import utils.lr_decay as lrd
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
import argparse
import logging
import torch
import utils.transform as transform


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1) for k in topk]


def eval(accelerator, model, dataloader, logger, info, type="val"):
    model.eval()
    gathered_acc1, gathered_acc5, gathered_losses = [], [], []
    with torch.no_grad():
        criterion_eval = torch.nn.CrossEntropyLoss()
        # switch to evaluation mode
        for step, (samples, target) in enumerate(dataloader):
            # compute output
            with accelerator.autocast():
                output, _ = model(samples)
                # output = model(samples)
                loss = criterion_eval(output, target)

            gathered_acc1_, gathered_acc5_ = accuracy(
                output, target, topk=(1, 5))
            gathered_acc1.append(gathered_acc1_)
            gathered_acc5.append(gathered_acc5_)
            gathered_losses.append(loss)
        gathered_acc1 = torch.cat(gathered_acc1, dim=0)
        gathered_acc5 = torch.cat(gathered_acc5, dim=0)
        gathered_losses = torch.stack(gathered_losses)
        gathered_acc1 = accelerator.gather_for_metrics(gathered_acc1)
        gathered_acc5 = accelerator.gather_for_metrics(gathered_acc5)
        gathered_losses = accelerator.gather(gathered_losses)
        final_acc1 = 100.0 * gathered_acc1.float().mean().item()
        final_acc5 = 500.0 * gathered_acc5.float().mean().item()
        final_loss = gathered_losses.mean().item()
        logger.info(
            f"[{type}] {info} acc1: {final_acc1:.4f} acc5: {final_acc5:.4f} loss: {final_loss:.4f}", main_process_only=True)
    return final_acc1


def get_args_parser():
    parser = argparse.ArgumentParser(
        'MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)

    # Dataset parameters

    parser.add_argument('--classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--eval", action='store_true',
                        help="Whether to evaluate the model after training")
    parser.set_defaults(eval=False)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduling")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Interval for logging training progress")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Interval for saving model checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to save the model checkpoint")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="Minimum learning rate ratio for cosine scheduler")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name to use for training/evaluation")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='Layer-wise learning rate decay from ELECTRA/DeiT (default: 0.75)')
    parser.add_argument('--TR', type=str, default=None,
                        help='Train Ratio for dataset sub-sampling')
    parser.add_argument('--eval_epoch', type=int, default=0,
                        help='epoch for evaluation if different from training epochs')
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Interval for evaluating the model")
    parser.add_argument('--img_size', type=int, default=320,
                        help='Image size (default: 320)')
    parser.add_argument('--lrd', type=str, default='moe_lrd',
                        help='learning rate decay function')
    return parser


def main(args):
    args.lr = args.blr * args.batch_size * args.ngpus / 256
    accelerator = Accelerator(
        mixed_precision=None,
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=True)]
    )
    logger = get_logger(__name__)

    if accelerator.is_main_process:
        os.makedirs(f"{args.log_dir}/{args.dataset}", exist_ok=True)
        log_file_path = f"{args.log_dir}/{args.dataset}/finetune.log"
        if args.eval:
            log_file_path = f"{args.log_dir}/{args.dataset}/eval.log"
        file_handler = logging.FileHandler(log_file_path, mode='a')
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)

    transform_val = transform.__dict__[args.dataset](
        input_size=args.img_size, split="val")

    if not args.eval:
        transform_train = transform.__dict__[args.dataset](
            input_size=args.img_size, aa=args.aa, reprob=args.reprob, remode=args.remode, recount=args.recount, split="train")
        dataset_train = datasets.__dict__[args.dataset](
            args.root_dir, transform_train, "train", tr=args.TR)

        dataset_val = datasets.__dict__[args.dataset](
            args.root_dir, transform_val, "val", tr=args.TR)
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    else:
        dataset_val = datasets.__dict__[args.dataset](
            args.root_dir, transform_val, "test")

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    args.classes = dataset_val.num_classes

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        # print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.classes)

    model = models.__dict__[args.model](
        imgsize=args.img_size,
        num_classes=args.classes,
        # drop_path_rate=args.drop_path,
        # global_pool=args.global_pool,
    )
    config = {
        "model": args.model,
        "lrd": args.lrd,
        "num_classes": args.classes,
        "dataset": args.dataset,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "TR": args.TR,
        "data_info": {
            "train_len": len(dataset_train) if not args.eval else 0,
            "val_len": len(dataset_val),
        },
        "epochs": args.epochs,
        "lr": args.lr,
        "blr": args.blr,
        "layer_decay": args.layer_decay,
        "drop_path": args.drop_path,
        "smoothing": args.smoothing,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
    }
    logger.info(json.dumps(config, indent=4), main_process_only=True)

    if args.finetune:
        checkpoint = torch.load(
            args.finetune, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        msg = model.load_state_dict(checkpoint, strict=False)

        ckpt_pos_embed = checkpoint['pos_embed']
        ckpt_W = int((ckpt_pos_embed.shape[1])**0.5)
        det_W = int((model.pos_embed_det.shape[1])**0.5)
        cls_pos_embed = None
        if ckpt_pos_embed.shape[1] == ckpt_W**2 + 1:
            cls_pos_embed = ckpt_pos_embed[:, :1, :]
            ckpt_pos_embed = ckpt_pos_embed[:, 1:, :]
        if ckpt_pos_embed.shape != model.pos_embed_det.shape and not det_W**2 + 1 == ckpt_W**2 + 1:
            print(
                f"ckpt: {ckpt_pos_embed.shape} det: {model.pos_embed_det.shape}\nPosition embedding shape mismatch, interpolate it.")
            ckpt_pos_embed = ckpt_pos_embed.reshape(
                -1, ckpt_W, ckpt_W, ckpt_pos_embed.shape[-1]).permute(0, 3, 1, 2)
            ckpt_pos_embed = torch.nn.functional.interpolate(
                ckpt_pos_embed, size=(det_W, det_W), mode='bicubic', align_corners=False)

            ckpt_pos_embed = ckpt_pos_embed.permute(
                0, 2, 3, 1).flatten(1, 2)
        else:
            print("Position embedding successfully loaded.")
        if model.pos_embed_det.shape[1] == det_W**2 + 1:
            if cls_pos_embed is not None:
                ckpt_pos_embed = torch.cat(
                    (cls_pos_embed, ckpt_pos_embed), dim=1)
                model.pos_embed_det.data.copy_(ckpt_pos_embed)
            else:
                model.pos_embed_det[:, 1:, :].data.copy_(ckpt_pos_embed)
        else:
            model.pos_embed_det.data.copy_(ckpt_pos_embed)
        logger.info(
            f"missing keys: {msg.missing_keys}\nunexpected keys: {msg.unexpected_keys}")

    if not args.eval:
        optimizer = torch.optim.AdamW(lrd.__dict__[args.lrd](
            model, args.lr, 0.05, layer_decay=args.layer_decay), lr=args.lr)

        dataloader_train, dataloader_val, model, optimizer = accelerator.prepare(
            dataloader_train, dataloader_val, model, optimizer)

        min_lr, max_lr = 10.0, 0.0

        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        logger.info(f"layer decay optimizer: min_lr: {min_lr} max_lr: {max_lr}",
                    main_process_only=True)
    else:
        dataloader_val, model = accelerator.prepare(dataloader_val, model)

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        accelerator.load_state(args.resume)
        args.start_epoch = int(args.resume.split("-")[-1].strip())

    if not args.eval:
        lr_scheduler = get_scheduler(
            "cosine_with_min_lr",
            optimizer=optimizer,
            num_warmup_steps=int(args.warmup_ratio *
                                 args.epochs * len(dataloader_train)),
            num_training_steps=args.epochs * len(dataloader_train),
            # minimum learning rate ratio
            scheduler_specific_kwargs={"min_lr_rate": args.min_lr_ratio}
        )

        for _ in range(args.start_epoch * len(dataloader_train)):
            lr_scheduler.step()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "model": args.model, }
    if args.eval:
        _ = eval(accelerator=accelerator, model=model,
                 dataloader=dataloader_val, logger=logger, info="")
        return
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        model.train(True)
        min_lr, max_lr = 10.0, 0.0
        gathered_acc1, gathered_acc5, gathered_loss = [], [], []
        info = f"epoch: [{epoch}/{args.epochs}]"

        for step, (samples, target) in enumerate(dataloader_train):
            # print(targets.shape)
            optimizer.zero_grad()
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, target)
            with accelerator.autocast():
                outputs, aux_loss = model(samples)
                
                loss = criterion(outputs, targets) + aux_loss.sum() * 0.000005
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                # print(outputs.shape, target.shape)
                gathered_acc1_, gathered_acc5_ = accuracy(
                    outputs, target, topk=(1, 5))
                gathered_acc1.append(gathered_acc1_)
                gathered_acc5.append(gathered_acc5_)
                gathered_loss.append(loss)
                if (step+1) % args.log_interval == 0:
                    logger.info(f"{info} [{step+1}/{len(dataloader_train)}] acc1: {100.0*gathered_acc1_.float().mean().item():.4f} acc5: {500.0*gathered_acc5_.float().mean().item():.4f} loss: {loss.item():.4f}({sum(gathered_loss).item()/len(gathered_loss):.4f})", main_process_only=True)
        with torch.no_grad():
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
            gathered_acc1 = torch.cat(gathered_acc1, dim=0)
            gathered_acc5 = torch.cat(gathered_acc5, dim=0)
            gathered_loss = torch.stack(gathered_loss)
            gathered_acc1 = accelerator.gather_for_metrics(gathered_acc1)
            gathered_acc5 = accelerator.gather_for_metrics(gathered_acc5)
            gathered_loss = accelerator.gather(gathered_loss)
            # gathered_acc1 = gathered_acc1[:dataset_train_len]
            # gathered_acc5 = gathered_acc5[:dataset_train_len*5]
            gathered_loss = gathered_loss.mean().item()
            gathered_acc1 = 100.0 * gathered_acc1.float().mean().item()
            gathered_acc5 = 500.0 * gathered_acc5.float().mean().item()

            logger.info(
                f"{info} acc1: {gathered_acc1:.4f} acc5: {gathered_acc5:.4f} loss: {gathered_loss:.4f} min_lr: {min_lr} max_lr: {max_lr}", main_process_only=True)
            if epoch >= args.eval_epoch or (epoch + 1) % args.eval_interval == 0:
                acc = eval(accelerator=accelerator, model=model,
                           dataloader=dataloader_val, logger=logger, info=info)

                if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs or (acc >= max_accuracy and acc > 92.0):
                    max_accuracy = max(acc, max_accuracy)
                    os.makedirs(f"{args.checkpoint}/{args.dataset}",
                                exist_ok=True)
                    save_dir = f"{args.checkpoint}/{args.dataset}/checkpoint-{epoch + 1}"
                    accelerator.save_state(save_dir)
                    logger.info(
                        f"Saved checkpoint to {save_dir}", main_process_only=True)
    accelerator.end_training()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

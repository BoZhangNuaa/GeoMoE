import argparse
import torch
from transformers import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
import models.models as models
import utils.dataset as dataset
import json
from accelerate.logging import get_logger
import logging
import os
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser('TCMAE pre-training', add_help=False)
    parser.add_argument("--ngpus", default=1, type=int,
                        help="Number of GPUs to use")
    parser.add_argument("--batch_size", default=256,
                        type=int, help="Batch size per GPU")
    parser.add_argument("--blr", type=float, default=1e-4,
                        help="Base learning rate for the optimizer")
    parser.add_argument("--start_epoch", default=0, type=int,
                        help="Starting epoch for training")
    parser.add_argument("--epochs", default=200, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--warmup_ratio", default=0.1,
                        type=float, help="Warmup ratio for learning rate")
    parser.add_argument("--min_lr_ratio", default=0.,
                        type=float, help="Minimum learning rate ratio")
    parser.add_argument("--mask_ratio", default=0.75,
                        type=float, help="Mask ratio for MAE")
    parser.add_argument("--model", default="mae_vit_large_patch16",
                        type=str, help="Model architecture")
    parser.add_argument("--resume", default="", type=str,
                        help="Path to latest checkpoint (default: none)")
    parser.add_argument("--checkpoints", default="save/pretrain",
                        type=str, help="Directory for saving checkpoints")
    parser.add_argument("--log_dir", default="logs",
                        type=str, help="Directory for logging")
    parser.add_argument("--log_interval", default=50, type=int,
                        help="Interval for logging training status")
    parser.add_argument("--save_interval", default=1, type=int,
                        help="Interval for saving checkpoints")
    parser.add_argument("--input_size", default=224, type=int,
                        help="Input image size for the model")
    parser.add_argument("--norm_pix_loss", action="store_true",
                        help="Use normalized pixel-wise loss")
    parser.add_argument("--num_workers", default=16, type=int,
                        help="Number of data loading workers")
    parser.add_argument("--datasets", default="IN1K",
                        type=str, help="Dataset to use for training")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--build_ratio", default=0.5,type=float,)
    parser.add_argument("--dataset_path", default='/datasets/OpticalRS-4M',
                        type=str, help="Path to the dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    args.blr = args.blr * args.build_ratio / args.mask_ratio
    args.lr = args.blr * args.ngpus * args.batch_size / 256

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs]
    )

    logger = get_logger(__name__)
    if accelerator.is_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file_path = f"{args.log_dir}/train.log"
        file_handler = logging.FileHandler(log_file_path, mode='a')
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.ERROR)

    #with open(args.data_config, "r") as f:
    #    data_config = json.load(f)
    config = {
        "model": args.model,
        "datasets": args.datasets,
        "mask_ratio": args.mask_ratio,
        "build_ratio": args.build_ratio,
        "norm_pix_loss": args.norm_pix_loss,
        "base_lr": args.blr,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
    }
    logger.info(json.dumps(config, indent=4), main_process_only=True)

    model = models.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(
            0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_path = args.dataset_path
    train_dataset = dataset.AllDataset(dataset_path, train=True, transform=transform_train)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    logger.info(json.dumps(
        {args.datasets: len(train_dataset)}, indent=4), main_process_only=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    model, optimizer, data_loader_train = accelerator.prepare(
        model, optimizer, data_loader_train)
    data_len = len(data_loader_train)

    lr_scheduler = get_scheduler(
        "cosine_with_min_lr",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio *
                             args.epochs * data_len),
        num_training_steps=args.epochs * data_len,
        scheduler_specific_kwargs={"min_lr_rate": args.min_lr_ratio}
    )

    if args.resume:
        logger.info(f"Resuming from {args.resume}", main_process_only=True)
        accelerator.load_state(args.resume)
        args.start_epoch = int(args.resume.split("-")[-1].strip())
    for _ in range(args.start_epoch * data_len):
        lr_scheduler.step()

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        info = f"epoch: [{epoch}/{args.epochs}]"
        loss_epoch = 0.0
        metrics = {
            "epoch": epoch + 1,
            "aux_loss_sum": torch.zeros(len(model.module.blocks3)),
            "bias": [],
            "expert_loads": torch.zeros((len(model.module.blocks3), 23))
        }
        bias_change = 0.001 if epoch < 800 else 0.
        for step, images in enumerate(data_loader_train):
            with accelerator.autocast():
                loss, aux_loss, expert_loads = model(
                    images, mask_ratio=args.mask_ratio, kept_mask_ratio=args.build_ratio)
            loss = loss + aux_loss.sum() * 0.0001
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            with torch.no_grad():
                expert_loads = accelerator.gather(
                    expert_loads.unsqueeze(0)).sum(dim=0)
                expert_loads_mean = expert_loads.float().mean(dim=-1)
                for i in range(len(model.module.blocks3)):
                    model.module.blocks3[i].mlp.bias[expert_loads[i]
                                                        < expert_loads_mean[i]] += bias_change
                    model.module.blocks3[i].mlp.bias[expert_loads[i]
                                                        > expert_loads_mean[i]] -= bias_change
                metrics["expert_loads"] += expert_loads.detach().cpu()
            if accelerator.is_main_process:
                loss_epoch += loss.item()
                metrics["aux_loss_sum"] += aux_loss.detach().cpu() * 0.0001
            if (step+1) % args.log_interval == 0:
                logger.info(
                    f"{info} step: [{step+1}/{data_len}] loss: {loss.item():.4f}({(loss_epoch/(step+1)):.4f}) lr: {optimizer.param_groups[0]["lr"]}", main_process_only=True)
            # break

        logger.info(
            f"{info} loss: {loss.item():.4f}({(loss_epoch/data_len):.4f}) lr: {optimizer.param_groups[0]["lr"]}", main_process_only=True)
        if accelerator.is_main_process:
            metrics["aux_loss_sum"] = (
                metrics["aux_loss_sum"] / data_len).tolist()
            metrics["expert_loads"] = metrics["expert_loads"].tolist()
            for i in range(len(model.module.blocks3)):
                metrics["bias"].append(
                    model.module.blocks3[i].mlp.bias.cpu())
            metrics["bias"] = torch.stack(metrics["bias"]).tolist()
            os.makedirs(f"{args.log_dir}/metrics", exist_ok=True)
            with open(f"{args.log_dir}/metrics/metrics-{epoch}.json", "w") as f:
                f.write(json.dumps(metrics, indent=4))
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            os.makedirs(args.checkpoints, exist_ok=True)
            save_dir = f"{args.checkpoints}/checkpoint-{epoch + 1}"
            accelerator.save_state(save_dir)
            logger.info(
                f"Saved checkpoint to {save_dir}", main_process_only=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()

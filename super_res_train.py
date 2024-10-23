"""
Train a super-resolution model.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import argparse

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import blobfile as bf

def main():
    args = create_argparser().parse_args()
    #args.use_fp16=True

    dist_util.setup_dist()
    logger.configure(dir='./experiments/'+args.save_forder)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys()),data_dir=args.data_dir+'.'+args.cloudmodel
    )
    model.to(dist_util.dev())
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    logger.log(args.model_path)
    model_path = args.model_path
    if(len(sorted(bf.listdir(model_path)))):
        entry = sorted(bf.listdir(model_path))[-1]
        full_path = bf.join(model_path, entry)
        logger.log(model_path)
        model.load_state_dict(
            dist_util.load_state_dict(full_path, map_location="cpu"),strict=False
        )
        model.to(dist_util.dev())
        logger.log("load model from "+entry)

    logger.log("creating data loader...")
    data_dir = "./data/"+args.data_dir+"/train/cloud"
    data = load_superres_data(
        data_dir,
        args.batch_size,
        large_size=args.image_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
    )
    logger.log("lr:" +str(args.lr))
    logger.log("image size:" +str(args.image_size))

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    for large_batch, model_kwargs in data:
        # large_batch = large_batch['label']
        # model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        model_kwargs["low_res"] = large_batch['cloud']
        model_kwargs["previous"] = large_batch['previous']
        large_batch = large_batch['label']
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir="RICE2",  #换数据集
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=64,   #根据image_size改 64
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=1000,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        save_forder='train_model',
        model_path="./pre_train",
        image_size=64,
        cloudmodel = 'mn'
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import os
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from PIL import Image
from evaluations.evaluator import Measure
from torch.utils.data import DataLoader, Dataset
from guided_diffusion.image_datasets import ImageDataset, _list_image_files_recursively
import time
import torch.optim as optim

def save_image(name,img,path):
    img = ((img+1)*127.5).clamp(0, 255).to(th.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous().cpu()[0]
    Image.fromarray(np.uint8(img)).save(path+name+'.png')

def save_imagenir(name,img,path):
    img = ((img+1)*127.5).clamp(0, 255).to(th.uint8)
    img = img.permute(0, 2, 3, 1)[0]
    nir = img[:,:,3]
    nir = nir.expand(3,256,256)
    nir = nir.permute( 1, 2, 0)
    img = img.contiguous().cpu()
    nir = nir.contiguous().cpu()
    Image.fromarray(np.uint8(img[:,:,0:3])).save(path+name+'.png')
    Image.fromarray(np.uint8(nir)).save(path+name+'nir.png')
from mpi4py import MPI
import torch.nn.functional as F
#from guided_diffusion.diff.prior import general_cond_fncloud

def get_image_arr(dataset):
    if(dataset=='RICE1'):
        return ['0','105','143','368','425','458','495']
    elif(dataset=='RICE2'):
        return ['421']
    elif(dataset =='T-Cloud'):
        return ['278','142','162','449','930','1261','1652']
    elif(dataset=='CUHK-CR1' or dataset=='CUHK-CR2'):
        return ['4','5','7','8','36','37','65']
    else:
        return []
    
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
def sample():
    args = create_argparser().parse_args()

    #dist_util.setup_dist()
    logger.configure(dir='./experiments/'+args.save_forder)

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys()),data_dir=args.base_samples+'.'+args.cloudmodel
    )
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    logger.log(args.model_path)
    entry = sorted(bf.listdir(args.model_path))[0]
    full_path = bf.join(args.model_path, entry)
    logger.log("load model from "+entry)
    model.load_state_dict(
        dist_util.load_state_dict(full_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    base_samples = "./data/"+args.base_samples+"/test/cloud" # The path of the dataset
    logger.log("loading data...")
    data = load_superres_data(
        base_samples,
        args.batch_size,
        large_size=args.image_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
    )
    logger.log("Py:" + str(args.py))
    logger.log("creating samples...")
    measure = Measure()
    for i,(batch,out_dict) in enumerate(data):
        cloud = {'low_res':batch['cloud'].to(dist_util.dev()), 'previous':batch['previous'].to(dist_util.dev()) }
        if args.timestep_respacing.startswith("ddim"):
            if(args.py):
                sample = diffusion.ddim_pyramid_sample(
                    model,
                    cloud['low_res'],
                    clip_denoised=args.clip_denoised,
                    model_kwargs=cloud,
                )
            else:
                sample = diffusion.ddim_sample_loop(
                    model,
                    (args.batch_size, cloud['low_res'].shape[1], args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=cloud,

                )
        else:
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, cloud['low_res'].shape[1], args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=cloud,
            )
        

        label = batch['label'].to(dist_util.dev())
        measure.measure(sample[0],label[0])
        if(batch['filename'][0] in get_image_arr(args.base_samples)):
            if('My' in args.base_samples):
                save_imagenir(batch['filename'][0]+'LR',cloud['low_res'],'./experiments/'+args.save_forder+'/image/')
                save_imagenir(batch['filename'][0]+'HR',label,'./experiments/'+args.save_forder+'/image/')
                save_imagenir(batch['filename'][0]+'SR',sample,'./experiments/'+args.save_forder+'/image/')
            else:
                save_image(batch['filename'][0]+'LR',cloud['low_res'],'./experiments/'+args.save_forder+'/image/')
                save_image(batch['filename'][0]+'HR',label,'./experiments/'+args.save_forder+'/image/')
                save_image(batch['filename'][0]+'SR',sample,'./experiments/'+args.save_forder+'/image/')
    measure.caculate_all()
    logger.log("sampling complete")


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    all_files = _list_image_files_recursively(data_dir)
    dataset = ImageDataset(
            large_size,
            all_files,
            classes=None,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            random_crop=False,
            random_flip=False,
        )
    loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    return loader

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=1,
        use_ddim=True,
        base_samples="RICE2",  #choose from RICE2 CUHK-CR1 CUHK-CR2
        model_path="./pre_train",
        save_forder='test_model',
        timestep_respacing='ddim10',
        image_size=256,
        cloudmodel = 'mn', #choose from mdsa mn
        py=False,
    )
    n=sr_model_and_diffusion_defaults()
    n.update(defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, n)
    return parser


if __name__ == "__main__":
    sample()

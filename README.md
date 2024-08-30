# Diffusion-Enhancement-for-CR

🔥 Congradualation to my collaborators and myself! This paper has been accepted by IEEE Transactions on Geoscience and Remote Sensing! It's a new start!

The code for [Diffusion Enhancement for Cloud Removal in Ultra-Resolution Remote Sensing Imagery](https://ieeexplore.ieee.org/abstract/document/10552304/), which is based on [ADM](https://github.com/openai/guided-diffusion). 

Several conventional CR models could refer to [https://github.com/littlebeen/Cloud-removal-model-collection](https://github.com/littlebeen/Cloud-removal-model-collection)!

# Usage

Download the pretain e2e model (mdsa/mn as your need) and put it into guided_diffusion/couldnet/mdsa/pretrain or guided_diffusion/couldnet/mn/pretrain. The model could be found at https://pan.baidu.com/s/1lyKRG67AxM5SZEj3mhTKWA code:bean

 You could choose the appropriate pretrain model as your need (Which model You want to train or test based on? memorynet or mdsa?). Must be done before training and testing

**Train**
1. Pure diffusion. respace.py:gaussian_diffusion; unet.py: UnetModel
2. Locked diffusion + Trained WA :gaussian_diffusion_enhance; unet.py: UnetModel256; locked in train_util.py line74
3. ALL-in change train_util.py line74

```python super_res_train.py```

**Test**

1. Put the pre-train model into 'pre_train'

```python super_res_sample.py```

**Weight**

Our pre-train models on RICE2 with mn and mdsa are uploaded. [https://pan.baidu.com/s/1SvnPL7HRKqSK0zDixYpHow](https://pan.baidu.com/s/1SvnPL7HRKqSK0zDixYpHow) code：bean

# CUHK-CR

A novel multispectral Cloud Removal dataset

Download link: [https://pan.baidu.com/s/1z2SgORYz5_t94kya8CeqiQ](https://pan.baidu.com/s/12yNH5eowjGA1fFM5sXUzBw) code:bean

-CUHK-CR1 (The RGB images for thin dataset CUHK-CR1)

-CUHK-CR2 (The RGB images for think dataset CUHK-CR2)

-nir (the near-infrared images for CUHK-CR1 and CUHK-CR2)

If you need image with 4 bands (RGB + nir), you could load the images in the RGB dataset and the nir dataset and combine the 4 channels together. 


# Cite

If this project is useful to you, please cite this paper :)

```
@article{sui2024diffusion,
  author={Sui, Jialu and Ma, Yiyang and Yang, Wenhan and Zhang, Xiaokang and Pun, Man-On and Liu, Jiaying},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Diffusion Enhancement for Cloud Removal in Ultra-Resolution Remote Sensing Imagery}, 
  year={2024},
  volume={62},
  pages={1-14},
  doi={10.1109/TGRS.2024.3411671}}
```
If you have any question, be free to contact with me!

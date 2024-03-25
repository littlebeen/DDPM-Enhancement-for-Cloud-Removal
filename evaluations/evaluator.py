import torch
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import lpips
import numpy as np
from guided_diffusion import logger

loss_fn = lpips.LPIPS(net='alex', version=0.1)

def caculate_lpips(img0,img1):
    im1=np.copy(img0.cpu().numpy())
    im2=np.copy(img1.cpu().numpy())
    im1=torch.from_numpy(im1.astype(np.float32))
    im2 = torch.from_numpy(im2.astype(np.float32))
    im1.unsqueeze_(0)
    im2.unsqueeze_(0)
    current_lpips_distance  = loss_fn.forward(im1, im2)
    return current_lpips_distance 

class Measure:
    def __init__(self):
       self.criterionMSE=torch.nn.MSELoss()
       self.psnr=0
       self.ssim=0
       self.sample=0
       self.lpips=0
       self.allpsnr=[]
       self.alllpips=[]
       self.allssim=[]
    
    def measure(self, imgA, imgB):
        """

        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [-1,1]  #up
            imgB: [C, H, W] uint8 or torch.FloatTensor [-1,1]  #hr
            img_lr: [C, H, W] uint8  or torch.FloatTensor [-1,1]
            sr_scale:

        Returns: dict of metrics

        """
        imgA = ((imgA+1)*127.5).clamp(0, 255).to(torch.uint8)
        imgB = ((imgB+1)*127.5).clamp(0, 255).to(torch.uint8)
        psnr = self.caculate_psnr(imgA, imgB)
        ssim=0
        lpips=0
        if(imgA.shape[0]==4):
            for i in range(imgA.shape[0]):
                imA = imgA[i]
                imA = imA.expand(3,256,256)
                imB = imgB[i]
                imB = imB.expand(3,256,256)
                ssim1 = self.caculate_ssim(imA, imB)
                lpips1 = caculate_lpips(imA, imB)
                ssim+=ssim1
                lpips+=lpips1
            ssim=ssim/imgA.shape[0]
            lpips=lpips/imgA.shape[0]
        else:
            ssim = self.caculate_ssim(imgA, imgB)
            lpips = caculate_lpips(imgA, imgB)
        self.allpsnr.append(psnr)
        self.allssim.append(ssim)
        self.alllpips.append(lpips)
        self.psnr=self.psnr+psnr
        self.ssim=self.ssim+ssim
        self.lpips=self.lpips+lpips
        self.sample+=1


    def caculate_ssim(self, imgA, imgB):
        imgA1 = np.tensordot(imgA.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        imgB1 = np.tensordot(imgB.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        score = SSIM(imgA1, imgB1, data_range=255)
        return score

    def caculate_psnr(self, imgA, imgB):
        imgA1 = imgA.cpu().numpy().transpose(1, 2, 0)
        imgB1 = imgB.cpu().numpy().transpose(1, 2, 0)
        psnr = PSNR(imgA1, imgB1, data_range=255)
        return psnr
    
    def caculate_all(self):
        logger.log("psnr: {:.3f}".format(self.psnr/self.sample))
        logger.log("ssim: {:.4f}".format(self.ssim/self.sample))
        logger.log("lpips: {:.4f}".format(self.lpips.item()/self.sample))

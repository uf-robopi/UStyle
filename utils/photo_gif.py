"""
Copyright (C) 2018 NVIDIA Corporation.    All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from PIL import Image
import numpy as np
import cv2
from cv2.ximgproc import guidedFilter
import torch

class GIFSmoothing():
        
    def __init__(self, r, eps):
        super(GIFSmoothing, self).__init__()
        self.r = r
        self.eps = eps

    def process(self, initImg, contentImg):
        return self.process_opencv(initImg, contentImg)
        
    def process_opencv(self, initImg, contentImg):
        '''
        :param initImg: intermediate output. Can be a file path, a PIL Image, or a Torch Tensor.
        :param contentImg: content image. Can be a file path, a PIL Image, or a Torch Tensor.
        :return: stylized output image as a PIL Image.
        '''
        # --- Process initImg ---
        if isinstance(initImg, torch.Tensor):
            # Remove the batch dimension and rearrange from (1, 3, H, W) to (H, W, 3)
            init_img = initImg.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            # Multiply by 255 (assuming the tensor is in the [0,1] range) and convert to uint8
            init_img = (init_img * 255).astype(np.uint8)
            # Reverse channels from RGB to BGR for OpenCV and force contiguous memory
            init_img = np.ascontiguousarray(init_img[:, :, ::-1])
        elif type(initImg) == str:
            init_img = cv2.imread(initImg)
        else:
            # Assume initImg is a PIL Image or a NumPy array in RGB order.
            init_img = np.array(initImg)
            # If the image appears to be normalized (values in [0,1]), convert to 0-255 range.
            if init_img.max() <= 1.0:
                init_img = (init_img * 255).astype(np.uint8)
            else:
                init_img = init_img.astype(np.uint8)
            # Reverse channels (RGB -> BGR) and copy to ensure positive strides.
            init_img = np.ascontiguousarray(init_img[:, :, ::-1])
    
        # --- Process contentImg ---
        if isinstance(contentImg, torch.Tensor):
            cont_img = contentImg.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            cont_img = (cont_img * 255).astype(np.uint8)
            cont_img = np.ascontiguousarray(cont_img[:, :, ::-1])
        elif type(contentImg) == str:
            cont_img = cv2.imread(contentImg)
        else:
            cont_img = np.array(contentImg)
            if cont_img.max() <= 1.0:
                cont_img = (cont_img * 255).astype(np.uint8)
            else:
                cont_img = cont_img.astype(np.uint8)
            cont_img = np.ascontiguousarray(cont_img[:, :, ::-1])
    
        # Ensure both images have the same shape
        if init_img.shape != cont_img.shape:
            cont_img = cv2.resize(cont_img, (init_img.shape[1], init_img.shape[0]))
    
        output_img = guidedFilter(guide=cont_img, src=init_img, radius=self.r, eps=self.eps)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        output_img = Image.fromarray(output_img)
        return output_img

        


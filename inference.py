# this is a demo script for running inference on a single image using the bedroom model
import json
import cv2
import PIL
import pdb
import torch
from PIL import Image
import numpy as np
from E2EVE.main_functions import instantiate_from_config
from E2EVE.data.E2EVE_data import DemoData

if __name__ == '__main__':
    
    # # remove this
    # image = cv2.imread("../E2EVE_meta/LSUN_meta/0129.jpg")
    # DataProcessor = DemoData()
    # source_image = DataProcessor._source_image_padder_and_resize(image)
    # source_image = cv2.rectangle(source_image,(70,20),(200,150),(255,0,0),10)
    # mask = np.zeros_like(source_image)
    # cv2.imwrite('temp.jpg',source_image)

    # mask = np.zeros_like(source_image)
    # mask[20:150,70:200,:] = 255
    # cv2.imwrite('temp.jpg',mask)
    # pdb.set_trace()
    
    # (1) load the model
    
    # model_config = "configs/LSUN_bedroom/inference_block_edit.json"
    
    # with open(model_config, "rb") as f:
    #     data = json.load(f)

    # model = instantiate_from_config(data['model']).cuda()
    # model.eval()
    
    # (2) load the images
    
    source_image_path = "../E2EVE_meta/LSUN_meta/source.jpg"
    driver_image_path = "../E2EVE_meta/LSUN_meta/driver.jpg"
    mask_image_path = "../E2EVE_meta/LSUN_meta/mask.jpg"
    
    with open(source_image_path, "rb") as fid:
        source_image = PIL.Image.open(fid)
        if source_image.mode != 'RGB':
            source_image = source_image.convert('RGB')
        source_image = np.array(source_image)
            
    with open(driver_image_path, "rb") as fid:
        driver_image = PIL.Image.open(fid)
        if driver_image.mode != 'RGB':
            driver_image = driver_image.convert('RGB')
        driver_image = np.array(driver_image)
            
    with open(mask_image_path, "rb") as fid:
        mask_image = PIL.Image.open(fid)
        if mask_image.mode != 'RGB':
            mask_image = mask_image.convert('RGB')
        mask_image = np.array(mask_image)
    
    DataProcessor = DemoData()
    
    # pad and reshape the source and drivers
    source_image = DataProcessor._source_image_padder_and_resize(source_image)
    driver_image = DataProcessor._driver_resize(driver_image)
    
    batch, mask, source, masked_source, target_in =  DataProcessor._load_fixed_data(source_image, driver_image, mask_image)
    
    pdb.set_trace()
    
    # (3) run some stuff
    
    images = model.log_images(batch,do_not_remake=True, only_deterministic=False)

    images["samples_det"] = torch.clamp(images['samples_det'], -1., 1.).detach().cpu()
    image = (images["samples_det"]+1.0)/2.0
    image = image[0].transpose(0,1).transpose(1,2).squeeze(-1).cpu().detach().numpy()
    image_det = (image*255).astype(np.uint8)
    
    images["samples"] = torch.clamp(images['samples'], -1., 1.).detach().cpu()
    image = (images["samples"]+1.0)/2.0
    image = image[0].transpose(0,1).transpose(1,2).squeeze(-1).cpu().detach().numpy()
    image = (image*255).astype(np.uint8)
    
    Image.fromarray(image).save('temp.jpg')
    Image.fromarray(image_det).save('temp_det.jpg')
    
    pdb.set_trace()

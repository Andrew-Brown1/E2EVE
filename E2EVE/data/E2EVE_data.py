# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from torch.utils.data import Dataset
from utils.utils import auto_init_args
import torchvision.transforms as transforms
import os 
import random
import pickle
import PIL
from PIL import Image
from scipy.special import binom
import numpy as np 
import cv2

class E2EVE_dataset_base(Dataset):
    def __init__(self,
                 source_image_size,
                 driver_image_size, 
                 image_scale_transform):
        super().__init__()
        auto_init_args(self)
        
        # transforms 
        self.source_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.source_image_size, scale=(self.image_scale_transform,1), ratio=(1,1)),
                    transforms.RandomHorizontalFlip(),
                ]
            )

        self.driver_transforms =  transforms.Compose([
            transforms.RandomResizedCrop(self.driver_image_size,
                                scale=(1, 1.),
                                ratio=(1., 1.)),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def _array_image(self, image):
        # from PIL to numpy array and normalise

        image = np.array(image).astype(np.uint8)
        image = (image/127.5 - 1.0).astype(np.float32)

        return image
    
    def _sample_R_width(self):
        # sample the width of the masked region R
        return random.uniform(0.4, 0.6)

    def _get_region_coordinates(self, centre_x, centre_y, region_width):
        # return the coordinates of a square region, with provided centre and width (within image)
        return [
            int(max(centre_x-(self.source_image_size*(region_width/2)),0)),
            int(min(centre_x+self.source_image_size*(region_width/2),self.source_image_size)),
            int(max(centre_y-(self.source_image_size*(region_width/2)),0)),
            int(min(centre_y+self.source_image_size*(region_width/2),self.source_image_size))  
        ]
        
    def _mask_image(self, image, masked_region_coors, random_mask):
        # mask a region in the input image, return PIL image
        masked_image = np.array(image).astype(np.uint8)
        if self.mask_method == 'block':
            masked_image[masked_region_coors[2]:masked_region_coors[3],masked_region_coors[0]:masked_region_coors[1],:] = 0
        elif self.mask_method == 'random_mask':
            masked_image = (random_mask==0)*masked_image
        return Image.fromarray(masked_image)
        
    def _driver_position_augment(self, driver_image_coors ,masked_region_coors):
        # move the position of the masked region (while keeping the driver image within)
        
        masked_region_width = masked_region_coors[1] - masked_region_coors[0]
        masked_region_height = masked_region_coors[3] - masked_region_coors[2]
        x_2_give = masked_region_coors[1] - driver_image_coors[1]
        y_2_give = masked_region_coors[3] - driver_image_coors[3]

        min_x1 = max(0,masked_region_coors[0]-x_2_give)
        max_x1 = min(driver_image_coors[0], self.source_image_size-masked_region_width)

        min_y1 = max(0,masked_region_coors[2]-y_2_give)
        max_y1 = min(driver_image_coors[2], self.source_image_size-masked_region_height)

        jittered_x1 = random.uniform(min_x1,max_x1)
        jittered_x2 = jittered_x1 + masked_region_width

        jittered_y1 = random.uniform(min_y1,max_y1)
        jittered_y2 = jittered_y1 + masked_region_height

        return [int(jittered_x1), int(jittered_x2), int(jittered_y1), int(jittered_y2)]
    
    def _get_driver_image(self,source_image):
        # sample the position of the driver image from within the source image

        driver_image_width = int(self.source_image_size*random.uniform(0.2,0.3))
        
        y1_position = int(max(random.uniform(0, self.source_image_size-driver_image_width),0))
        y2_position = int(min((y1_position+driver_image_width),self.source_image_size))
        
        x1_position = int(max(random.uniform(0, self.source_image_size-driver_image_width),0))
        x2_position = int(min((y1_position+driver_image_width),self.source_image_size))
        
        return [x1_position,x2_position,y1_position,y2_position]
    
    def _get_mask_spatial_extent(self, mask):
        """ this gets the spatial extent of the mask"""

        if np.sum(mask) == 0:
            return [96,160,96,160]
        # iterate from above and below, left, and right, looking for first row/column where sum > 1
        for row_ind in range(0,mask.shape[0]):
            if np.sum(mask[row_ind,:]) > 0:
                break

        y1 = row_ind

        for row_ind in range(0,mask.shape[0]):
            if np.sum(mask[-1-row_ind,:]) > 0:
                break

        y2 = mask.shape[0] - 1 - row_ind

        for col_ind in range(0,mask.shape[0]):
            if np.sum(mask[:,col_ind]) > 0:
                break

        x1 = col_ind

        for col_ind in range(0,mask.shape[0]):
            if np.sum(mask[:,-1-col_ind]) > 0:
                break

        x2 = mask.shape[0] - 1 - col_ind

        if (y2-y1) < 2 or (x2-x1) < 2:
            raise Exception('error')
            None

        return [y1, y2, x1, x2]
        

class E2EVE_train_dataset(E2EVE_dataset_base):
    
    def __init__(self, 
                 source_image_size=256, 
                 driver_image_size=64, 
                 alpha_min=0.6,
                 alpha_max=0.6,
                 pos_augment=False, 
                 image_scale_transform=0.75, 
                 image_list='',
                 path_to_images='',
                 mask_method='block',
                 VQGAN_driver=False):
        
        super().__init__(source_image_size, driver_image_size, image_scale_transform)
        auto_init_args(self)
        
        # read the dataset paths
        with open(self.image_list, "r") as f:
            self.dataset = f.read().splitlines()
                        
        for i in range(len(self.dataset)):
            self.dataset[i] = os.path.join(self.path_to_images,self.dataset[i])
        
        if self.mask_method == 'random':
            # pos_augment is not designed to be used alongside random masking
            assert(not self.pos_augment)
            
        self.random_masker = Random_Masker_E2EVE()
            
            
    def __getitem__(self, i):
        
        image_path = self.dataset[i]
        
        # read the image 
        with open(image_path, "rb") as fid:
            source_image = PIL.Image.open(fid)
            if source_image.mode is not 'RGB':
                source_image = source_image.convert('RGB')
            source_image = self.source_transforms(source_image)
            
        # get the initial driver image location (where the sampling starts)
        coors = self._get_driver_image(source_image)
        
        # -------------------------------
        # sample the masked source region
        # --------------------------------
        
        # sample the width of the masked region, R
        masked_region_width = self._sample_R_width()
        
        # find the coordinates of the masked region, centred on the driver image
        centre_x =  (list(coors)[0] + list(coors)[1]) / 2
        centre_y = (list(coors)[2] + list(coors)[3]) / 2
        masked_region_coors = self._get_region_coordinates(centre_x, centre_y, masked_region_width)
        
        # -------------------------------
        # sample the driver image region, given the masked region
        # --------------------------------
        
        # find the width of the driver image (given the input alpha parameters)
        driver_image_width = masked_region_width*(random.uniform(self.alpha_min,self.alpha_max))
        
        # find the coordinates of the driver image
        if self.mask_method == 'block':
            # if block masking used - sample a square region
            driver_image_coors = self._get_region_coordinates(centre_x, centre_y, driver_image_width)
            random_mask=0
        elif self.mask_method == 'random_mask':
            # if random masking used - sample the random mask and a driver region from within that mask
            random_mask, driver_image_coors, _, _ = self.random_masker._get_mask_and_box()
            masked_region_coors = self._get_mask_spatial_extent(random_mask)
            masked_region_width = masked_region_coors[1] - masked_region_coors[0]
        
        
        if self.pos_augment:
            # currently the driver image is centred within the masked region. If pos_augment, 
            # then move the masked region (while keeping the driver image within)
            masked_region_coors = self._driver_position_augment(driver_image_coors,masked_region_coors)
            
        # -------------------------------
        # prepare the driver image 
        # --------------------------------
        driver_image = Image.fromarray(np.array(source_image).astype(np.uint8)[driver_image_coors[2]:driver_image_coors[3],driver_image_coors[0]:driver_image_coors[1],:])

        driver_image = np.expand_dims(self._array_image(self.driver_transforms(driver_image)),0)
        
        # -------------------------------
        # prepare the masked source image 
        # --------------------------------

        # mask the region out of the source image
        masked_source_image = self._mask_image(source_image, masked_region_coors, random_mask)
        
        masked_source_image = np.expand_dims(self._array_image(masked_source_image),0)
        # -------------------------------
        # prepare the source image
        # --------------------------------

        source_image = self._array_image(source_image)
        
        # -------------------------------
        # assemble the outputs
        # --------------------------------
        output = {}
        output['source_image'] = source_image
        output['driver_image'] = driver_image
        output['masked_source'] = masked_source_image

        output['meta_coordinates'] = np.asarray(masked_region_coors + driver_image_coors)
        output['masked_region_width'] = masked_region_width
        
        if self.VQGAN_driver:
            # training the VQ-GAN to reconstruct driver images 
            output['source_image'] = output['driver_image'][0]

        return output



    
class E2EVE_val_dataset(E2EVE_dataset_base):
    # this class implements the validation described in the paper, where driver and source image
    # are sourced from different images, and the resulting sample is evaluated.
    # the assembling of the driver and source image is deterministic, and is guided by some accompanying
    # metadata
    
    def __init__(self, 
                 source_image_size=256, 
                 driver_image_size=64, 
                 path_to_images='',
                 validation_dataset_meta='',
                 non_paired_validation=True,
                 non_paired_validate_every_n_epoch=3,
                 filter_by_visual_cue=True,
                 num_val_samples_to_keep=2,
                 num_validation_samples=4,
                non_paired_val_inception_features_path = '',
                non_paired_val_patch_inception_features_path = '',
                non_paired_val_retrieval_dict_path = '',
                mask_method='block'):

        super().__init__(source_image_size, driver_image_size, image_scale_transform=1)
        auto_init_args(self)
        
        # load the validation meta data 
        with open(self.validation_dataset_meta,"rb") as f:
            data = pickle.load(f)
            
        [self.data_dict, self.dataset] = data
        self.dataset = self.dataset[:2] 
        
    def __getitem__(self, i):
        
        # -------------------------------
        # load the two images where source and driver will be taken from 
        # --------------------------------
        
        image_a_name = self.dataset[i][0]
        image_b_name = self.dataset[i][1]

        with open(os.path.join(self.path_to_images,image_a_name), "rb") as fid:
            image_a = PIL.Image.open(fid)
            if image_a.mode is not 'RGB':
                image_a = image_a.convert('RGB')
            # to solve a data loading bug
            image_a = self.source_transforms(image_a)
            
        with open(os.path.join(self.path_to_images,image_b_name), "rb") as fid:
            image_b = PIL.Image.open(fid)
            if image_b.mode is not 'RGB':
                image_b = image_b.convert('RGB')
            # to solve a data loading bug
            image_b =self.source_transforms(image_b)
            
        # -------------------------------
        # get the masked source image (from image_a)
        # --------------------------------
        
        if self.mask_method == 'block':
            masked_region_coors = [int(self.data_dict[image_b_name][1][0]*self.source_image_size), 
                                int(self.data_dict[image_b_name][1][1]*self.source_image_size),
                                int(self.data_dict[image_b_name][1][2]*self.source_image_size),
                                int(self.data_dict[image_b_name][1][3]*self.source_image_size)]
            random_mask = 0
        elif self.mask_method == 'random_mask':
            random_mask = self.data_dict[image_a_name][2]
            random_mask = np.concatenate((random_mask,random_mask,random_mask),axis=2)
            masked_region_coors = self._get_mask_spatial_extent(random_mask)

        masked_source_image = self._mask_image(image_a, masked_region_coors, random_mask)
        
        masked_source_image = np.expand_dims(self._array_image(masked_source_image),0)
        
        # -------------------------------
        # get the driver image (from image_b)
        # --------------------------------
        if self.mask_method == 'block':
            driver_image_coors = self.data_dict[image_b_name][0]
        elif self.mask_method == 'random_mask':
            driver_image_coors =  self.data_dict[image_a_name][3]
                
        driver_image_coors = [max(int(driver_image_coors[0]*self.source_image_size),0), 
                              min(int(driver_image_coors[1]*self.source_image_size),self.source_image_size),
                              max(int(driver_image_coors[2]*self.source_image_size),0),
                              min(int(driver_image_coors[3]*self.source_image_size),self.source_image_size)]
        
        driver_image = Image.fromarray(np.array(image_b).astype(np.uint8)[driver_image_coors[2]:driver_image_coors[3],driver_image_coors[0]:driver_image_coors[1],:])

        driver_image = np.expand_dims(self._array_image(self.driver_transforms(driver_image)),0)
        
        # mark on image_b where the driver was taken from
        image_b_with_driver_removed =  np.array(image_b).astype(np.uint8)
        image_b_with_driver_removed[driver_image_coors[2]:driver_image_coors[3],driver_image_coors[0]:driver_image_coors[1],0] = 0
        image_b_with_driver_removed = (image_b_with_driver_removed/127.5 - 1.0).astype(np.float32)
        
        # -------------------------------
        # get the driver image from image_a (so that NLL can be computed using ground truth of paired source and driver images)
        # --------------------------------

        if self.mask_method == 'block':
            paired_driver_image_coors = self.data_dict[image_b_name][0]
        elif self.mask_method == 'random_mask':
            paired_driver_image_coors =  self.data_dict[image_a_name][3]
            
        paired_driver_image_coors = [int(paired_driver_image_coors[0]*self.source_image_size),
                                     int(paired_driver_image_coors[1]*self.source_image_size),
                                     int(paired_driver_image_coors[2]*self.source_image_size),
                                     int(paired_driver_image_coors[3]*self.source_image_size)]

        paired_driver_image = Image.fromarray(np.array(image_a).astype(np.uint8)[driver_image_coors[2]:driver_image_coors[3],driver_image_coors[0]:driver_image_coors[1],:])
        
        paired_driver_image = np.expand_dims(self._array_image(self.driver_transforms(paired_driver_image)),0)
        
        # -------------------------------
        # prepare the source images
        # --------------------------------
        
        source_image = self._array_image(image_a)
        image_b = self._array_image(image_b)
        
        # -------------------------------
        # assemble the outputs
        # --------------------------------
        output = {}
        output['source_image'] = source_image
        output['driver_image'] = driver_image
        output['masked_source'] = masked_source_image
        
        output['meta_coordinates'] = np.asarray(masked_region_coors + driver_image_coors)
        
        output['mask'] = random_mask
        
        output["file_path_"] = str(i) + '_' + image_a_name + '_' + image_b_name
         
        output["paired_driver_image"] = paired_driver_image
        output["image_b_with_driver_removed"] = image_b_with_driver_removed
                 
        return output





# ----------------------------------------------------------------------------------------------------------------------------
# code for generating random masks
# -----------------------------------------------------------------------------------------------------------------------------




class Random_Masker_E2EVE():
    def __init__(self):

        self.shift_covariance = 0.01

    def _get_curve(self, points, **kw):
        segments = []
        for i in range(len(points)-1):
            seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
            segments.append(seg)
        curve = np.concatenate([s.curve for s in segments])
        return segments, curve

    def _ccw_sort(self, p):
        d = p-np.mean(p,axis=0)
        s = np.arctan2(d[:,0], d[:,1])
        return p[np.argsort(s),:]

    def _get_bezier_curve(self, a, rad=0.2, edgy=0):
        """ given an array of points *a*, create a curve through
        those points.
        *rad* is a number between 0 and 1 to steer the distance of
            control points.
        *edgy* is a parameter which controls how "edgy" the curve is,
            edgy=0 is smoothest."""
        p = np.arctan(edgy)/np.pi+.5
        a = self._ccw_sort(a)
        a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
        d = np.diff(a, axis=0)
        angle = np.arctan2(d[:,1],d[:,0])
        def f(angle):
            return (angle>=0)*angle + (angle<0)*(angle+2*np.pi)
        # f = lambda angle : (angle>=0)*angle + (angle<0)*(angle+2*np.pi)
        angle = f(angle)
        ang1 = angle
        ang2 = np.roll(angle,1)
        angle = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
        angle = np.append(angle, [angle[0]])
        a = np.append(a, np.atleast_2d(angle).T, axis=1)
        s, c = self._get_curve(a, r=rad, method="var")
        x,y = c.T
        return x,y, a

    def _get_random_points(self, n=5, scale=0.8, mindst=None, rec=0):
        """ create n random points in the unit square, which are *mindst*
        apart, then scale them."""
        mindst = mindst or .7/n
        aa = np.random.rand(n,2)
        d = np.sqrt(np.sum(np.diff(self._ccw_sort(aa), axis=0), axis=1)**2)
        if np.all(d >= mindst) or rec>=200:
            # change coors then scale then change back
            return aa
        else:
            return self._get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

    def _get_spatial_extent(self,coors):
        # get the boundaries of the mask

        x1 = np.min(coors[0,:])
        x2 = np.max(coors[0,:])
        y1 = np.min(coors[1,:])
        y2 = np.max(coors[1,:])

        return int(x1), int(x2), int(y1), int(y2)

    def _get_driver_box(self, image, coors):
        # find a square box that is completely contained in the mask - ideally as large as possible.
        # This is very hard to do in tractable time, so doing it with a non-optimal hack
        # (image, [x1,y1,x2,y2]) are the inputs


        # start with 2/3 width, and have 10 tries down to 5 pixels

        OG_box_width = (coors[2]-coors[0])
        OG_box_height = (coors[3]-coors[1])

        offset_x = int(coors[0])
        offset_y = int(coors[1])

        possibles = []
        for box_width in np.linspace(OG_box_width*(2/3),5,10):
            # gradually decrease the box size
            # box_width_ratio = random.uniform(parameters['crop_min'], parameters['crop_max'])

            # do a broad sweep of locations to see if the whole box would fit
            x1_values = np.linspace(0,OG_box_width-box_width,20)
            y1_values = np.linspace(0,OG_box_height-box_width,20)
            possibles = []
            totals = []
            for x1 in x1_values:
                for y1 in y1_values:

                    bbox = (int(x1) + offset_x, int(x1+box_width) + offset_x, int(y1) + offset_y, int(y1+box_width) + offset_y)
                    total = np.sum(image[bbox[2]:bbox[3],bbox[0]:bbox[1]])
                    lower_bound = box_width*box_width*0.001*255
                    if total < lower_bound:
                        possibles.append(bbox)
                        totals.append(total)

            if len(possibles) > 0:
                # this finds the largest box size at which one of the locations fits
                break

        if len(possibles) == 0:
            return None
        else:
            bbox = random.choice(possibles)
            return bbox

    def _get_mask_and_box(self, lower_bound=0.05, upper_bound = 0.6):

        driver_box = None
        # repeat until a feasible example is found
        # sample an area


        while driver_box is None:
            # get the mask
            rad = random.uniform(0.01,0.05)
            edgy = random.uniform(0,0.01)
            scale = 0.1
            area_target = random.uniform(lower_bound,upper_bound)
            area = 0
            # get the random white image

            c = np.array([0,0])
            aa = self._get_random_points(n=7, scale=scale) + c
            shift = np.random.multivariate_normal((0,0),np.array([[self.shift_covariance,self.shift_covariance],[self.shift_covariance,self.shift_covariance]]))

            bad = False
            while area < area_target:
                previous_area = area
                image = (np.ones((256,256))*255).astype('uint8')
                points = (aa - np.array([0.5,0.5]))*scale + np.array([0.5,0.5])
                points = points + shift
                x,y, _ = self._get_bezier_curve(points,rad=rad, edgy=edgy)
                x = np.clip(x*256, 0, 256)
                y = np.clip(y*256, 0, 256)
                draw_points = (np.asarray([x, y]).T).astype(np.int32)
                image = np.expand_dims(cv2.fillPoly(image, [draw_points], (0,0)),axis=-1).repeat(3,2)
                area = np.sum(image[:,:,0] == 0) / (256*256)
                # print(str(area) + ' ' + str(scale))
                scale += 0.04
                # Image.fromarray(image).save('/data/sandcastle/boxes/out/random.jpg')
                if area - previous_area < 0.005:
                    bad = True
                    break

            if not bad:
                # get the spatial extent

                x1, x2, y1, y2 = self._get_spatial_extent(np.asarray([x, y]))

                driver_box = self._get_driver_box(image, [x1,y1,x2,y2])
            else:
                driver_box = None


        annotated_image = cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 3)
        annotated_image = cv2.rectangle(annotated_image, (driver_box[0],driver_box[2]), (driver_box[1],driver_box[3]), (0,255,0), 5)

        image = image/255
        image =  np.expand_dims((1 - image).astype('uint8')[:,:,0],axis=-1)
        image = image.repeat(3,2)
        
        return image, list(driver_box), annotated_image, area

def bernstein(n, k, t):
    return binom(n,k)* t**k * (1.-t)**(n-k)
# bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)
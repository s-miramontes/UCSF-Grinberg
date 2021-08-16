'''
Script written for the generation of masks
given a list of os paths to .tif files.

These files must be in accordance to protocols
outlined by the markdown files in Grinberg's lab
GitHub repo or google doc shared for this project.

Author: Silvia Miramontes
Last Edit: 06/10/2021
'''

# dependencies
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import os
import shutil
import sys
import PIL
import time
import glob

import numpy as np
#import pandas as pd
from PIL import Image
from skimage.util import img_as_ubyte

import tensorflow as tf
from skimage import io
from keras import backend as K



from scipy import stats
from joblib import Parallel, delayed
import scipy.ndimage as ndi
from skimage import morphology, exposure, color
from skimage.exposure import rescale_intensity
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import square, erosion, disk

#import cv2
import collections
#import tensorflow_io as tfio

# Global Vars
IMG_SIZE = 128
M = IMG_SIZE
N = IMG_SIZE

# OS Functions
def sortPath(path):
    '''
    Sorts the list of paths in
    the list 'path'.
    
    Returns sorted list 's'.
    
    Author: Silvia Miramontes
    '''
    s = sorted(glob.glob(path), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    
    return s

def make(path):
    '''
    Makes directory in 'path'.
    
    Author: Silvia Miramontes.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def createSpace(region):
    
    '''
    Creates tmp folders for generated crops and outputs
    according to the image region (single image input).
    
    ------
    Input
        region: prefix (image name)
    ------
    Output
        cropsIN,cropsOUT,inputCrops,outputCrops,groundCrops,
        results:
        tupple of os paths to save all of the generated
        images.
    '''
    # create data folder
    make('data/')
    
    # crops pre-prediction (need the `make` function) *** SWITCH TO TMP? AND THEN DEL?
    # deleted model_name right after in/out
    cropsIN = 'data/tmp/generatedCrops/in/' + region + '/'
    cropsOUT = 'data/tmp/generatedCrops/out/'+ region + '/' # do we need this?
    make(cropsIN)
    make(cropsOUT)
    
    # crops post-prediction (need the `make` function) *** SWITCH TO TMP? AND THEN DEL?
    # added /tmp/ after data and before Predictions, deleted model_name in between
    # predictions/ and region
    inputCrops = 'data/tmp/Predictions/' + region + '/inputModel/'
    outputCrops = 'data/tmp/Predictions/' + region + '/outputModel/'
    groundCrops = 'data/tmp/Predictions/' + region + '/gtModel/' # do we need this?
    make(inputCrops)
    make(outputCrops)
    make(groundCrops)
    
    # changed from /Results to /Masks and model_name after Masks
    results = 'data/Masks/' + region + '/'
    make(results)
        
    return cropsIN, cropsOUT, inputCrops, outputCrops, groundCrops, results


# Preprocessing Functions
def fix_dim(mask):
    """
    Fixes dimensions of masks to be 2D or 3D
    
    Parameters:
    ------------
        mask: Masked image to fix
        
    Returns:
    -----------
        img: fixed channels for mask
    
    Author: Dani Ushizima
    """
    
    if len(mask.shape) != 3:
        img = color.gray2rgb(mask)
    else:
        img = color.rgb2gray(mask[:,:,0])
        
    return img

def enhanceSlice(img):
    """
    Enhances pixel intensities of image slice.
    
    Parameters:
    ---------------
        img: Image slice to enhance
    
    Returns:
    --------------
        newSlice: Enhanced image
    
    Notes:
    -------------
        Author: Dani Ushizima.
    """
    
    vmin, vmax = stats.scoreatpercentile(img, (1, 99.5))
    
    newSlice = np.clip(img, vmin, vmax)
    newSlice = (newSlice - vmin) / (vmax - vmin)
    newSlice = ndi.median_filter(newSlice, footprint=morphology.disk(radius=3))
    
    return newSlice

def preprocess(inputImg, index, neun=True, other=False):
    '''
    Prepares images to make predictions. This is
    2-fold
    
    1. Either all 3 channels on the input channel
       are NeuN (NeuN, NeuN, Neun).
    2. Or the 3 channels to be inputted are:
       (NeuN, TBR1(other), MAX(NeuN,TBR1)).
       
    The parameters in the function are as follows:
    --------
        inputImg: Multichannel image (not cropped)
        
        neun: True if a) is what you want.
        
        other: True if b) is what you want.
        
        No, you cannot flag both as true. PLEASE.
        
    Output:
    ---------
        inp: preprocessed image (with proper image
        channel stacking).
        out: dummy mask for algorithm
        
    Author: Silvia Miramontes.
    '''
    if neun:
        if index:
            NeuN = enhanceSlice(inputImg[:,:,index]).reshape((inputImg.shape[0],
                                                      inputImg.shape[1], 1))
            inp = np.concatenate((NeuN, NeuN, NeuN), axis=2)
        else: # removed the enhance of the sinlge picture
            NeuN = enhanceSlice(inputImg).reshape((inputImg.shape[0],
                                                      inputImg.shape[1], 1))
            inp = np.concatenate((NeuN, NeuN, NeuN), axis=2)
    elif other:
        NeuN = enhanceSlice(inputImg[:,:,1]).reshape((inputImg.shape[0],
                                                      inputImg.shape[1], 1))
        other = enhanceSlice(inputImg[:,:,2]).reshape((inputImg.shape[0],
                                                      inputImg.shape[1], 1))
        MAX = np.maximum(NeuN, other).reshape((inputImg.shape[0],
                                                      inputImg.shape[1], 1))
        inp = np.concatenate((other, NeuN, MAX), axis=2)
        
    out = np.empty((inputImg.shape[0], inputImg.shape[1]))
    
    return inp, out

def nextDigit(n, m):
    '''
    Find the number closest to n and divisible by m
    This function is used for the padding of the 
    images. 
    
    That is, in case the images given are not divisible
    by 128. So we need to expand the image given to
    have its dimensions be divisible by 128.
    
    Note that this is all necessary because the input
    images required for the model are to be (128,128,3)
    That is 128x128x3. 
    
    Author: Silvia Miramontes
    '''
    q = int(n/m)
    
    # first possible closest num
    n1 = m * q
    
    # second possible closest num
    n2 = (m * (q + 1)) 
    
    # pick the biggest one always
    if n1 < n:
        return n2
    return n1

def padImage(orig_img, dim, threeD=False, twoD=False):
    '''
    Pads orgiginal image to make
    its dimensions divisible by dim
    
    Input
    ------
        orig_img = original image, raw
        dim = desired dimension to divibe by
        
    Return
    ------
        newImage = padded image
        
    Author: Silvia Miramontes
    '''
    
    if threeD:
        w, l, chan = orig_img.shape[0], orig_img.shape[1], orig_img.shape[2]
        
        if w%dim != 0 or l%dim != 0:
            new_w = nextDigit(w, dim)
            new_l = nextDigit(l, dim)
            
            moreW = new_w - w
            moreL = new_l - l
            orig_img = np.pad(orig_img,[(0,moreW), (0,moreL), (0,0)], mode='constant')
            
            return orig_img
        
        else:
            return orig_img

    elif twoD:
        w, l = orig_img.shape[0], orig_img.shape[1]
        
        if w%dim != 0 or l%dim != 0:
            new_w = nextDigit(w, dim)
            new_l = nextDigit(l, dim)
            #newImage = np.empty((new_w,new_l))
            #newImage[:w,:l] = orig_img
            
            moreW = new_w-w
            moreL = new_l-l
            orig_img = np.pad(orig_img,[(0,moreW),(0,moreL)], mode='constant')
            
            # Checking outputs
            #plt.imshow(orig_img)
            #plt.title('Fake mask')
            #plt.show()
            
            return orig_img
        else:
            return orig_img    

def crop(M, N, raw_2d, mask_2d):
    '''
    Generate crops of raw and masks of size MxN
    
    Parameters
    ----------
        - M: first dimension of crop
        - N: second dimension of crop 
        - raw_2d: 2D raw image 
        - mask_2d: 2D mask of raw image
        
    Outputs
    ---------
        - raw: array of image crops MxN 
        - masks: array of mask image crops MxN
        
    Author: Silvia Miramontes
    '''
    # since now multichannel the shape dimensions are saved as (r, c, chans)
    raw = [raw_2d[x:x+M, y:y+N,:] for x in range(0,raw_2d.shape[0],M) for y in range(0,raw_2d.shape[1],N)]

    # image dimensions for TBR1 mask remain the same they are 2D
    masks = [mask_2d[x:x+M, y:y+N] for x in range(0,mask_2d.shape[0],M) for y in range(0,mask_2d.shape[1],N)]
    
    return raw, masks

def save_img(path, num, image):
    """
    Saves image crops in provided path
    as .png format.
    
    Params
    -------
        path: file path where to save
        num: index number for saving
        image: image to save
        
    Returns
    -------
        N/A, this function saves the img.
        
    Author: Silvia Miramontes
    """
    save_to = path + str(num) + '.png'
    #img = Image.fromarray(image)
    #img.save(save_to)
    io.imsave(save_to, image.astype('uint8'))

def saveCrops(pathList, cropsList):
    '''
    Saves the generated crops in given
    path list.
    
    Inputs:
    -------
        - pathList: a list containing prefix
                    to save crops_input, and
                    crops_output. Order of list
                    matters.
                    e.g. paths=[crops_input,
                    crops_output]
        
        - cropsList: a list containing lists of
                    crops of either input (raw)
                    or masks. Order also matters.
                    e.g. crops = [raws, masks]
    Outputs:
    --------
        None
    '''
    # msg to user
    #print('Saving generated crops...')
    
    for i, path in enumerate(pathList):
        #access the ith set of crops
        ndarr = cropsList[i]
        # start index at 0
        j = 0
        
        # for each img in ndarr list
        for im in ndarr:
            # this is for masks
            if i>=1:
                save_img(path, j, im.astype('uint8'))
            # this is for input imgs
            else:
                # add if statement to check dtype
                #print(im.dtype)
                #im_enhanced = img_as_float32(im)
                to_uint = im*255
                save_img(path, j, to_uint.astype('uint8'))
            # increasing index by 1, for naming conventions.
            j +=1
            
    #print('Done!')


# Tensorflow Functions
def parse_image(img_path: str) -> dict:
    """
    Load an image and its annotation (mask) and returning a dictionary.
    
    Params
    -------------
    img_path: str
        Imag (not the mask) location.
        
        
    Returns
    ------------
    dict
        Dictionrary wrapping an image and its annotation
    """
    
    image = tf.io.read_file(img_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # For one Image path:
    # data/P2593_ROI_1/training/0.tiff
    # Its corresponding annotation path is:
    # 'data/P2593_ROI_1/annotations/0.tiff'
    
    mask_path = tf.strings.regex_replace(img_path, "in",
                                         "out")
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)
    
    # Substituting 255 pixel values in mask for 1
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)
    
    return {'image': image, 'segmentation_mask': mask}

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.
    
    Notes
    -------
    Since this is for the test set, we don't need to apply
    any data augmentation technique.
    
    Parameters
    -----------
    datapoint: dict
        A dict conatining an image and its annotation.
        
    Returns
    -------
    tuple
        A modified image and its annotation
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))
    
    #input_image, input_mask = normalize(input_image, input_mask)
    
    return input_image, input_mask

def display_sample(display_list):
    
    """Show side-by-side an input image,
    the ground truth and the prediction."""
    
    plt.figure(figsize=(18,18))
    
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]),
                       cmap='gray')
        #plt.axis('off')
    #plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis = -1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def save_predictions(lst, idx, raw_crops, gt_crops, pred_crops):
    '''
    Save the crops for the predictions made.
    Since we go in order, with raw, ground, and pred
    the raw images should preserve the appropriate order.
    
    Inputs
    ---------
        lst: list of images [raw, ground truth, pred]
        idx: image # to save with
    
    Outputs
    --------
        None
    '''
    
    # a= raw, b=gt, c=pred
    # rawCrops, gtCrops and predCrops are file paths def above
    
    a = io.imsave(raw_crops + str(idx) + '.png', lst[0].numpy().astype('uint8'))
    b = io.imsave(gt_crops + str(idx) + '.png', lst[1].numpy().astype('uint8'))
    c = io.imsave(pred_crops + str(idx) + '.png', lst[2].numpy().astype('uint8'))
    

def show_predictions(model, raw_crops, gt_crops, pred_crops, dataset=None, num=1):
    if dataset:
        j=0
        for image, mask in dataset.take(num):
            #print(j)
            #predict the mask for 'image'
            #print('actually predicting..')
            pred_mask = model.predict(image)
            #print('creating mask...')
            new_mask= create_mask(pred_mask)
            #display_sample([image[0], mask[0], new_mask])
            #print('saving predictions...')
            save_predictions([image[0], mask[0], new_mask], j,
                             raw_crops, gt_crops, pred_crops)
            j += 1
    else:
        return

    
# Restitching Functions
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def orderSmall(path_list):
    '''
    Ordering Images that have <= 100 
    image crops
    '''
    
    total = len(path_list)
    
    a = [] 
    b = [] 
    
    a.append(path_list[:1])
    
    groupTens = (total//10)-1 
    units = total%10
    
    decs = True
    uns = False

    i = -10
    while decs:
        i += 11
        a.append(path_list[i]) 
        b.append(path_list[i+1:i+11]) 
        
        if len(b) == groupTens:
            
            i += 11 # i = 34
            a.append(path_list[i])
            b.append(path_list[i+1:i+units+1]) 
            i = i + units + 1
            decs = False
            uns = True
            
    if uns:
            a.append(path_list[i:])
            
    a = list(flatten(a))
    b = list(flatten(b))
    
    ordered = a + b
    
    return ordered

def orderStitch(path_list):
    '''
    This is under the assumption that the 1000's
    don't change. 
    '''
    
    total = len(path_list)
    
    a = [] 
    b = []
    c = []
    
    a.append(path_list[:2])
    
    totalTens = total//10
    hundreds = total//100
    units = total%10
    tens = (total%100) - units 
    
    count_100s = 1 
    
    if total > 100 and total <= 1000:
        big = True
        small = False

        i = -9
        
        while big:
            i += 11 
            b.append(path_list[i]) 
            c.append(path_list[i+1:i+11])
            
            if len(c)%10 == 0: 
                count_100s += 1 
                a.append(path_list[i+11])
                i += 1 
            if count_100s == hundreds:
                tens -= 10 
                if tens < 0 and units ==0: 
                    c.append(path_list[i+1:i+11])

                    i += 11

                    nextDec = (total%100)-10
                    left = nextDec - (totalTens-10)
                    b.append(path_list[i:i+left]) 

                    big = False
                    small = True
                    end2 = i+left
                elif tens <0 and units != 0:
                    
                    i += 11 
                    b.append(path_list[i])

                    end = i+(units+1) 
                    c.append(path_list[i+1:end]) 

                    nextDec= (total%100)- units 
                    groupsLeft = (total%100)-10 
                    left = nextDec - groupsLeft 
                    b.append(path_list[end:end+left]) 
                    
                    end2 = end+left

                    big = False
                    small = True
        i = end2       
        while small:
            try:
                a.append(path_list[i]) 
                b.append(path_list[i+1:i+11])
                i += 11
                
            except IndexError:
                a = list(flatten(a))
                b = list(flatten(b))
                c = list(flatten(c))

                in_order = a + b + c

                small=False

                return in_order
    
    elif total <= 100:
        
        final = orderSmall(path_list)
        return final

def fixPaths(ordered_list, bad_list_path):
    """
    Given an `ordered_list` of paths, extract the titles
    of the images and attach to the prefix of the given
    `bad_list_path`.
    
    Inputs
    ----------
        ordered_list: list of ordered paths
        bad_list: out of order prefix
    
    Returns
    ---------
        fixed: the bad list now with correct order
        
    Author: Silvia M.
    """
    fixed = []
    
    for i,el in enumerate(ordered_list):
        
        item = el.split('/')[-1] # image title
        
        fixed_path = bad_list_path + item
        fixed.append(fixed_path)
        
    return fixed  

def stitchBack(crops, img_dim, size1, size2):
    '''
    Stitch back together the predicted and 
    generated crops.
    
    Inputs
    -------
        crops: an ordered list of crops
        img_dim: image dimensions desired
        size1: length of crop
        size2: width of crop
        
    Outputs
    -------
        img: image of the stitched back crops
    '''
    
    img = np.empty(img_dim)
    
    
    idx = 0
    
    for k in range(0, img.shape[0], size1):
        for l in range(0, img.shape[1], size2):
            if len(img.shape) > 2:
                img[k:k+size1, l:l+size2,:] = io.imread(crops[idx])
            else:
                img[k:k+size1, l:l+size2] = io.imread(crops[idx])
            idx +=1
            
    return img

def correct_order(ogInput, compareWith):
    '''
    Obtains the correct order of images.
    Assuming that ogInput file list preserved
    the appropriate order during the cropping.
    
    Inputs:
    ------------
        ogInput: sorted list of cropped files before prediction
        compareWith: sorted list of cropped files after prediction
        
    Output:
    ------------
        list_correct = a list of the .png files in appropriate order.
        
    NOTES:
        This is under the assumption that the ogInput list has preserved
        the appropriate order generated by the 'crop' function.
        All lists passed in must be sorted with the key 'lambda' function.
    '''
    list_correct=[]
    
    for raw in list(ogInput):
        for pred in compareWith:
            if equal(raw, pred):
                path = pred.split('/')[-1]            
                if path not in list_correct:
                    list_correct.append(path)
                
    return list_correct

# Postprocessing Functions
def cleanMask(labelImg, minsize_area, removehole_area):
    '''
    Removes small holes so that neuron segmentations
    are uniform. The function also removes noise
    or spreckles within the detection.
    Ouput is unit8.
    
    --------
    Input: labelImg = uint8 mask
           minSize = minimum size for pepper to be
           removed.
    -------
    Output:
           final = final mask now cleaned
           
    Auth: Silvia M.
    '''
    final = morphology.remove_small_holes(labelImg, removehole_area,
                                             connectivity=2)
    final = morphology.remove_small_objects(final, minsize_area)
    final = img_as_ubyte(final)
    
    return final

# Main Functions
def getMask(data, model_path, index, minsize_area=20, removehole_area=500, save_input=False):

    '''
    getMask creates a mask using the model stored in os model_path.

    ----
    Input:
        data: list of os paths to .tif files
        model_path: os path to Model 
        index: index to use for image channel

    Output:
        finishedMasks: list of finished masks from data inputted.

    Author: Silvia Miramontes
    '''
     
    finishedMasks= []
    modelInputs = []
    #for model in model_list:
    modelName = model_path.split('/')[-1] # use model name to load .h5 file

    for pic in data:        
        # get region name (without file extension .tif)
        region = pic.split('/')[-1].split('.')[0] #this just gets the file name, hence name 'region'

        # folders to save the generated crops should have been created (function should output path locations)
        #print('Getting crops...')
        cropIN, cropOUT, cropINpred, cropOUTpred, cropGTpred, res = createSpace(region)

        # read image
        im = io.imread(pic)

        # preprocess
        #print('Preprocessing image...')
        inp, msk = preprocess(im, index) #check channel number, should be @ 4 in fiji 3 py
        inp = padImage(inp, M, threeD=True) 
        msk = padImage(msk, M, twoD=True)
           
        # generate crops 128x128
        #print('Generating crops...')
        inp_crops, out_crops = crop(M, N, inp, msk)
        crops = [inp_crops, out_crops]
        here = [cropIN, cropOUT] 
        saveCrops(here, crops)

        # get Crops
        #print('Getting correct paths...')
        pathInput = sortPath(cropIN+ '*.png')
        pathOutput = sortPath(cropOUT + '*.png')

        #print('Converting dataset to tensors...')
        # convert to tensors
        to_list = tf.data.Dataset.list_files(pathInput, shuffle = False,
                                                 seed=False)
        tensors = to_list.map(parse_image)
        tensors = tensors.map(load_image_test)
        tensors = tensors.batch(1)

        # predict tensors
        #print('Model loading...')
        
        model = tf.keras.models.load_model(model_path)
        #print('Model loaded, now predicting..')
        show_predictions(model, cropINpred, cropGTpred, cropOUTpred, 
                             dataset=tensors, num=len(inp_crops))

        # sort predictions
        #print('Sorting predictions...')
        listCrops1 = sortPath(cropINpred+ '*.png')
        listCrops2 = sortPath(cropGTpred+ '*.png')
        listCrops3 = sortPath(cropOUTpred+ '*.png')

        # restitch (input, then stained.)
        #print('Putting images in order....')
        ordered = orderStitch(listCrops1) 
        stained = fixPaths(ordered, cropINpred)
        preds = fixPaths(ordered, cropOUTpred)
        gts = fixPaths(ordered, cropGTpred)

        # get mask ordered and to correct size (PUT THEM BACK TOGETHER)
        #print('Restitching...')
        finalMask = stitchBack(preds, msk.shape, 128, 128) 
        finalMask = finalMask[0:im.shape[0], 0:im.shape[1]]
        inp = inp[0:im.shape[0], 0:im.shape[1]]
            
        # postprocessing
        #print('Postprocessing...')
        finalMask = cleanMask(finalMask.astype('uint8'), minsize_area, removehole_area)
        finishedMasks.append(finalMask)
        modelInputs.append(inp)

        #save final outputs
        #print('Saving...')
        resSave = res + region 
        if save_input: io.imsave(resSave + '-Input.png', inp)    
        io.imsave(resSave + '-Mask.png', finalMask.astype('uint8'))
        
        # Clean up: delete tmp files
        shutil.rmtree('data/tmp')
            
    return finishedMasks, modelInputs
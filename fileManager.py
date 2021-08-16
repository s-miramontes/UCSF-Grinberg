'''
Script to check whether files are compliant
to protocols establsihed and outlined in other
documents shared with the lab.

Author: Silvia Miramontes
'''

import os
import sys
import glob
import tifffile
import numpy as np 

import boto3
from PIL import Image
from io import BytesIO

def rgb(im):
    dims = len(im.shape)
    if dims != 3:
        print('Check if images are all RGB')
        return False
    elif im.shape[2] != 3:
        print('Check whether file is RGB')
        return False
    else:
        return True     

def file_size(im):
    w, l = im.shape[0], im.shape[1]
    
    if w < 800 or w > 1000:
        print("Check your width size")
        return False 
    elif l < 800 or l > 1000:
        print('Check your length size')
        return False
    else:
        return True

def file_checker(fileList):
    '''
    This function takes in as an argument
    the os path to the directory where all
    stained files will be saved before counting
    each of the stains.
    
    -------
    Input:
        dir_path: os path to folder

    Output:
    	bool: T/F
    -------
    Note:
        The files within the directory path
        must meet the requirements/protocols
        outlined at___. Otherwise the tool
        will not work.
    '''
    
    #fileList = os.listdir(dir_path)
    
    for file in fileList:
        # extension
        #print(file)
        if file.endswith('.tif') or file.endswith('.tiff'):
            read = io.imread(file)
            # file size
            if file_size(read):
                # rgb check 
                if rgb(read):
                    return True
                else:
                    print("You can't use this yet. Check for RGB.")
                    return False
            else:
                print("You can't use this yet. Check your image size.")
                return False
        else:
            print('Check your file extensions!')
            print("You can't use this yet.")
            return False
        
def show_me(your_masks, your_modelInputs):
    
    for j,i in enumerate(your_masks):
        f, a = plt.subplots(1,2, figsize=(20,16))
        a[0].imshow(i, cmap='gray')
        a[0].set_title('Mask Output')
        a[1].imshow(your_modelInputs[j])
        a[1].set_title('Model Input')
    print('Satisfied? If not, feel free to edit the parameters mentioned above...')
    plt.show()
    

# Boto3 File imports
def read_image_from_s3(usr, directory, bucket):
    '''
    Load file(s) from Jupyter S3
    
    Params
    ------
    usr: username
    bucket: string with username
    key: string Path in S3
    
    Returns
    -------
    np array (list?)
        List of np array images
    '''
    
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('bchsi-spark02')
    
    path_s3 =  "jupyter/" + usr + "/Data/" + directory + "/*.tif"
    
    obj = bucket.Object(path_s3)
    response = obj.get()
    
    

def get_files(user, dir_prefix):
    '''
    get_files returns a prefix to where all files will be
    in current directory (jupyterhub home) to reload.
    
    Inputs:
    ---------
        - user : username
        - dir_prefix: where files are stored in S3 bucket
    
    Output:
    ---------
        - PREFIX: os path to dir where all image files 
                are now located.
    '''
    
    s3 = boto3.client('s3')
    BUCKET = 'bchsi-spark02'
    PREFIX = 'home/' + user + '/' + dir_prefix + '/'
    objects = s3.list_objects(Bucket=BUCKET, Prefix=PREFIX)['Contents']
    
    print('Downloading your files to current directory...')
    for s3_object in objects:
        
        s3_key = s3_object['Key']
        path, filename = os.path.split(s3_key)
        
        if not filename.startswith('.'): #avoiding .s3 key files
            if len(path) != 0 and not os.path.exists(path):
                os.makedirs(path)
            if not s3_key.endswith('/'):
                download_to = path + '/' + filename
                s3.download_file(BUCKET, s3_key, download_to)
    print('Done!')
    
    print('     ')
    print('Your Files are here:', PREFIX)
    
    return PREFIX


def makeMasks(your_region, model_location, min_area_input, smallhole_inp):
    '''
    Simple function to check your files and generate
    masks given the region `your_region`
    
    Input
    ------
        - your_region: list of `.tif` files
    Outputs
    ------
        - msks: generated masks (these are saved)
        - inps: inputs to the model (for comparison)
    '''
    if file_checker(your_region):
        print('Working...')
        # note how I am inputting the variable 'region_1' to get the masks for that particular region.
        msks, inps = getMask(your_region, model, 2, minsize_area = min_area, removehole_area = smallhole_)
        print('Your masks are ready!')
    return msks, inps
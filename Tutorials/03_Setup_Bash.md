# Getting Ready to use Segmentation tool

When you first start using the tool you will need to paste some of these commands in your terminal/bash so that you can obtain the notebooks/python scripts necessary.

### Please follow these instructions along with the video uploaded here (click on the image):

[![Tutorial 3](https://img.youtube.com/vi/73LvjtIDEWo/0.jpg)](https://youtu.be/73LvjtIDEWo)


# Get notebook and python files
  - Once you've successfully logged into Information Commons (IC) run this command but **substitute USERNAME** for **your IC username**.

    `aws s3 sync s3://bchsi-spark02/data_grinberglab/ s3://bchsi-spark02/jupyter/USERNAME/`
    
  - This should have placed all of the files necessary for the segmentation tool to run in your jupyter environment. **You will not need to do this again, unless there are updates.**
  
# Check JupyterHub
  - Proceed to log on to JupyterHub as in the first tutorial, to see whether the folder `tools` appears on your left hand side.
  - You can access this folder by double-clicking into it.
  - You should see 3 files `get-masks-SU21.ipynb`, `fileManager.py`, and `maskGenerator.py`.
  - There should also be a folder titled `Models`. This is where the segmentation model is saved.
  - Seeing these files on your left-hand-side, means that these files are saved in your jupyter s3 bucket. 
  - These files/directories will not be deleted and will remain in your jupyter bucket everytime you log into IC's JupyterHub. 

## To upload your data see: 03_data_upload.md

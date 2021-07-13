# Getting Ready to use Segmentation tool

When you first start using the tool you will need to paste some of these commands in your terminal/bash so that you can obtain the notebooks/python scripts necessary.
*Please follow these instructions along with the video uploaded here*.

1. Get notebook and python files
  - Once you've successfully logged into Information Commons (IC) run this command but **substitute USERNAME** for **your IC username**.

    `aws s3 sync s3://bchsi-spark02/data_grinberglab/ s3://bchsi-spark02/jupyter/USERNAME/`
    
  - This should have placed all of the files necessary for the segmentation tool to run in your jupyter environment. **You will not need to do this again, unless there are updates.**
  
2. Check whether your files are uploaded on your jupyterhub.
  - Proceed to connect into jupyterhub to see whether the folder `tools` appears on your left hand side.
  - You can access this folder by double-clicking into it.
  - You should see 3 files `get-masks-SU21.ipynb`, `fileManager.py`, and `maskGenerator.py`.
  - By seeing these files on your left-hand-side, means that these files are saved in your jupyter s3 bucket. 
  - These will not be deleted and will be there everytime you log into IC's JupyterHub. 

3. Transfer your data.

# Data Upload

#### How do I upload my data into information commons JupyterHub?

Do not. **DO NOT** upload your images using the JupyterHub interface. File uploads above 1MB will be deprecated if uploaded through that interface.
So please avoid that. 

## Here is how:
1. Upload your folder containing `.tif` images into your master node (via cyberduck) here is how (scroll to the cyberduck section): https://wiki.library.ucsf.edu/display/IC/Storing+and+Loading+Data.
2. Once this is succesfully done, you will use the following command to copy your uploaded directory/folder of `.tif` images **into your personal s3 bucket.**
3. Type your username in place of `USER_NAME`, also note that `myNewData` below is a placeholder for your actual folder name, so please change accordingly.

  `aws s3 cp myNewData s3://bchsi-spark02/home/USER_NAME/`
  
4. **Please upload directories and not single *.tif files.**

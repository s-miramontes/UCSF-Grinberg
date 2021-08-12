# File Protocol for Segmentation Tool

It is important to keep the data that is fed into the segmentation tool to follow the *same* file protocol. This entails:
- File extension
- Number of Channels to Input
- Image size (in pixels)

The reason why this is relevant is because this tool was created to follow a very specific pattern, and thus, expects the same input everytime it is used. 

# File Exension
All files inside your `counting` folder **must be in .tiff** `.tif` and `.tiff` extensions ok.
Not `.jp2`, not `.png`.

# Number of Channels to Input
All files inputted must be 3 channels (RGB), in magenta color.

## Image Size
All files in the folder must be >= 800x800 and  <=1000x1000 pixels, nothing more, nothing less.

# What happens if these folders don't follow this protocol?
Failure to meet these rules will result in an error, and you will not be able to use the segmentation tool. 

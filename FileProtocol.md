# How do I name my Files?

With an effor to stay organized and have the semi-automated segmentation tool in JupyterHub, all files must follow the file naming protocols established here. 

**Why do I need to follow these rules?**

*The segmentation tool parses through the filename you generate to determine which is the channel of interest (to create a mask), how many channels there are in each of your scanned images, how many different staining mechanisms, and what kind of brain region. All of this is important for the notes outputted with each segmentation.*

*TLDR; Whether you are scanning at UC Berkeley or UCSF, if you don't name your files properly, you cannot use the segementation tool.*


## The Rules

1. Images greated than 1000x1000 px will **not** be accepted. See "How to Crop" below. 
2. All files **must be** a .tif extension.
3. The only `.` in the file name must be the one that separates the file extension from the rest of the file name (e.g. `TonyaPic_45.tif`).
4. The only non-alphanumeric characters allowed are underscores `_` as these are used to separate the letter characters from each other.
5. If another non-alphanumeric character is detected, the tool **will not** run. 
6. The order of the contents in the filenames **must always** be followed.
7. Date Format must always be DD/MM/YYYY.
8. Cropping the scanned region must follow a specified order in the file name (e.g. `ROI1` VS `ROI2` VS `ROI3` are all different regions.)
9. Only one channel to count for at a time. 


## How to Crop

1. Each of the scanned brain regions must be cropped to 1000x1000 px or less. Anything bigger will not be accepted.
2. When the crops are generated, you must name each image accordingly. Please see the protocols to follow below, we've also included an example for your convenience. 


## A note on Non-Alphanumeric Characters

Each alphanumerical character in your filename should be separated by the underscore `_` character (excluding the `.` before the `tif` file extension). See example below.
**IMPORTANT: Any other `.` within the alphanumeric characters will cause the segmentation tool to fail.**


## Is there an example I can follow?

Of course! 
Take for example the following file name:

1. `Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.tif`

OR

3. `Tonya_EC_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.tif`

Let's break that filename 1. down:

- `Tonya` is an example of the name of whose project you're working on.
 
- Obligatory character to separate `_`

- `TMA` and `_` and `A7` depict the brain region that is scanned. Note that brain region can also be simply denoted by `TC`.
 
- Again, the obligatory character to separate `_`.

- `NeuN` and `_` and `350` denote the biomarker (`NeuN`), obligatory character to separate (`_`), and biomarker channel (`350`).

- Again, the obligatory character to separate `_`.

- `TTC` and `_` and `488` denote the biomarker (`TTC`), obligatory character to separate (`_`), and biomarker channel (`488`).

- Again, the obligatory character to separate `_`.

- `CP13` and `_` and `546` denote the biomarker (`CP13`), character to separate (`_`), and biomarker channel (`546`).

- Again, the obligatory character to separate `_`.

- `BME` and `_` and `647` denote the biomarker (`BME`), character to separate (`_`), and biomarker channel (`647`).

- Again, the obligatory character to separate `_`.

- `TAU` and `_` and `790` denote the biomarker (`TAU`), character to separate (`_`), and biomarker channel (`790`).

- Again, the obligatory character to separate `_`.

- `01202021` denotes the date in which you completed the scan. The date format must be MM/DD/YYYY.

- Again, the obligatory character to separate `_`.

- `P5590` is the name of the specimen you scanned.

- Again, the obligatory character to separate `_`.

- `ROI1`, is the name of the region of the slide you scanned for the crops you generate. Generally follow `ROI#`, for `#` is whatever crop number you generated for that specific brain region.

- Again (and the last one required), the obligatory character to separate `_`.

- `CH01` denotes the channel of interest to create a mask for. This is the channel number the tool will use to count the neurons. In this case Channel 1 (`CH01` formatted as: `CH##`), refers to the `NeuN` in the filename. 

**IMPORTANT:** No more than 1 channel specified per image. 

## Examples of unnaceptable file names:

1. `Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.ome.tif`

    - **no extra periods**
      - ".ome" should not be in this file name. This is also two file extensions in one image. Only ".tif" extensions allowed.

2. `Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1.2.4_CH01.2.tif`

     - **no extra periods**
       - "ROI1.2.4" nor "CH01.2" in your file name besides the allowed file extension. 

3. `Tonya_Piergies__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.tif`

    - **no last names separated by extra `_`** 
      - ("Tonya_Piergies.")

4. `Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01_CHO3.tif`

    - **no extra channels at the end**
      - "CHO1_CH03" if you need to count **two or more** channels of the same region you must save each image separately with the  appropriate file name. 

      That is two separate files as:

         - `Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.tif`
         - `Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH03.tif`

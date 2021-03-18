# How do I name my Files?

With an effor to stay organized and have the semi-automated segmentation tool in JupyterHub, all files must follow the file naming protocols established here. 

**Why do I need to follow these rules?**

*The segmentation tool parses through the filename you generate to determine which is the channel of interest (to create a mask), how many channels there are in each of your scanned images, how many different staining mechanisms, and what kind of brain region. All of this is important for the notes outputted with each segmentation.*

*TLDR; Whether you are scanning at UC Berkeley or UCSF, if you don't name your files properly, you cannot use the segementation tool.*


## The Rules

1. Images greated than 1000x1000 px will **not** be accepted. See "How to Crop" below. 
2. All files **must be** a .tif extension.
3. The only '.' (period) in the file name must be the one that separates the file extension from the rest of the file name (e.g. TonyaPic_45.tif).
4. The only non-letter characters allowed are underscores _ These are used to separate the letter characters from each other.
5. If another non-letter character is detected, the tool will not run. 
6. The order of the contents in the filenames **must always** be followed.
7. Date Format must always be DD/MM/YYYY.
8. Cropping the scanned region must follow a specified order in the file name (e.g. ROI1 VS ROI2 VS ROI3 are all different regions.)
9. Only one channel to count for at a time. 


## How to Crop

1. Each of the scanned brain regions must be cropped to 1000x1000 px or less. Anything bigger will not be accepted.
2. When the crops are generated, you must name each image accordingly. Please see the protocols to follow below, we've also included an example for your convenience. 


## A note on Non-Alphanumeric Characters

Each alphanumerical character in your filename should be separated by the underscore "\_" character (excluding the "."" before the 'tif' file extension). See example below.
**IMPORTANT: Any other '.' within the alphanumeric characters will cause the segmentation tool to fail.**


## Is there an example I can follow?

Of course! 
Take for example the following file name:

1. **Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.tif**
OR
2. **Tonya\_EC_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.tif**

Let's break that filename (1) down:

1. "Tonya" is an example of the name of whose project you're working on.
 
2. Obligatory character to separate '\_'

3. "TMA" and "\_" and "A7" depict the brain region that is scanned. Note that brain region can also be simply denoted by "TC".
 
4. Again, the obligatory character to separate '\_'.

5. "NeuN" and "\_" and "350" denote the biomarker (NeuN), obligatory character to separate (\_), and biomarker channel (350).

6. Again, the obligatory character to separate '\_'.

7. "TTC" and "\_" and "488" denote the biomarker (TTC), obligatory character to separate (\_), and biomarker channel (488).

8. Again, the obligatory character to separate '\_'.

9. "CP13" and "\_" and "546" denote the biomarker (CP13), character to separate (\_), and biomarker channel (546).

10. Again, the obligatory character to separate '\_'.

11. "BME" and "\_" and "647" denote the biomarker (BME), character to separate (\_), and biomarker channel (647).

12. Again, the obligatory character to separate '\_'.

13. "TAU" and "\_" and "790" denote the biomarker (TAU), character to separate (\_), and biomarker channel (790).

14. Again, the obligatory character to separate '\_'.

15. "01202021" denotes the date in which you completed the scan. The date format must be MM/DD/YYYY.

16. Again, the obligatory character to separate '\_'.

17. "P5590" is the name of the specimen you scanned.

18. Again, the obligatory character to separate '\_'.

19. "ROI1", is the name of the region of the slide you scanned for the crops you generate. Generally follow ROI#, for '#' is whatever crop number you generated for that specific brain region.

20. Again (and the last one required), the obligatory character to separate '\_'.

21. "CH01" denotes the channel of interest to create a mask for. This is the channel number the tool will use to count the neurons. In this case Channel 1 ("CH01" formatted as: CH##), refers to NeuN. 


**IMPORTANT:** No more than 1 channel specified per image. 

## Examples of unnaceptable file names:

1. **Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.ome.tif**

NO EXTRA PERIODS --> ".ome" should **NOT** be in this file name. This is also two file extensions in one image. Only ".tif" extensions allowed.

2. **Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1.2.4_CH01.2.tif**

    NO EXTRA PERIODS -> "ROI1.2.4" nor "CH01.2" in your file name besides the allowed file extension. 

3. Tonya_Piergies__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.tif**

    NO LAST NAMES SEPARATED BY "\_" ("Tonya Piergies.")

4. **Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01_CHO3.tif**

    NO EXTRA CHANNELS AT THE END --> "CHO1_CH03" if you need to count **two or more** channels of the same region you must save each image separately with the  appropriate file name. 

That is:

- **Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH01.tif** (is one file).
- **Tonya__TMA_A7_NeuN_350_TTC_488_CP13_546_BME_647_Tau_790_01202021_P2590_ROI1_CH03.tif** (is the other needed file).

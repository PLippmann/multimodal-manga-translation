# LoveHina JA-PL Manga Translation Data Set

## Images
As the images are taken from the Manga109-s data set, which prohibits sharing the images with a third party, they are not included in this repository. You can download the images for Love Hina volumes 1 and 14 from the Manga109-s website: http://www.manga109.org/en/download_s.html

These should then be placed in this directory under the following paths:

```./LoveHina_vol1_page_split/``` and ```./LoveHina_vol14_page_split/```, each containing the numbered .jpg files for the pages of the respective volume.

## JA-PL Translation Annotations
The annotations that we contribute for the new JA-PL translation data set are contained in this directory. These are distributed as .json files.

The original Japanese annotations are distributed as .xml files. We convert these to .json using ```xml_operations.ipynb```.

The resulting ```LoveHina_volX.json``` files contain the original Japanese anotations for the respective volumes of the manga, as well as the Polish annotations we provide. 

```LoveHina_volX_fixed_order.json``` files, following the reading order panel by panel. 

You can also generate these yourself should you wish, using the provided ```love_hina_modif.ipynb``` notebook, which contains the code for the conversion. 

If you need the annotations to be split by page instead, use ```love_hina_modif_split_page.ipynb```.
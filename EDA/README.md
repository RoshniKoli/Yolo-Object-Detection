# Exploratory Data Analysis
The image annotations in the dataset are in COCO format. YOLO requires the data to be in YOLO specified format, which is, for each bounding box and category pair in the image a corresponding text file with category id, and bounding box parameters normalized by image dimensions is required. 
The format is as follows - 

class_id  x  y  width  height

I have written a few functions to convert the bounding box parameters from COCO to YOLO format, obtain category ids and create txt files for each image. 
Eventually we have 191962 images and corresponding text files for training,     32153 images and corresponding text files for validation and 62629 images in test data.

Besides these folders, the input to darknet for training YOLO needs -

classes.names-file containing the name of each category
train.txt-having full path to each image in the training data
test.txt-having full path to each image of validation data
labelled_data.data having the following 5 lines in order - 

  classes = number of classes 
  
  train = full path to train.txt
  
  test = full path to test.txt
  
  names = full path to classes.names
  
  backup  = location to store the weights file. 
  
These files have been created using corresponding python functions.

Preprocessing file list: 

CV fashion data analysis.ipynb

Creating train.txt and test.txt.ipynb

S3 data transfer.ipynb

I uploaded the dataset on AWS S3 for easy access by EC2 GPU instance.

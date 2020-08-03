# Yolo-Object-Detection
 

Bounding Box detection using YOLO
09.05.2020
─
Roshni Koli
Introduction
Object detection models are extremely powerful — from finding dogs in photos to improving healthcare, training computers to recognize which pixels constitute items unlocks near limitless potential. However, one of the biggest blockers keeping new applications from being built is adapting state-of-the-art, open source, and free resources to custom problems. 

When it comes to deep learning-based object detection, there are three primary object detectors :
R-CNN and their variants
Single Shot Detector (SSDs)
YOLO
R-CNNs are one of the first deep learning-based object detectors and are an example of a two-stage detector.While R-CNNs tend to be very accurate, the biggest problem with the R-CNN family of networks is their speed — they were incredibly slow, obtaining only 5 FPS on a GPU.
To help increase the speed of deep learning-based object detectors, both Single Shot Detectors (SSDs) and YOLO use a one-stage detector strategy. These algorithms treat object detection as a regression problem, taking a given input image and simultaneously learning bounding box coordinates and corresponding class label probabilities. So these tend to be generally less accurate than two-stage detectors, but are incredibly faster.
For this project I am using YOLO, i.e. You Look Only Once. First introduced in 2015 by Redmon et al., their paper, You Only Look Once: Unified, Real-Time Object Detection, details an object detector capable of super real-time object detection, obtaining 45 FPS on a GPU.
YOLO has gone through a number of different iterations, including YOLO9000: Better, Faster, Stronger (i.e., YOLOv2), capable of detecting over 9,000 object detectors and YOLOv3: An Incremental Improvement. For the purpose of this project we will be focusing more on YOLOv3. 

Dataset description

The dataset comprises images from both shops and user posted ones. The train data has 191961 images, validation data has 32123 images and the test data has 62629 images.
The dataset has images with objects from 13 classes and the distribution in test and validation sets is as shown in figure below.


Dataset Source : https://github.com/switchablenorms/DeepFashion2

Yolov3 description 

YOLO stands for You Only Look Once. It's an object detector that uses features learned by a deep convolutional neural network to detect an object.
YOLO makes use of only convolutional layers, making it a fully convolutional network (FCN). It has 75 convolutional layers, with skip connections and upsampling layers. No form of pooling is used, and a convolutional layer with stride 2 is used to downsample the feature maps. This helps in preventing loss of low-level features often attributed to pooling.
Being a FCN, YOLO is invariant to the size of the input image. So I kept the sizes as is since resizing such a dataset on CPU proved computationally expensive on my system.
Before v3, YOLO used to softmax the class scores. However, that design choice has been dropped in v3, and authors have opted for using sigmoid instead. The reason is that Softmaxing class scores assume that the classes are mutually exclusive which is rarely the case in a real-time object detection scenario.

 Reference - https://github.com/AlexeyAB/darknet

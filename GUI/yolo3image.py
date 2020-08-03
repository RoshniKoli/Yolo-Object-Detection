# Detecting Objects on Image with OpenCV deep learning library

# Importing needed libraries
import numpy as np
import cv2
import time


# Defining function for processing given image
def yolo3(path):
    # Reading image with OpenCV library
    image_BGR = cv2.imread(path)

    print('Image shape:', image_BGR.shape)

    # Getting spatial dimension of input image
    h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

    print('Image height={0} and width={1}'.format(h, w))

    # Getting blob from input image

    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    print('Blob shape:', blob.shape)

    """
    Start of:
    Loading YOLO v3 network
    """
    # Create a list of labels from the classes file.
    with open(r'C:\Users\roshn\Documents\DataAnalytics\Springboard\CV_Fashion_dataset\GUI_And_test_images\gui\YOLO data\classes.names') as f:

        labels = [line.strip() for line in f]
        print(labels)

    # Loading trained YOLO v3 Objects Detector with the help of 'dnn' library from OpenCV

    network = cv2.dnn.readNetFromDarknet(r'C:\Users\roshn\Documents\DataAnalytics\Springboard\CV_Fashion_dataset\GUI_And_test_images\gui\YOLO data/yolov3_cvfashion_test.cfg',
                                       r'C:\Users\roshn\Documents\DataAnalytics\Springboard\CV_Fashion_dataset\GUI_And_test_images\gui\YOLO data/yolov3_cvfashion_train.backup')
    print(network)


    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = network.getLayerNames()

    print(layers_names_all)

    # Getting only output layers' names that we need from YOLO v3 algorithm
    layers_names_output =  [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

    print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

    # Setting minimum probability to eliminate weak predictions
    probability_minimum = 0.5

    # Setting threshold for filtering weak bounding boxes with non-maximum suppression
    threshold = 0.3

    # Generating colours for representing every detected object
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    """
    End of:
    Loading YOLO v3 network
    """

    """
    Start of:
    Implementing Forward pass
    """

    # Implementing forward pass with our blob and only through output layers

    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    print()
    print('Objects Detection took {:.5f} seconds'.format(end - start))

    """
    End of:
    Implementing Forward pass
    """

    """
    Start of:
    Getting bounding boxes
    """

    # Preparing lists for detected bounding boxes,obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting classes probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

             # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:

                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    """
    End of:
    Getting bounding boxes
    """

    """
    Start of:
    Non-maximum suppression
    """

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    """
    End of:
    Non-maximum suppression
    """

    """
    Start of:
    Drawing bounding boxes and labels
    """

    # Defining counter for detected objects
    counter = 1

    # Checking if there is at least one detected object after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Showing labels of the detected objects
            print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

            # Incrementing counter
            counter += 1

            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i]].tolist()


            # Drawing bounding box on the original image
            cv2.rectangle(image_BGR, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

    # Comparing how many objects where before and after non-maximum suppression
    print()
    print('Total objects been detected:', len(bounding_boxes))
    print('Number of objects left after non-maximum suppression:', counter - 1)

    """
    End of:
    Drawing bounding boxes and labels
    """

    # Saving resulted image in jpg format 
    cv2.imwrite('result.jpg', image_BGR)

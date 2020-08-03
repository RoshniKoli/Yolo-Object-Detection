

# Creating GUI interface to load and show image
import sys
# QtWidgets to work with widgets
from PyQt5 import QtWidgets
# QPixmap to work with images
from PyQt5.QtGui import QPixmap

# Importing designed GUI in Qt Designer as module
import Yolov3GUI

# Importing YOLO v3 module to Detect Objects on image
from yolo3image import yolo3

"""
Start of:
Main class to connect objects in designed GUI with Yolo code
"""

class MainApp(QtWidgets.QMainWindow, Yolov3GUI.Ui_MainWindow):
    # Constructor of the class
    def __init__(self):

        super().__init__()

        self.setupUi(self)

        # Connecting event of clicking on the button with function that later calls yolo
        self.OpenImage.clicked.connect(self.update_label_object)

    def update_label_object(self):

        # Showing text while image is loading and processing
        self.label.setText('Processing ...')

        image_path = \
            QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image to Open',
                                                  '.',
                                                  '*.png *.jpg *.bmp')

        # Checkpoint
        print(type(image_path))
        print("image path {}".format(image_path[0]))
        print("image type {}".format(image_path[1]))

        image_path = image_path[0]

        yolo3(image_path)

        #  QPixmap class opens result image.This is later paased to the label object for display.
        pixmap_image = QPixmap('result.jpg')

        # Passing opened image to the Label object
        self.label.setPixmap(pixmap_image)

        # resizing Label object according to image height and width
        self.label.resize(pixmap_image.width(), pixmap_image.height())


"""
End of: 
Main class to add functionality of designed GUI
"""


"""
Start of:
Main function
"""


# Defining main function to be run
def main():
    # Initializing instance of Qt Application
    app = QtWidgets.QApplication(sys.argv)

    # Initializing object of designed GUI
    window = MainApp()

    # Showing designed GUI
    window.show()

    # Running application
    app.exec_()


"""
End of: 
Main function
"""

if __name__ == '__main__':
    main()

# Face Mask Detector

### I/O Screenshots

- Mask Detecting 
![Mask Detecting](https://github.com/PrakharJindal/Face-Mask-Detection/blob/main/media/Mask.jpeg)
- No-Mask Detecting
![No-Mask Detecting]("https://github.com/PrakharJindal/Face-Mask-Detection/blob/main/media/No Mask.jpeg")
![No-Mask Detecting]("https://github.com/PrakharJindal/Face-Mask-Detection/blob/main/media/No Mask (Small).jpeg")

- Flow Chart
![Flow Chart]("https://github.com/PrakharJindal/Face-Mask-Detection/blob/main/media/flowchart.jpeg")
![Flow Chart]("https://github.com/PrakharJindal/Face-Mask-Detection/blob/main/media/flowchart (Small).jpeg")

### Introduction

In this peoject we are using a deep learning based model for detecting masks over faces. It will help us to curtail community spread of Coronavirus is presented.

### Using the application:

The website uses OpenCV to access the webcam. The image frames recieved are processed for the model. Faces are detected in the images and face mask are predicted . A box is generated around faces and accuracy percentage is printed along with bounding box.

### Dataset

The dataset includes 2 folders with

### Model Generation

MobileNet ( with Non-Trainable Layers) with extra fully connected layers is being used and a H5 file is generated.

### Running the application locally

- Install python ,django, tensorflow, keras and opencv
- Clone the repo
- Go to the project root dir
- open the root folder of the project in cmd
- run command : python manage.py runserver

# Face Mask Detector

### Links

- Mask Detecting 
![Mask Detecting](https://drive.google.com/file/d/1YXuC39aOwAFLdkuSjLjDY56njkQktuVU/view?usp=sharing)
- !No-Mask Detecting
![No-Mask Detecting](https://drive.google.com/file/d/1dCWLagxLtm85pZLPs0qJX2akIOmTIydK/view?usp=sharing)

- [Flow Chart](https://drive.google.com/file/d/1F21K_6y1qwv14tALZHg8-Zz6hSpj--iH/view?usp=sharing)

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

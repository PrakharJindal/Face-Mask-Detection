# Face Mask Detector

### Links

- [Live Link](https://6145cefcd1832a59bb13e7ae--books-recommender.netlify.app/)
- [IO Screenshot 1](https://drive.google.com/file/d/1vNWRtETDzkAS_MFO5nxBeY2Oa7tGBysu/view?usp=sharing)
- [IO Screenshot 1](https://drive.google.com/file/d/1vNWRtETDzkAS_MFO5nxBeY2Oa7tGBysu/view?usp=sharing)
- [IO Screenshot 2](https://drive.google.com/file/d/1-RHswveBPa7DA7tR7kcKL6cIurM-oybo/view?usp=sharing)
- [IO Screenshot 3](https://drive.google.com/file/d/1cKQ4KZ4xRUDPaJwFXK1pylv_dVzClxzg/view?usp=sharing)
- [IO Screenshot 4](https://drive.google.com/file/d/1KF7QrB8s8Ck2Pn9QBQQ92utKRpiq55cS/view?usp=sharing)
- [IO Screenshot 5](https://drive.google.com/file/d/1cFUs80nu98mrkacU5upS5yzr_uiJTsXY/view?usp=sharing)
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

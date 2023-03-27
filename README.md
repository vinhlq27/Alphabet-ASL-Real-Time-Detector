# Alphabet-ASL-Real-Time-Detection
A real-time alphabet American Sign Language recognition using MediaPipe and CNN.

## Authors

- [@vinhlq27](https://github.com/vinhlq27)


## Introduction

In this project, I propose a real-time alphabet sign language recognition system by using deep learning. The sign language that I used is American Sign Language (ASL). I limit our work to static hand gestures excluding j and z from the classification.

## How To Use

In this project, I use the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) belong to AKASH. 

If you want train the model by yourself, you can use the [train-model.ipynb](https://github.com/vinhlq27/Alphabet-ASL-Real-Time-Classification/blob/main/train-model.ipynb) file.

Or you can download the already trained model at [model_ASL_shuffle.h5](https://github.com/vinhlq27/Alphabet-ASL-Real-Time-Classification/blob/main/model_ASL_shuffle.h5). This model achieves an acceptable recognition rate of 97.18%.

Now, you can download the [Python-Code](https://github.com/vinhlq27/Alphabet-ASL-Real-Time-Classification/tree/main/Python-Code) file to use. The [HandData.py](https://github.com/vinhlq27/Alphabet-ASL-Real-Time-Classification/blob/main/Python-Code/HandData.py) will collect data and send them to the [main.py](https://github.com/vinhlq27/Alphabet-ASL-Real-Time-Classification/blob/main/Python-Code/main.py). The main program will process base on the trained model and the information from camera.

![Result](https://github.com/vinhlq27/Alphabet-ASL-Real-Time-Classification/blob/main/Demo/Demo.gif)



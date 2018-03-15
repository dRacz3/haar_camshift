# image_processing
Anyag:  
http://users.nik.uni-obuda.hu/vamossy/GepiLatas2017/  
http://users.nik.uni-obuda.hu/vamossy/Kepfeldolgozas2017/

## Purpose
This is a midterm project for image processing. The goal of the project is to create a camera controlled mouse with an inhomogen background.
The computer mouse is controlled via moving the hand in front of the camera.
The algorithm developed here is a couple of chained transformations to extract every unneccessary pixel, then letting a cascade object detection algorithm detect fingers. If enough fingers are detected a camshift algorighm is used to track the averaged position of the fingers. If the camshift bounding box would grow/shrink too much, the algorithm drops it and waits for a new detection to start again.

## Dependencies
 - python3.6
 - pyautogui
 - python-opencv
 - numpy


## To run:
```console
$ cd src/
$ python MainProgram.py
```


 ## Known issues:
 - algorithm cannot detect hand close to the edge and throws an error that is catch via a try block currently.
 - no gesture recognition -> Mouse can only be moved, but cannot interact more.
 - sometimes camshift stick to the forearm / head instead of the palm. Fix: wait a few seconds until it drops, then move your hand to the center until it is recognized.
 

# EyeSpeak

Patients with paralysis, Locked-in Syndrome(LIS), Amyotrophic Lateral Sclerosis (ALS) and various other motor neuron diseases are unable to communicate effectively or even communicate at all. These patients face many inconveniences in their daily lives due to their inability to communicate. However, most of these patients’ eye muscles are unaffected, meaning they are able to look anywhere and even blink. In this project, we aim to use computer vision to aid typing on screen for such patients with just the eyes. We can achieve this with gaze detection and blink detection. Based on the eye movement, the virtual keyboard will display a set of characters out of which the user can choose one by blinking. This will be done with openCV, Dlib and convolutional neural networks. The purpose is to help people with speech and movement disorders communicate.

### Libraries
* OpenCV
* Tensorflow
* Dlib

### Methodology
1. Process Video Stream - Every frame is converted to grayscale using openCV for easy computation.
2. Detecting Facial Landmarks - Given an input image, a shape predictor attempts to localize key points of interest along the shape. We'll be using the Dlib 68 points Face Landmark Detection which is  Dlib’s pre-built model for face detection, we only need to import the shape_predictor_68_face_landmarks.dat file. It can be found [here](https://github.com/tzutalin/dlib-android/raw/master/data/shape_predictor_68_face_landmarks.dat).
3. Detect Eyes - The indexes of the 68 coordinates can be visualized on the image below: ![Unknown](https://929687.smushcdn.com/2633864/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg?size=630x508&lossy=1&strip=1&webp=1) 

Since we are only interested in the eye area, we can extract the same from their respective landmarks i.e [37, 38, 39, 40, 41, 42] for the left eye and [43, 44, 45, 46, 47, 48] for the right eye.

4. Blink and gaze detection - The blink detection model checks if the eyes are open or closed. In gaze detection, a gaze can be classified into one of the three classes i.e right, left or center.
5. Interact with the User Interface - Here is how the keyboard works
  * When selecting columns -
    * Look LEFT to move left
    * Loof RIGHT to move right
    * BLINK to select current column
  * Once a column is selected - 
    * Look LEFT to move up
    * Look RIGHT to move down
    * BLINK to type current key 
    * BLINK LEFT EYE to exit the selected column  

![Screenshot 2022-03-25 at 3 45 48 PM](https://user-images.githubusercontent.com/35341758/160101808-7970359b-9a39-4bed-a6fc-cef35df6f1ba.png)

### References 
[Dlib 68 points Face landmark Detection with OpenCV and Python](https://www.studytonight.com/post/dlib-68-points-face-landmark-detection-with-opencv-and-python)

[Facial landmarks with dlib, OpenCV, and Python](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

B.Chandra, Sharon Hajhan L U, Vignesh C P, Sriram R, “Eye Blink Controlled Virtual Interface Using Opencv And Dlib”, European Journal of Molecular & Clinical Medicine ISSN 2515-8260 Volume 07, Issue 08, 2020

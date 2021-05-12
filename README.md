# Self-it
OpenCV and ML project using media-pipe framework
## Human Pose Detection:
Human body pose estimation or detection is the analysis of algorithms, programmes, and pre-trained models that detect the pose of a body made up of components and joint using image-based observations
in computer vision/graphics. Estimation techniques have been increasingly developing in tandem with ever-increasing technical advances, and have made major strides in recent years by incorporating different
aspects of artificial intelligence. The aim of human pose estimation is to identify body parts of a human in photographs or videos automatically.

![alt text](/AiTrainer/blazePose33Points.jpg)
### Application:
To continue, a computer system must be able to accurately discern a person from the surrounding objects, identifying and distinguishing the various parts of the body without defects, to strengthen the
recognition of human poses. This is where deep learning fits in. The machine is fed photographs of various body parts and angles, resulting in a qualified model. We now have an autonomous device
capable of recognizing and separating a human body from its environment after completing the above versions. This is where computer graphics come into play. 

We will detect 33 different landmarks within the human body. This will be the first look at the basic code required to run, after which we will create a module. In this project, our method is to track human
pose by inferring the 33 2D landmarks of a body from a single frame using machine learning (ML). 

Then, using OpenCV and Python, we created an ML guide. We will use the CPU's pose estimation to find the correct points, and then use these points to calculate the desired angles. Then, based on these angles, we can calculate a variety of gestures, including the number of biceps curls.

![alt text](https://developers.google.com/ml-kit/images/vision/pose-detection/warrier2_sketch.png)

# Interface 

To make the interface for this ML module i used [streamlit](https://streamlit.io) library, which is in the code file "WebAPP.py" .

NOTE: Live camera experiment can be applied to it through [streamlit-webrtc](https://pypi.org/project/streamlit-webrtc/) , but currently it's causing error.

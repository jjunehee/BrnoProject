
#Brno Univ. 교환학생 프로젝트

# BrnoProject 

Team number: 4  
Team name: Brnology  
Contact:  
   -Cho Junhee  (jjhjjh1159@kyonggi.ac.kr)  
   -Kim Kyumin (lemon6565@naver.com)  
   -Elvis Juma (elvis.juma@strathmore.edu)  
   -KimHyoeun (dhy04029@gmail.com)  
   
## Topic
Hand gesture recognition for improving online class efficiency
Description of the projects:      <t> Mediapipe, LSTM model  
With the arrival of Covid-19, Most of the students are taking online classes using programs such as Zoom and Google Meet. For that reason,  It is not easy to meet the needs of students. In order to better accommodate their needs,Our team is trying to solve this through hand gesture recognition. Hand gestures will be used to meet the needs of the students in class. For example, if a student asks a question, can’t hear the voice well, or the ppt is difficult to read. Because we plan to recognize students' hands through real-time images, learning will be conducted using webcams. The tools we used are Mediapipe, Numpy and OpenCV. By using this technology, it is expected that students will be able to create a high-quality class atmosphere by increasing their concentration.

# Hand Gesture Recognition
<img src="https://user-images.githubusercontent.com/83155528/182612808-93267711-f29c-4bd1-aa67-a7c10a33efd4.gif" width="260" height="300"/>                               <img src="https://user-images.githubusercontent.com/83155528/182612843-a3c1429a-ab03-47fd-8daa-96ea6c04d341.gif" width="260" height="300"/>
<img src="https://user-images.githubusercontent.com/83155528/182612858-a8fe6663-86be-4c57-9213-f741961f1ad9.gif" width="260" height="300"/>

                                                                                                           
Deep learning based hand gesture recognition using LSTM and MediaPipie.


## Files


Pretrained model in *models* directory.


**create_dataset.py**


Collect dataset from webcam.


**train.ipynp**


Create and train the model using collected dataset.


**test.py**


Test the model using webcam or video.



## Dependency


- Python 3

- TensorFlow 2.4

- sklearn

- numpy

- OpenCV

- MediaPipe


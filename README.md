# Snooze You Loose

Snooze You Loose is a web-based tool to be used on online EdTech platforms for detecting when a student is drowsy. Upon drowsiness detection, the platform raises alerts to increase student attentiveness and promote e-education. 

## Problem

Last year, saw the physical classrooms converting to virtual rooms, thanks to Covid. While the classrooms have altered, students are a bit behind adapting to this change. With increasing ['Zoom Fatigue'](https://www.shethepeople.tv/top-stories/opinion/zoom-fatigue-is-real-students-share-their-experiences-of-online-classes/), drowsiness while attending online classes is a concern. 

The Snooze You Loose tool helps alleviate the issue by detecting if a student is getting sleepy and provides alerts accordingly. The tool gathers data over the EdTech platform courses and provides analytics to the content creators so that they can then improve their courses. 

## Methodology

Using Artificial Intelligence, the Snooze You Looze tool analyzes the student video and detects whether the student is drowsy or alert without compromising user privacy. 

The tool uses [Google's MediaPipe](https://google.github.io/mediapipe/) model for detecting a face in the video stream. As this model runs on the browser (on the user's side), no video frame is ever sent to the server, protecting user privacy. 

<img src="https://raw.githubusercontent.com/Samradh007/Snooze_You_Loose/master/assets/face_mesh.jpg?token=AI2XX7HYMRZQXXGNB2FXWW3AFFRBC" width="250" alt="FaceMesh-Output">

###### Fig 1: FaceMesh MediaPipe Output

Features based on facial landmark positions are calculated and are then sent over to the server. At the server, Deep Learning or Machine Learning models are used to provide classification scores for the facial features. 

<img src="https://raw.githubusercontent.com/Samradh007/Snooze_You_Loose/master/assets/alert_0.jpg?token=AI2XX7ET3N3MGLKYJI3LAH3AFFRMU" width="720" alt="Alert Prediction"> 
<img src="https://raw.githubusercontent.com/Samradh007/Snooze_You_Loose/master/assets/drowsy_0.jpg?token=AI2XX7DI3WOQZ5P2YNCO3XTAFFROI" width="720" alt="Drowsy Prediction">

###### Fig 2: Alert, Drowsy Prediction 

To overcome the lack of a publically available dataset for this task, for the hackathon we collected a dataset and trained models using the same. The dataset is available for use [here](https://drive.google.com/drive/folders/1aryWCejRbGSKL75a4LhmK3QosFfeOklb?usp=sharing). At present, three different models are provided which can be changed at run-time to see the functioning. 

To understand the training process, please training code [readme file](https://github.com/Samradh007/Snooze_You_Loose/blob/master/training/Readme.md).


## WebApp Quick Start

1. Initialize and activate a virtualenv:
  ```
  $ virtualenv --no-site-packages env
  $ source env/bin/activate
  ```

2. Install the dependencies:
  ```
  $ pip install -r requirements.txt
  ```

3. Run the development server:
  ```
  $ python app.py
  ```

4. Navigate to [http://localhost:5000](http://localhost:5000)


## Future Aspects:

Detecting drowsiness and providing alerts to the user is helpful. We believe that engaging the student with small games when detected drowsy state can help better engage the student and reduce the Zoom Fatigue.

We would like to extend the tool and develop it as an API, which can be then integrated into online EdTech platforms. This can enable the analytics feature over the courses and help improve the contents, benefitting both the student and the platforms. 

Additionally, the dataset collected for this task can be further expanded which can help train models achieving higher performance. 

## Acknowledgements:

This project is developed as a part of HackerEarth Hack 2021. We thank the sponsors for providing us with this opportunity to build an impactful application. 


## Contributors:
* [Samradh Agarwal](https://github.com/Samradh007)
* [Abhishek Tandon](https://github.com/Tandon-A)


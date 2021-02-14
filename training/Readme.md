# Training Readme:

For the hackathon, we collected our dataset which covers three classes:
1. **Alert**: User is looking at the laptop while studying or working. 
2. **Drowsy**: User is in a drowsy state, expressed by closing eyes, yawning, rubbing eyes. 
3. **Distracted**: User is distracted, expressed by gazing elsewhere, using a mobile phone. 
The dataset is available for use [here](https://drive.google.com/drive/folders/1aryWCejRbGSKL75a4LhmK3QosFfeOklb?usp=sharing). 

As described in the project methodology, the tool uses Google MediaPipe model to detect and track face landmarks. This model is highly robust and provides features in almost all cases. 

<img src="https://github.com/Samradh007/sleeplessAcademy/blob/main/assets/face2.jpg" width="600" alt="Spectacles viddeo feed"/>
<img src="https://github.com/Samradh007/sleeplessAcademy/blob/main/assets/face1.jpg" width="600" alt="Low quality video feed"/>


###### Figure 1: Spectacles, Poor Quality Video feed

Using the face landmarks, facial features are calculated:
1. Eye Aspect Ratio (EAR): EAR is the ratio of the length of the eyes to the width of the eyes. 
2. Mouth Aspect Ratio (MAR): Similar to the EAR, MAR is the ratio of the length of the mouth to the width of the mouth. 
3. Pupil Circularity (PUC): This feature measures the state of the iris. 
4. Mouth over Eyes (MOE): This is simply the ratio of MAR to the EAR. This feature helps to detect subtle changes. 

To account for the fact that the neutral state of every user is different, these features are first normalized. Feature values obtained in the first 25 frames of the user alert video are then used to normalize the features for the three cases.
At test time, there is a calibration step, which replicates this process and obtains the normalization values for the features. 

The feat_csv.py file process the videos and saves the features in csv file. Model experimentation is then performed using this csv file. 

In our experiments, the Random Forest model performe the best amongst Machine Learning models. (Logistic Regression, Decision Tree, Naive Bayes, Random Forest). While Random Forest trains to achieve good accuracy, the model works on per frame feature basis. In reality, there is a temporal context in these features. We further train LSTM model to learn this temporal context and provide inference over frames. 


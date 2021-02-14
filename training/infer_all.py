import cv2 
import joblib
import mediapipe as mp
import math
import numpy as np 
import os 
import time
import torch


def distance(p1, p2):
    p1 = p1[:2]
    p2 = p2[:2]
    return (((p1 - p2)**2).sum())**0.5

def eye_aspect_ratio(landmarks, eye):
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    return (eye_aspect_ratio(landmarks, left_eye) + \
    eye_aspect_ratio(landmarks, right_eye))/2

def mouth_feature(landmarks):
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3)/(3*D)

def pupil_circularity(landmarks, eye):
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
            distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
            distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
            distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
            distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
            distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
            distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
            distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4*math.pi*area)/(perimeter**2)

def pupil_feature(landmarks):
    return (pupil_circularity(landmarks, left_eye) + \
        pupil_circularity(landmarks, right_eye))/2

def run_face_mp(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z])
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]


        for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar/ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe, image

def calibrate(calib_frame_count=25):
    ears = []
    mars = []
    pucs = []
    moes = []

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(alert_video)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # continue
            break

        ear, mar, puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)
            
            cv2.putText(image, "EAR: %.2f" %(ears[-1]), (int(0.02*image.shape[1]), int(0.07*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "MAR: %.2f" %(mars[-1]), (int(0.27*image.shape[1]), int(0.07*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "PUC: %.2f" %(pucs[-1]), (int(0.52*image.shape[1]), int(0.07*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "MOE: %.2f" %(moes[-1]), (int(0.77*image.shape[1]), int(0.07*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(image, "State: Calibration", (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if len(ears) >= calib_frame_count:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)
    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], \
        [pucs.mean(), pucs.std()], [moes.mean(), moes.std()]

def get_classification(input_data, model, model_type):
    # Manual 
    if model_type == 0:
        model_input = np.array(input_data)[:, 3]
        preds = np.where(np.logical_or(model_input > 4, model_input == -1000), 1, 0)
        return int(preds.sum() >= 13)
    
    # Python RF
    elif model_type == 1:
        model_input = np.array(input_data)
        preds = model.predict(model_input)
        return int(preds.sum() >= 13)

    # PyTorch_LSTM
    elif model_type == 2: 
        model_input = []
        model_input.append(input_data[:5])
        model_input.append(input_data[3:8])
        model_input.append(input_data[6:11])
        model_input.append(input_data[9:14])
        model_input.append(input_data[12:17])
        model_input.append(input_data[15:])
        model_input = torch.FloatTensor(np.array(model_input))
        preds = torch.sigmoid(model(model_input)).gt(0.5).int().data.numpy()
        return int(preds.sum() >= 5)
    
    else:
        raise ValueError('Invalid model type')

def infer(ears_norm, mars_norm, pucs_norm, moes_norm, model, model_type):
    ear_main = 0
    mar_main = 0
    puc_main = 0
    moe_main = 0
    decay = 0.9

    label = None

    input_data = []
    frame_before_run = 0

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(drowsy_video)
    # cap = cv2.VideoCapture(alert_video)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # continue
            break

        ear, mar, puc, moe, image = run_face_mp(image)
        if ear != -1000:
            ear = (ear - ears_norm[0])/ears_norm[1]
            mar = (mar - mars_norm[0])/mars_norm[1]
            puc = (puc - pucs_norm[0])/pucs_norm[1]
            moe = (moe - moes_norm[0])/moes_norm[1]
            if ear_main == -1000:
                ear_main = ear
                mar_main = mar
                puc_main = puc
                moe_main = moe
            else:
                ear_main = ear_main*decay + (1-decay)*ear
                mar_main = mar_main*decay + (1-decay)*mar
                puc_main = puc_main*decay + (1-decay)*puc
                moe_main = moe_main*decay + (1-decay)*moe
        else:
            ear_main = -1000
            mar_main = -1000
            puc_main = -1000
            moe_main = -1000
        
        if len(input_data) == 20:
            input_data.pop(0)
        input_data.append([ear_main, mar_main, puc_main, moe_main])

        frame_before_run += 1
        if frame_before_run >= 15 and len(input_data) == 20:
            frame_before_run = 0
            label = get_classification(input_data, model, model_type)
            print ('got label ', label)
        
        cv2.putText(image, "EAR: %.2f" %(ear_main), (int(0.02*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "MAR: %.2f" %(mar_main), (int(0.27*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "PUC: %.2f" %(puc_main), (int(0.52*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "MOE: %.2f" %(moe_main), (int(0.77*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if label is not None:
            cv2.putText(image, "State %s" %(states[label]), (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()
    cap.release()

def change_model(model_type=0):
    if model_type == 0:
        model = None
    elif model_type == 1:
        model = joblib.load(model_rf_path)
    elif model_type == 2:
        model = torch.jit.load(model_lstm_path)
        model.eval()
    else:
        raise ValueError('Please pass valid model type')
    return model

right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]
iris_right = [473, 474, 475, 476, 477],
iris_left =  [468, 469, 470, 471, 472]
states = ['alert', 'drowsy']
model_types = {
    'Manual' : 0,
    'Python_RF' : 1, 
    'PyTorch_LSTM' : 2
}
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


model_lstm_path = 'C:\\Users\Abhishek\Desktop\proj\hack21\models\clf_lstm_jit6.pth'
model_rf_path = 'C:\\Users\Abhishek\Desktop\proj\hack21\models\RF_drowsy_alert.joblib'

alert_video = 'C:\\Users\Abhishek\Desktop\proj\hack21\data\data\\alert_11_0.mp4'
drowsy_video = 'C:\\Users\Abhishek\Desktop\proj\hack21\data\data\\drowsy_11_0.mp4'

# model_type = model_types['Manual']
# model_type = model_types['Python_RF']
model_type = model_types['PyTorch_LSTM']
model = change_model(model_type)

print ('Starting calibration. Please be in neutral state')
time.sleep(1)
ears_norm, mars_norm, pucs_norm, moes_norm = calibrate()

print ('Starting main application')
time.sleep(1)
infer(ears_norm, mars_norm, pucs_norm, moes_norm, model, model_type)

face_mesh.close()
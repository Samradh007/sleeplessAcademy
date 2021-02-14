import cv2 
import csv
import itertools
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import numpy as np 
import os 
import time


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

def run_face_det(mode='calibrate', ears_norm=None, pucs_norm=None, \
    mars_norm=None, moes_norm=None, video_src=None, label=None):
    if video_src == None:
        print ('provide video_src')
        return None
    if mode != 'calibrate':
        if ears_norm == None or mars_norm == None or \
            pucs_norm == None or moes_norm == None:
            print ('provide normalization')
            return None 
    
    ears = []
    mars = []
    pucs = []
    moes = []

    ear_main = 0
    mar_main = 0
    puc_main = 0
    moe_main = 0


    decay = 0.9

    cap = cv2.VideoCapture(video_src)
    max_frame_count = min(300, 25*(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 25))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            landmarks_positions = []
            for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
                landmarks_positions.append([data_point.x, data_point.y, data_point.z])
            landmarks_positions = np.array(landmarks_positions)
            landmarks_positions[:, 0] *= image.shape[1]
            landmarks_positions[:, 1] *= image.shape[0]

            # for i, point in enumerate(list(itertools.chain(*mouth)) + \
            #     list(itertools.chain(*left_eye)) + list(itertools.chain(*right_eye))):
            #     landmark = landmarks_positions[point]
            #     image = cv2.circle(image, (int(landmark[0]), int(landmark[1])), 1, (0,255,0), 1)

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

            if mode != 'calibrate':
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
            ear = -1000
            mar = -1000
            puc = -1000
            moe = -1000

        if mode == 'calibrate':
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)
        else:
            ears.append(ear_main)
            mars.append(mar_main)
            pucs.append(puc_main)
            moes.append(moe_main)

        cv2.putText(image, "State: %s" %(mode), (int(0.02*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, "Label: %s" %(label), (int(0.52*image.shape[1]), int(0.14*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, "EAR: %.2f" %(ears[-1]), (int(0.02*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "MAR: %.2f" %(mars[-1]), (int(0.27*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "PUC: %.2f" %(pucs[-1]), (int(0.52*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "MOE: %.2f" %(moes[-1]), (int(0.77*image.shape[1]), int(0.07*image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        if mode == 'calibrate': 
            if len(ears) >= 25:
                break
        elif len(ears) >= max_frame_count:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    ears = np.array(ears)
    mars = np.array(mars)
    pucs = np.array(pucs)
    moes = np.array(moes)

    if mode == 'calibrate':
        print ('completed calibration')
        return [ears.mean(), ears.std()], [mars.mean(), mars.std()], \
            [pucs.mean(), pucs.std()], [moes.mean(), moes.std()]
    else:
        print ('run completed')
        return ears, mars, pucs, moes

def add_to_csv(csv_file, pnum, ears, mars, pucs, moes, label):
    data = [[pnum, row, ears[row], mars[row], pucs[row], moes[row], label] \
        for row in range(len(ears))]
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    

left_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
right_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.85)
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# Change to data path 
data_path = 'C:\\Users\Abhishek\Desktop\proj\hack21\data\data'
# Change to output csv file path
data_csv = 'C:\\Users\Abhishek\Desktop\proj\hack21\data\data_new.csv'

header = ['P_ID', 'Frame_Num', 'EAR', 'MAR', 'PUC', 'MOE', 'Label']

if not os.path.exists(data_csv):
    with open(data_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

exts = ['.mp4', '.mov', '.avi']
start_pnum = 0
end_pnum = 10

for pnum in range(start_pnum, end_pnum):
    video_found = False
    alert_video = os.path.join(os.path.join(data_path, 'alert_%d_0' %(pnum)))
    distracted_video = os.path.join(os.path.join(data_path, 'distracted_%d_0' %(pnum)))
    drowsy_video = os.path.join(os.path.join(data_path, 'drowsy_%d_0' %(pnum)))
    for ext in exts:
        alert_path = alert_video + ext
        distracted_path = distracted_video + ext
        drowsy_path = drowsy_video + ext
        if os.path.exists(alert_path) and os.path.exists(distracted_path) and os.path.exists(drowsy_path):
            video_found = True
            break
    if not video_found:
        print ('Video not found for ', pnum)
        continue


    print ('starting calibration for ', pnum)
    time.sleep(1)
    ears_norm, mars_norm, pucs_norm, moes_norm = run_face_det(mode='calibrate', video_src=alert_path, label='alert')
    
    print ('Starting main application')
    time.sleep(1)
    ears_pnum, mars_pnum, pucs_pnum, moes_pnum = run_face_det(mode='main', \
        ears_norm=ears_norm, mars_norm=mars_norm, pucs_norm=pucs_norm, moes_norm=moes_norm, \
        video_src=alert_path, label='alert')
    print (len(ears_pnum), len(mars_pnum), len(pucs_pnum), len(moes_pnum))
    add_to_csv(data_csv, pnum, ears_pnum, mars_pnum, pucs_pnum, moes_pnum, 'alert')

    time.sleep(1)
    ears_pnum, mars_pnum, pucs_pnum, moes_pnum = run_face_det(mode='main', \
        ears_norm=ears_norm, mars_norm=mars_norm, pucs_norm=pucs_norm, moes_norm=moes_norm, \
        video_src=distracted_path, label='distracted')
    print (len(ears_pnum), len(mars_pnum), len(pucs_pnum), len(moes_pnum))
    add_to_csv(data_csv, pnum, ears_pnum, mars_pnum, pucs_pnum, moes_pnum, 'distracted')

    time.sleep(1)
    ears_pnum, mars_pnum, pucs_pnum, moes_pnum = run_face_det(mode='main', \
        ears_norm=ears_norm, mars_norm=mars_norm, pucs_norm=pucs_norm, moes_norm=moes_norm, \
        video_src=drowsy_path, label='drowsy')
    print (len(ears_pnum), len(mars_pnum), len(pucs_pnum), len(moes_pnum))
    add_to_csv(data_csv, pnum, ears_pnum, mars_pnum, pucs_pnum, moes_pnum, 'drowsy')


face_mesh.close()

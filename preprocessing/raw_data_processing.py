import os
import shutil
import re
import subprocess
import re
import math
import PSpincalc as sp
import numpy as np
import json
import cv2
import mediapipe as mp
import time

class PreProcessing:
    def __init__(self, config):
        self.config = config

    def video_rename(self, path_to_dir, folder, r):
        listAU = []
        for file in os.listdir(os.path.join(path_to_dir, folder, r, self.config['face_folder'])):
            if ('mp4' in file):
                listAU.append(file)
        for i in range(len(listAU)):
            os.rename(os.path.join(path_to_dir, folder, r, self.config['face_folder'], listAU[i]),
                      os.path.join(path_to_dir, folder, r, self.config['face_folder'], self.config['video_sequence'][i] + '_' + folder + '.mp4'))

    def video_to_FeatureExtraction(self, path_to_dir, folder, r, path_to_openface):
        path = os.path.join(path_to_dir, folder, r, self.config['face_folder'])
        listAU = []
        for file in os.listdir(path):
            if 'mp4' in file:
                if (file.split('.')[1] == 'mp4'):
                    if (file.split('_')[0] in ['p2', 'p3', 'p5', 'p11']):
                        listAU.append(file)

        for i in range(len(listAU)):
            subprocess.run(os.path.join(path_to_openface,'FeatureExtraction.exe') + ' -f ' + os.path.join(path, listAU[i]) + ' -out_dir ' + path + ' -aus')

    def video_to_frames(self, path_to_video, path_output_dir, number):
        vidcap = cv2.VideoCapture(path_to_video)
        count = 0
        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                if count % number == 0:
                    print(os.path.join(path_output_dir, '%d.png') % count)
                    cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
                count += 1
            else:
                break
        #cv2.destroyAllWindows()
        vidcap.release()

    def faceImage(self, path_to_dir, folder, r):
        path = os.path.join(path_to_dir, folder, r, self.config['face_folder'])
        for file in os.listdir(path):
            if 'mp4' in file:
                if file.split('_')[0] in ['p12', 'p13']:
                    folder_frame = os.path.join(path, file.split('_')[0] + '_' + file.split('_')[1])
                    os.mkdir(folder_frame)
                    self.video_to_frames(os.path.join(path, file), folder_frame, self.config['every_frames'])
                    # print(folder_frame,file)
                if file.split('_')[0] == 'p1':
                    folder_frame = os.path.join(path, 'p1')
                    os.mkdir(folder_frame)
                    self.video_to_frames(os.path.join(path, file), folder_frame, 60)

    def video_to_FaceLandmarkImg(self, path_to_dir, folder, r, path_to_openface):
        path = os.path.join(path_to_dir, folder, r, self.config['face_folder'])
        task = ['p1', 'p12_1', 'p12_2', 'p12_3', 'p12_4', 'p12_5', 'p12_6', 'p13_1', 'p13_2', 'p13_3', 'p13_4', 'p13_5', 'p13_6']
        for j in range(len(task)):
            subprocess.run(os.path.join(path_to_openface, 'FaceLandmarkImg.exe') + ' -fdir ' + os.path.join(path, task[j]) + ' -out_dir ' + os.path.join(path, task[j]) + ' -aus')

    def send_lmt_to_LM(self, path_to_dir, folder, r, path_to_RecordPlaybackSample):
        path = os.path.join(path_to_dir, folder, r)
        if not os.path.isdir(os.path.join(path, self.config['hand_folder_LM'])):
            os.mkdir(os.path.join(path, self.config['hand_folder_LM']))
        folders = os.listdir(path)
        ms = [m for m in folders if re.findall(r'm\d+', m)]
        for m in ms:
            if os.path.isdir(os.path.join(path, m)):
                files = os.listdir(os.path.join(path, m))
                for file in files:
                    if file.split('.')[1] == 'lmt':
                       subprocess.run(os.path.join(path_to_RecordPlaybackSample,'RecordPlaybackSample.exe') + ' ' + os.path.join(path, m, file) + " " + os.path.join(path, self.config['hand_folder_LM'], file.split('.')[0] + '_' + m + '_' + folder + '.txt'))


    def LMJson(self, path_to_dir, folder, r):
        path = os.path.join(path_to_dir, folder, r, self.config['hand_folder_LM'])
        for file in os.listdir(path):
            if "txt" in file:
                print('file', file)
                f = open(os.path.join(path, file), 'r', encoding='utf-8', errors='ignore')
                d = {}
                res = []
                FINGER = ["THUMB_MCP", "THUMB_PIP", "THUMB_DIP", "THUMB_TIP",
                          "FORE_MCP", "FORE_PIP", "FORE_DIP", "FORE_TIP",
                          "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
                          "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
                          "LITTLE_MCP", "LITTLE_PIP", "LITTLE_DIP", "LITTLE_TIP"]
                for line in f:
                    print(line)
                    if line.find("frame") >= 0:
                        count = 0
                        a1 = line.lstrip(' ')
                        a2 = a1.rstrip(' ')
                        a3 = a2.split('.')
                        a4 = a3[0].split()

                        frame_number = int(a4[1])

                    if line.find("Confidence") >= 0:
                        confidence = float(line.split("Confidence")[1].split()[0])
                        id_frame = int(line.split("id_img")[1].split()[0])
                        visible_time = int(line.split("visible_time")[1].split()[0])
                        pinch_distance = float(line.split("pinch_distance")[1].split()[0])
                        pinch_strength = float(line.split("pinch_strength")[1].split()[0])
                        grab_angle = float(line.split("grab_angle")[1].split()[0])
                        grab_strength = float(line.split("grab_strength")[1].split()[0])
                        palm_width = float(line.split("palm_width")[1].split()[0])
                        timestamp = int(line.split("timestamp")[1].split()[0])
                        frame_id = int(line.split("img_id")[1].split()[0])
                        tracking_frame_id = int(line.split("tracking_img_id")[1].split()[0])
                        framerate = float(line.split("fps")[1].split()[0])
                        version = line.split("version")[1].split()[0]

                        d.update({'info':{"confidence": confidence,
                                  "id_frame": id_frame,
                                  "visible_time": visible_time,
                                  "pinch_distance": pinch_distance,
                                  "pinch_strength": pinch_strength,
                                  "grab_angle": grab_angle,
                                  "grab_strength": grab_strength,
                                  "palm_width": palm_width,
                                  "timestamp": timestamp,
                                  "frame_id": frame_id,
                                  "tracking_frame_id": tracking_frame_id,
                                  "framerate": framerate,
                                  "version":version}
                                  })

                    if line.find("Hand id") >= 0:
                        count = 0
                        result = re.search(r'\((.*?)\)', line).group(1)
                        result = result.split()
                        print(result)
                        result_x = float(result[0].split(',')[0])
                        result_y = float(result[1].split(',')[0])
                        result_z = float(result[2].split(',')[0])
                        result_w = float(result[2].split(',')[1])
                        result_wx = float(result[3].split(',')[0])
                        result_wy = float(result[4].split(',')[0])
                        result_wz = float(result[5])
                        quaternion = np.array([result_w * (math.sqrt(result_wz ** 2 + result_wy ** 2 + result_wx ** 2 + result_w ** 2)),result_wz, result_wy, result_wx])
                        ea = sp.Q2EA(quaternion, EulerOrder="zyx", ignoreAllChk=True)[0]
                        # res = (-ea[2] / (2 * 3.14)) * 360
                        if line.find("right hand") >= 0:
                            resCentre = (-ea[2] / (2 * 3.14)) * 360
                            if (resCentre < -150): #-150
                                resCentre = 180 #resCentre+180 #180
                            d2 = {'X': result_x, 'Y': result_y, 'Z': result_z, 'W': result_w, 'Wx': result_wx,
                                  'Wy': result_wy,
                                  'Wz': result_wz, 'Angle': resCentre}
                            d.update({'CENTRE': d2})
                            hand = "right hand"
                        if line.find("left hand") >= 0:
                            hand = "left hand"
                            resCentre = (ea[2] / (2 * 3.14)) * 360
                            if (resCentre < -150): #-150
                                resCentre = 180 #resCentre+180 #180
                            d2 = {'X': result_x, 'Y': result_y, 'Z': result_z, 'W': result_w, 'Wx': result_wx,
                                  'Wy': result_wy,
                                  'Wz': result_wz, 'Angle': resCentre}
                            d.update({'CENTRE': d2})

                    if line.find("bone with position") >= 0:
                        count = count + 1
                        result = re.search(r'\((.*?)\)', line).group(1)
                        result = result.split()
                        result_x1 = float(result[0].split(',')[0])
                        result_y1 = float(result[1].split(',')[0])
                        result_z1 = float(result[2].split(',')[0])
                        result_x2 = float(result[3].split(',')[0])
                        result_y2 = float(result[4].split(',')[0])
                        result_z2 = float(result[5].split(',')[0])
                        result_x3 = float(result[6].split(',')[0])
                        result_y3 = float(result[7].split(',')[0])
                        result_z3 = float(result[8].split(',')[0])
                        result_w3 = float(result[9])
                        quaternionFinger = np.array([result_w3 * (math.sqrt(result_z3 ** 2 + result_y3 ** 2 + result_x3 ** 2 + result_w3 ** 2)),result_z3, result_y3, result_x3])
                        eaF = sp.Q2EA(quaternionFinger, EulerOrder="zyx", ignoreAllChk=True)[0]
                        resFinger = (eaF[2] / (2 * 3.14)) * 360

                        d2 = {'X1': result_x1, 'Y1': result_y1, 'Z1': result_z1, 'X2': result_x2, 'Y2': result_y2,
                              'Z2': result_z2, 'X3': result_x3, 'Y3': result_y3, 'Z3': result_z3, 'W': result_w3,
                              'Angle': resFinger}
                        d.update({FINGER[count - 1]: d2})
                        dict_finger = {hand: d}

                        if (count == 20):
                            dict_finger.update({'frame': frame_number})
                            res.append(dict_finger)
                            d = {}

                with open(os.path.join(path, file.split('.')[0] + '.json'), 'w') as outfile:
                    json.dump(res, outfile)
                f.close()

    def face_processing(self, path_to_dir, folder, r):
        self.video_rename(path_to_dir, folder, r)
        self.video_to_FeatureExtraction(path_to_dir, folder, r, self.config['path_to_openface'])
        self.faceImage(path_to_dir, folder, r)
        self.video_to_FaceLandmarkImg(path_to_dir, folder, r, self.config['path_to_openface'])

    def hand_processing_LM(self, path_to_dir, folder, r):
        self.send_lmt_to_LM(path_to_dir, folder, r, self.config['path_to_lmt_exe'])
        self.LMJson(path_to_dir, folder, r)

    def MPjson(self, path_to_dir, folder, r): # функция для формирования json по анологии с leapmotion
        translate_hand = {'Left': 'right hand', 'Right': 'left hand'} #из-за того что видосы на фронталку, приходится руку вот так инвертировать
        POINTS = ['CENTRE', 'THUMB_MCP', 'THUMB_PIP', 'THUMB_DIP', 'THUMB_TIP', 
          'FORE_MCP', 'FORE_PIP', 'FORE_DIP', 'FORE_TIP', 
          'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP', 
          'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP', 
          'LITTLE_MCP', 'LITTLE_PIP', 'LITTLE_DIP', 'LITTLE_TIP']
        path_to_out = os.path.join(path_to_dir, folder, r, self.config['hand_folder_MP'])
        if not os.path.exists(path_to_out):
            os.makedirs(path_to_out)
        path = os.path.join(path_to_dir, folder, r)
        folders = os.listdir(path)
        ms = [m for m in folders if re.findall(r'm\d+', m)]
        print(ms, folder)
        for m in ms:
            if os.path.isdir(os.path.join(path, m)):
                files = os.listdir(os.path.join(path, m))
                print(files)
                for file in files:
                    if file.split('.')[1] in ['mp4', 'mkv', 'MOV', 'mov']:
                        res = []
                        path_to_file = os.path.join(path,m, file)
                        mp_hands = mp.solutions.hands
                        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
                        print('\n')
                        cap = cv2.VideoCapture(path_to_file)
                        print(f'\n\nProcessing file: {path_to_file}\n')
                        count_frame = 1
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            dict_points = {}
                            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                            if results.multi_hand_landmarks:
                                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                                    hand_label = handedness.classification[0].label
                                    for id, landmark in enumerate(hand_landmarks.landmark):

                                        if id==0:
                                            cords = {
                                                    "X": round(landmark.x, 3), 
                                                    "Y": round(landmark.y, 3), 
                                                    "Z": round(landmark.z, 3),
                                                    "X1": round(landmark.x, 3), 
                                                    "Y1": round(landmark.y, 3), 
                                                    "Z1": round(landmark.z, 3),
                                                    "W": 0, 
                                                    "Wx": 0, 
                                                    "Wy": 0, 
                                                    "Wz": 0, 
                                                    "Angle": 0
                                                    }
                                        else:
                                            cords = {"X1": round(landmark.x, 3), 
                                                    "Y1": round(landmark.y, 3), 
                                                    "Z1": round(landmark.z, 3),
                                                    "X": round(landmark.x, 3), 
                                                    "Y": round(landmark.y, 3), 
                                                    "Z": round(landmark.z, 3),
                                                    "W": 0, 
                                                    "Angle": 0}
                                        dict_points.update({POINTS[id]:cords})
                                res.append({translate_hand[hand_label]:dict_points, 'frame':count_frame})
                                print("frame:", count_frame, end="\r", flush=True)
                                count_frame += 1
                        print("frame:", count_frame, end="\n")
                        cap.release()
                        with open(os.path.join(path_to_out, file.split('.')[0]+f'_{m}_' + folder + '.json'), 'w') as outfile:
                            json.dump(res, outfile)


    def hand_processing_MP(self, path_to_dir, folder, r):
        '''
        написать функцию препроцессинга данных через MediaPipe (по примеру hand_processing_LM),
        где запись выодных данных производится в файл json с аналогичной структурой
        '''
        self.MPjson(path_to_dir, folder, r)

    def processing(self):
        for dataset in ['PD', 'HEALTHY', 'STUDENT']:
            path_to_dir = self.config[dataset]['path_to_directory']
            folders = [self.config[dataset]['id_name']+str(number) for number in self.config[dataset]['numbers']]
            for folder in folders:
                for r in os.listdir(os.path.join(path_to_dir, folder)):
                    if self.config['process_face']:
                        self.face_processing(path_to_dir, folder, r)
                    if self.config['process_hand_LM']:
                        self.hand_processing_LM(path_to_dir, folder, r)
                    if self.config['process_hand_MP']:
                        self.hand_processing_MP(path_to_dir, folder, r)


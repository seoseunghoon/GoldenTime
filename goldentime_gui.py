import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import time
import threading
#import pyautogui
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.compat.v1 as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
import winsound as sd



physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import logging
#from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('check_crash', True, 'check object box crash')
flags.DEFINE_boolean('run_video', False, 'run')

class Example(QWidget):
    # Definition of the parameters
    max_cosine_distance = None
    nn_budget = None
    nms_max_overlap = None
    
    # initialize deep sort
    model_filename = None
    encoder = None
    # calculate cosine distance metric
    
    # initialize tracker
    tracker = None

    # load configuration for object detector
    config = None
    session = None
    STRIDES = None
    ANCHORS = None
    NUM_CLASS = None
    XYSCALE = None
    input_size = None
    video_path = None
    saved_model_loaded = None
    infer = None
    vid = None
    out = None
    cpt = None
    human_c = 0
    car_c = 0
    # get video ready to save locally if flag is set
    frame_num = None
    human_table = [[]]
    
    # while video is running

    def __init__(self):
        super().__init__()
        
        
        self.setWindowTitle("CCTV")
        self.setGeometry(150,150,10 +800 +10 +600 +15,580)
        
        # Definition of the parameters
        self.max_cosine_distance = 0.4
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        
        # initialize deep sort
        self.model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
        # calculate cosine distance metric
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        # initialize tracker
        self.tracker = Tracker(self.metric)
        
        # load configuration for object detector
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = InteractiveSession(config=self.config)
        self.STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.input_size = FLAGS.size
        self.video_path = FLAGS.video
        
    
        self.saved_model_loaded = tf.saved_model.load_v2(FLAGS.weights, tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']

        # begin video capture
        

        self.out = None
        
        # get video ready to save locally if flag is set
        self.frame_num = 0
        # while video is running
        self.initUI()
        
    def initUI(self):
        
        try:
            self.cpt= cv2.VideoCapture(int(self.video_path))
            #cam사용.
            #self.cpt = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except:
            self.cpt = cv2.VideoCapture(self.video_path)
        
        
        self.fps=24
        _,self.img_o = self.cpt.read()
        self.img_o = cv2.cvtColor(self.img_o,cv2.COLOR_RGB2GRAY)
        cv2.imwrite('img_o.jpg',self.img_o)
        
        
        self.cnt=0

        self.frame = QLabel(self)
        self.frame.resize(800,500)
        self.frame.setScaledContents(True)
        self.frame.move(10,10) 
        
        
        self.logtitle=QLabel(self)
        self.logtitle.resize(200,25)
        self.logtitle.move(10+800+15,10)
        self.logtitle.setText("충돌 내역")
        self.logtitle.setFont(QFont("맑은 고딕",10,QFont.Bold))
        
        
        
        self.table=QTableWidget(self)
        self.table.resize(600,525)
        self.table.move(10+800+15,10+25+5)
        self.table.setRowCount(0)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['날짜','시각','차량id','사람id','위험등급'])
        self.table.setColumnWidth(0,self.table.width()*2/10)
        self.table.setColumnWidth(1,self.table.width()*2/10)
        self.table.setColumnWidth(2,self.table.width()*2/10)
        self.table.setColumnWidth(3,self.table.width()*2/10)
        self.table.setColumnWidth(4,self.table.width()*2/10)
        
    
        
        self.btn_on2 = QPushButton("영상 출력",self)
        self.btn_on2.resize(390,50)
        self.btn_on2.move(10,10+500+5)
        self.btn_on2.setStyleSheet("background-color:#C8C8C8;"  "font-weight: bold;")
        self.btn_on2.clicked.connect(self.start)

        self.btn_off = QPushButton("중지",self)
        self.btn_off.resize(395,50)
        self.btn_off.move(20+100+5+100+5+185,10+500+5)
        self.btn_off.setStyleSheet("background-color:#C8C8C8;"  "font-weight: bold;")
        self.btn_off.clicked.connect(self.stop)

        
        self.msg=QLabel(self)
        self.msg.setText("사람 : %d   차: %d  " % (self.human_c, self.car_c))
        self.msg.resize(400,40)
        self.msg.move(10,10)
        self.msg.setFont(QFont("맑은 고딕",20,QFont.Bold))
        self.msg.setStyleSheet("Color : Blue")

        self.prt=QLabel(self)
        self.prt.resize(200,25)
        self.prt.move(10+100+5+100+5+100+15+100+15+25+100-20,10+500+15)
        self.prt.setText("")
        self.prt.setFont(QFont("맑은 고딕",10,QFont.Bold))
        self.prt.setStyleSheet("Color : Green")

        self.show()
    



    def start(self):
        for n in range(6) : #위험등급 5까지제한.
            self.human_table.append([]) 
        FLAGS.run_video = False #정지변수
        while True:
            return_value, frame = self.cpt.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            #객체 카운팅 변수
            self.human_c = 0
            self.car_c = 0
            
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (self.input_size, self.input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set
            if FLAGS.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([self.input_size, self.input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([self.input_size, self.input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = self.infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            #NMS(Non Maxmimum Suppressions)
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )
                

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            #allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            # 인식 class
            allowed_classes = ['person','car', 'bus', 'truck']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                #test
                class_name_temp = class_names[class_indx]
                #test
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = self.encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

           
            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)
            check_num = 0
            car_table = list()
            n = 0
            # update tracks
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                global car_id
                global human_id
                global grade
                global per
                #박스 크기축소.
                per = 90 #%로 축소.
                w = int(bbox[2])-int(bbox[0])
                h = int(bbox[3])-int(bbox[1])
                control_size_x = w*0.5 * (1-(per*0.01))
                control_size_y = h*0.5 * (1-(per*0.01))
                bbox[0] += control_size_x
                bbox[1] += control_size_y
                bbox[2] -= control_size_x
                bbox[3] -= control_size_y
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                
                flag_overlap = True
                if FLAGS.check_crash:
                    if class_name == str("car") or class_name == str("bus") or  class_name == str("truck"):
                        car_table.append(str(track.track_id))
                        car_table.append(bbox)
                        self.car_c += 1
                    
                    if class_name == str("person") :
                        self.human_c += 1
                        #사람 위험도 테이블에 추가, 중복(아이디가 table에 존재하는 경우)제외
                        for i in self.human_table :
                            
                            for j in i:
                                if j == str(track.track_id) :
                                    flag_overlap = False
                                    
                        if flag_overlap:
                            self.human_table[0].append(str(track.track_id))
                            
                            
                        n += 1 #몇번째 사람.
                        
                        if len(car_table)>1 :
                            #human_table 2차원 리스트, 저장 위치로 id구별
                            #human_table은 프레임이 끝나도 유지
                            #list에서 같은 id를 찾고, 위험성체크.

                            for h in range(0,len(car_table)-1,2):
                                bbox_temp = list()
                                bbox_temp.append(car_table[h+1][0])
                                bbox_temp.append(car_table[h+1][1])
                                bbox_temp.append(car_table[h+1][2])
                                bbox_temp.append(car_table[h+1][3])
                                #차안 사람 충돌 제외
                                if bbox[0]>= bbox_temp[0] and bbox[2]<= bbox_temp[2] and bbox[1]>= bbox_temp[1] and bbox[3]<= bbox_temp[3]:
                                    pass
                                #충돌확인
                                elif bbox[2] >= bbox_temp[0] and bbox[0] <= bbox_temp[2] and bbox[3] >= bbox_temp[1] and bbox[1] <= bbox_temp[3]:
                                    #print(track.track_id,'번 째 사람 ',car_table[h] ,'차량과 충돌')
                                    #충돌, 위험등급 증가, human_table 재저장
                                    #위험등급이 체크
                                    dangerous_grade = 0
                                    
                                
                                    for a in range(len(self.human_table)): #최대 위험등급
                                        for b in self.human_table[a]:
                                            if  b == str(track.track_id):
                                                if a>3 :
                                                    dangerous_grade = a
                                                    color = 0xFF0000 #위험등급 도달한경우 박스 색깔 변경.
                                                    self.beepsound() #beep음
                                                    print('위험인물 식별. 확인바람.')
                                                    
                                                else :
                                                    dangerous_grade = a
                                                    
                                    #등급업
                                    if dangerous_grade <4 :
                                        #print(self.human_table)
                                        del self.human_table[dangerous_grade][self.human_table[dangerous_grade].index(str(track.track_id))]#자기자신 삭제
                                        self.human_table[dangerous_grade+1].append(str(track.track_id))   #다음 등급에 추가.
    
                                    car_id = str(car_table[h])
                                    human_id = str(track.track_id)
                                    grade = str(dangerous_grade+1)
                                    self.addRow()            
                
                
                

                
            # draw bbox on screen
                
                #위험 객체 굵기 강조.
                if color == 0xFF0000 :
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 4)
                else :
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                #cv2.rectangle(frame, (int(bbox[0]) + int(control_size_x), int(bbox[1])+ int(control_size_y)), (int(bbox[2])- int(control_size_x), int(bbox[3])- int(control_size_y)), color, 2)
                #cv2.rectangle(frame, (int(bbox[0]) + int(control_size_x), int(bbox[1]-30)+ int(control_size_y)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17- int(control_size_x), int(bbox[1])- int(control_size_y)), color, -1)
                
            # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                check_num += 1
                #
                
            #nextFrame car_table 초기화
            car_table.clear()
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            #result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.msg.setText("사람 : %d   차: %d  " % (self.human_c, self.car_c))

            #if not FLAGS.dont_show:
            #    cv2.imshow("Output Video", result)
            
            # if output flag is set, save video file
            #if FLAGS.output:
            #    self.out.write(result)
            
            
            self.codec = cv2.VideoWriter_fourcc(*'XVID')
            width = self.cpt.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cpt.get(cv2.CAP_PROP_FRAME_HEIGHT)

            #result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img =QImage(result, width, height, QImage.Format_RGB888) #크기정하기 조정중
            pix=QPixmap.fromImage(img)
            
            self.frame.setPixmap(pix)
            
            if FLAGS.run_video : 
                self.frame.clear()
                break
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        #cv2.destroyAllWindows()
    
    #비프소리조정.
    def beepsound(self):
        fr = 1000    # range : 37 ~ 32767
        du = 1000     # 1000 ms ==1second
        sd.Beep(fr, du)
            
    def stop(self):
        #self.prt.setText("대기 중입니다.")
        self.frame.setPixmap(QPixmap.fromImage(QImage()))
        FLAGS.run_video = True

    #테이블 추가.
    def addRow(self):
        rowPosition=self.table.rowCount()
        self.table.insertRow(rowPosition)
        t=time.localtime()

        self.table.setItem(rowPosition , 0, QTableWidgetItem("{}-{}-{}".format(t.tm_year, t.tm_mon, t.tm_mday)))
        self.table.setItem(rowPosition , 1, QTableWidgetItem("{}:{}:{}".format(t.tm_hour,t.tm_min,t.tm_sec)))
        self.table.setItem(rowPosition , 2, QTableWidgetItem(car_id))
        self.table.setItem(rowPosition , 3, QTableWidgetItem(human_id))
        self.table.setItem(rowPosition , 4, QTableWidgetItem(grade))
        res=""
        detail=""

        #충돌 테이블 색깔.
        if int(grade) > 4 :
            for c in range(5):
                myitem = self.table.item(rowPosition,c)
                myitem.setBackground(QColor(250,0,102))


    
    
if __name__ == '__main__':
    try:
        app=QApplication(sys.argv)
        w=Example()
        sys.exit(app.exec_())
        #app.run(main)
    except SystemExit:
        pass

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
flags.DEFINE_boolean('run_video', False, 'check object box crash')

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
    # get video ready to save locally if flag is set
    frame_num = None
    human_table = [[]]
    
    # while video is running

    def __init__(self):
        super().__init__()
        
        
        self.setWindowTitle("CCTV")
        self.setGeometry(150,150,10 +800 +10 +600 +15,580)
        
        #self.setGeometry(150,150,650,540)
        
        
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
            #self.cpt = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            #self.cpt= cv2.VideoCapture(0)
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
        self.table.resize(600,520)
        self.table.move(10+800+15,10+25+5)
        self.table.setRowCount(0)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['날짜','시각','차량id','사람id','위험등급'])
        self.table.setColumnWidth(0,self.table.width()*1/10)
        self.table.setColumnWidth(1,self.table.width()*1/10)
        self.table.setColumnWidth(2,self.table.width()*2/10)
        self.table.setColumnWidth(3,self.table.width()*3/10)
        self.table.setColumnWidth(4,self.table.width()*3/10)
        
        
    
        
        self.btn_on2 = QPushButton("카메라 연결",self)
        self.btn_on2.resize(200,50)
        self.btn_on2.move(10,10+500+15)
        self.btn_on2.setStyleSheet("background-color:#8FFFF7;"  "font-weight: bold;")
        #self.btn_on2.clicked.connect(self.start2)
        self.btn_on2.clicked.connect(self.test)

        self.btn_off = QPushButton("중지",self)
        self.btn_off.resize(200,50)
        self.btn_off.move(10+100+5+100+5,10+500+15)
        self.btn_off.setStyleSheet("background-color:#8FFFF7;"  "font-weight: bold;")
        self.btn_off.clicked.connect(self.stop)

        
        self.msg=QLabel(self)
        self.msg.setText("상태 메세지: ")
        self.msg.resize(100,25)
        self.msg.move(10+100+5+100+5+100+15+100+15+15,10+500+15)
        
        self.prt=QLabel(self)
        self.prt.resize(200,25)
        self.prt.move(10+100+5+100+5+100+15+100+15+25+100-20,10+500+15)
        self.prt.setText("대기 중입니다.")
        self.prt.setFont(QFont("맑은 고딕",10,QFont.Bold))
        self.prt.setStyleSheet("Color : Green")

        self.show()
    



    def test(self):
        FLAGS.run_video = False
        while True:
            return_value, frame = self.cpt.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            
            
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
            allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            #allowed_classes = ['person','car']

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
            if FLAGS.count:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
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

            #
            #print(scores)
            #print(scores[0])
            #print("classes"+ classes)
            #print(boxs)
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
                #크기축소.
                per = 40 #50%로 축소.
                w = int(bbox[2])-int(bbox[0])
                h = int(bbox[3])-int(bbox[1])
                control_size_x = w*0.5 * (1-(per*0.01))
                control_size_y = h*0.5 * (1-(per*0.01))
                bbox[0] += control_size_x
                bbox[1] += control_size_y
                bbox[2] -= control_size_x
                bbox[3] -= control_size_y

                
                flag_overlap = True
                if FLAGS.check_crash:
                    if class_name == str("car") :
                        car_table.append(str(track.track_id))
                        car_table.append(bbox)
    
                    
                    if class_name == str("person") :
                        #사람 위험도 테이블에 추가 #중복제외 필요. #아이디가 없으면 추가. #첫추가는 무조건.
                        for i in self.human_table :
                            
                            for j in i:
                                
                                
                                if j == str(track.track_id) :
                                    flag_overlap = False
                                    
                        if flag_overlap:
                            self.human_table[0].append(str(track.track_id))
                            
                            
                        
                        
                        #if x==car_table.get(id+track.track.id,x):#일치하는 id가 없으면 x리턴
                        #    car_table[id+track.track.id] = 0

                        n += 1 #몇번째 사람.
                        
                        if len(car_table)>1 :
                            #track.track_id를 다른곳에 저장을 시켜야.
                            #궁금한점. 한번 id지정후에 계속 실행을 했을때, id가 다시 사용되는지,아이디가 어디까지진행되는지.
                            #id는 +1되면서 새로운 인물추적시작.
                            #1. 데이터베이스로 참조.
                            #2. track에 속성을 추가해서 위험관리를 시킨다.
                            #3. 개별 human_table(hash형태)만들어서 관리를 한다.
                            #       human_table은 프레임이 끝나도 지속.
                            #       list에서 같은 id를 찾고, 다음을 참고하여, 위험성체크.
                            #   첫 발견할때 딕셔너리 추가하는건 맞는데, 매번추가할수는 없으니 있나확인하고 없으면 추가.
                            #   id가 그냥 숫자라서 검색할때 겹칠수있음. id1

                        
                            for h in range(0,len(car_table)-1,2):
                                bbox_temp = list()
                                bbox_temp.append(car_table[h+1][0])
                                bbox_temp.append(car_table[h+1][1])
                                bbox_temp.append(car_table[h+1][2])
                                bbox_temp.append(car_table[h+1][3])
                                #차안 사람 제외
                                if bbox[0]>= bbox_temp[0] and bbox[2]<= bbox_temp[2] and bbox[1]>= bbox_temp[1] and bbox[3]<= bbox_temp[3]:
                                    print("", end='')

                                #충돌확인
                                elif bbox[2] >= bbox_temp[0] and bbox[0] <= bbox_temp[2] and bbox[3] >= bbox_temp[1] and bbox[1] <= bbox_temp[3]:
                                    print(track.track_id,'번 째 사람 ',car_table[h] ,'차량과 충돌')
                                    #충돌할때마다. 위험등급을 늘려서 human_table 저장시켜준다.
                                    #위험등급이몇인지 체크
                                    dangerous_grade = 0
                                    
                                
                                    for a in range(len(self.human_table)): #숫자 : 최대 위험등급
                                        for b in self.human_table[a]:
                                            if  b == str(track.track_id):
                                                if a>10 :
                                                    dangerous_grade = a
                                                    print('위험인물 식별. 확인바람.') # 경고를 해주고, 등급을 올리지않는다.
                                                else :
                                                    dangerous_grade = a
                                                    
                                                    
                                                    #위험등급 업.
                                    
                                    #충돌은 한상태. 등급업
                                    if dangerous_grade <4 :
                                        
                                        print(self.human_table)
                                        
                                        #if track.track_id in self.human_table:
                                        self.human_table.append([])
                                        del self.human_table[dangerous_grade][self.human_table[dangerous_grade].index(str(track.track_id))]#자기자신 삭제
                                        
                                        self.human_table[dangerous_grade+1].append(str(track.track_id))   #+=1 이거도 중복없게.
                                        print('등급업')
                                    car_id = str(car_table[h])
                                    human_id = str(track.track_id)
                                    grade = str(dangerous_grade+1)
                                    self.addRow()            
                
                
                ##0829
                #충돌 테스트.
                #print(check_num)
                
                #if class_name == str("person") :

                    #print("%s",track[0].get_class())
                    
                    #밑 코드는 자신번호 -1의 객체와 비교.
                    #num_temp = check_num
                    #if num_temp > 0:s
                    #   num_temp -= 1
                    #  class_name_temp = track[num_temp].get_class()
                    #   bbox_temp = track[--num_temp].to_tlbr()
                    #    x1,y1, x2,y2: bbox[0],bbox[1],bbox[2],bbox[3]
                    #                   bbox_temp[0],bbox_temp[1],bbox_temp[2],bbox_temp[3]
                    #   if x1 + w1 >= x2 and x1 <= x2 + w2 and y1 + h1 >= y2 and y1 <= y2 + h2:
                    #   if bbox[2] >= bbox_temp[0] and bbox[0] <= bbox_temp[2] and bbox[3] >= bbox_temp[1] and bbox[1] <= bbox_temp[3]:
                    #       print("충돌")
                    #   else:
                    #       print("미충돌")
                    #차인경우만 비교 if class_name_temp == str("car") :
                    # 

                
            # draw bbox on screen
                #박스크기 조정 테스트
                
                
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                #cv2.rectangle(frame, (int(bbox[0]) + int(control_size_x), int(bbox[1])+ int(control_size_y)), (int(bbox[2])- int(control_size_x), int(bbox[3])- int(control_size_y)), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                #cv2.rectangle(frame, (int(bbox[0]) + int(control_size_x), int(bbox[1]-30)+ int(control_size_y)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17- int(control_size_x), int(bbox[1])- int(control_size_y)), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                
            # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                check_num += 1
                #
                
            #다음프레임 car_table 초기화
            car_table.clear()
            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            #result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            

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

            
    def stop(self):
        self.prt.setText("대기 중입니다.")
        self.prt.setStyleSheet("Color : Green")
        #self.movielabel.show()
        self.frame.setPixmap(QPixmap.fromImage(QImage()))
        #self.frame.setPixmap(self.icon)
        #self.movielabel.setMovie(self.movie)
        #self.movie.start()
        FLAGS.run_video = True

                
    def nextFrameSlot2(self):
        self.prt.setText("탐지 중입니다 . . .")
        self.prt.setFont(QFont("맑은 고딕",10,QFont.Bold))
        self.prt.setStyleSheet("Color : blue")
        _,cam=self.cpt.read() 
        cam=cv2.cvtColor(cam,cv2.COLOR_BGR2RGB) #bgr rgb로 변환
        img=Image.fromarray(cam)
        #detected_image,label=helmetAlarm.detect_image(img)
        result=np.asarray(img)#detected_image
        cam=result
        img=QImage(cam,cam.shape[1], cam.shape[0],QImage.Format_RGB888) #input, width, heigth, format
        pix=QPixmap.fromImage(img)
        self.frame.setPixmap(pix)
        t=time.localtime()
        if t.tm_sec%3==0:
            self.prt.setText("탐지 중입니다 . ")
        elif t.tm_sec%3==1:
            self.prt.setText("탐지 중입니다 . . ")
        else:
            self.prt.setText("탐지 중입니다 . . .")
                
    """
        if(not(t.tm_sec==recent_prt_sec and t.tm_min == recent_prt_min) and len(label)!=0):
            self.addRow(label)
    """    

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
        """
        self.table.scrollToBottom()
        
        global recent_prt_sec
        recent_prt_sec=t.tm_sec
        global recent_prt_min
        recent_prt_min=t.tm_min
        print(recent_prt_sec,recent_prt_min)
        """
    
    
if __name__ == '__main__':
    try:
        app=QApplication(sys.argv)
        w=Example()
        sys.exit(app.exec_())
        #app.run(main)
    except SystemExit:
        pass

import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
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

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None
    human_table = [[]]
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    
    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
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
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
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
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        #
        #print(scores)
        #print(scores[0])
        #print("classes"+ classes)
        #print(boxs)
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        check_num = 0
        car_table = list()
        n = 0
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            flag_overlap = True
            if FLAGS.check_crash:
                if class_name == str("car") :
                    car_table.append(str(track.track_id))
                    car_table.append(bbox)
  
                  
                if class_name == str("person") :
                    #사람 위험도 테이블에 추가 #중복제외 필요. #아이디가 없으면 추가. #첫추가는 무조건.
                    for i in human_table[0] :
                        if i == str(track.track_id) :
                            flag_overlap = False
                    if flag_overlap:
                        human_table[0].append(str(track.track_id))    
                    print(human_table[0])
                    
                    #if x==car_table.get(id+track.track.id,x):#일치하는 id가 없으면 x리턴
                    #    car_table[id+track.track.id] = 0

                    n += 1 #몇번째 사람.
                    
                    if len(car_table)>2 :
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
                                print('차량안의 사람 제거')
                            #충돌확인
                            elif bbox[2] >= bbox_temp[0] and bbox[0] <= bbox_temp[2] and bbox[3] >= bbox_temp[1] and bbox[1] <= bbox_temp[3]:
                                print(n,'번 째 사람 ',car_table[h] ,'차량과 충돌')
                                #충돌할때마다. 위험등급을 늘려서 human_table 저장시켜준다.
                                #위험등급이몇인지 체크
                                dangerous_grade = 0
                                for a in range(len(human_table)): #숫자 : 최대 위험등급.
                                    for b in human_table[a]:
                                        if  b == track.track_id:
                                            if a==4 :
                                                print('위험인물 식별. 확인바람.') # 경고를 해주고, 등급을 올리지않는다.
                                            else :
                                                dangerous_grade = a
                                                #위험등급 업.

                                #충돌은 한상태. 등급업
                                if dangerous_grade <4 :               
                                    del human_table[[dangerous_grade].index(track.track_id)]#자기자신 삭제
                                    human_table[dangerous_grade+1].append(track.track_id)   #+=1
                                    print('등급업')
                                                    
            
            
            #0829
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
            per = 80 #50%로 축소.
            w = int(bbox[2])-int(bbox[0])
            h = int(bbox[3])-int(bbox[1])
            control_size_x = w*0.5 * (1-(per*0.01))
            control_size_y = h*0.5 * (1-(per*0.01))
            #테스트
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]) + int(control_size_x), int(bbox[1])+ int(control_size_y)), (int(bbox[2])- int(control_size_x), int(bbox[3])- int(control_size_y)), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            #cv2.rectangle(frame, (int(bbox[0]) + int(control_size_x), int(bbox[1]-30)+ int(control_size_y)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17- int(control_size_x), int(bbox[1])- int(control_size_y)), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            check_num += 1
        #다음프레임 car_table 초기화
        car_table.clear()
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


#! /usr/bin/env python

import os

# 카메라 모듈 불러오기
from camera import *

# 모터 제어 발행 코드 불러오기
from scout_motor_pub import *

# key 변수로 모터 제어 함수 불러오기
from key_move import *

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

# webcam 사용 여부 결정
use_webcam = True

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', '/home/aiffel-dj40/st_mini/src/scout_mini_ros/scout_bringup/checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
if use_webcam:
    flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam') # '/home/aiffel-dj40/st_mini/src/scout_mini_ros/scout_bringup/data/video/two_people.mp4'
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

# main 함수 = DeepSORT 모델 + ROS Publisher + depth camera
def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = '/home/aiffel-dj40/st_mini/src/scout_mini_ros/scout_bringup/model_data/mars-small128.pb'
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
    if use_webcam:
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
    if use_webcam:
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    
    # Depth camera class 불러오기
    if not use_webcam:
        dc = DepthCamera()

    frame_num = 0

    # 로봇 모터 제어를 위한 초깃값 설정
    x = 0
    y = 0
    z = 0
    th = 0
    speed = 0.5
    turn = 1

    target_id = False

    # while video is running
    while not rospy.is_shutdown():
        if use_webcam:
            return_value, frame = vid.read()
        else:
            # depth camera 사용
            return_value, depth_frame, frame = dc.get_frame()
        
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
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person', 'cell phone', 'tie', 'stop sign']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
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

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        """
        로봇 제어를 위한 code 시작
        """

        # <st-mini 제어를 위한 Publisher code>
        go = scout_pub_basic()
        rate = rospy.Rate(50)
        go.update(x,y,z,th,speed,turn)
        go.sendMsg()

        # track id list 만들기
        track_id_list = []

        # class name list 만들기
        class_name_list = []

        # person id to bbox dict 만들기
        person_id_to_bbox = {}
        """아래와 같은 형식의 데이터
        person_id_to_bbox = {
            1 : [100,200,300,400] # id : bbox
        }
        """

        # recognition bbox list 만들기
        recognition_bbox_list = []
        
        # 인식용 객체(인식용 객체가 사람 bbox 안에 있다면 그 사람의 id를 계속 추적, 이후 target lost를 하게 되면 동일하게 인식 가능)
        recognition_object = 'cell phone'

        # track id list 만들기
        for track in tracker.tracks:
            class_name = track.get_class()
            track_id = track.track_id
            bbox = track.to_tlbr()
            
            # track id list 만들기
            track_id_list.append(track_id)

            if class_name == 'person': person_id_to_bbox[track_id] = bbox
            elif class_name == recognition_object: recognition_bbox_list.append(bbox)

    
        # person bbox 내부에 recognition object bbox가 있으면 해당 person id를 계속 추적하도록 target_id 할당
        if not target_id: # False라면 사람 id를 할당해야한다.
            try:
                print('try 실행')
                for person_id, person_bbox in person_id_to_bbox.items():
                    person_bbox = list(map(int, person_bbox))
                    for rec_bbox in recognition_bbox_list:
                        rec_bbox = list(map(int, rec_bbox))
                        if person_bbox[0] <0: person_bbox[0] = 0
                        elif person_bbox[1] < 0: person_bbox[1] = 0
                        elif person_bbox[2] > frame.shape[0]: person_bbox[2] = frame.shape[0]
                        elif person_bbox[3] > frame.shape[1]: person_bbox[3] = frame.shape[1]
                        if rec_bbox[0] >= person_bbox[0] and rec_bbox[1] >= person_bbox[1] and rec_bbox[2] <= person_bbox[2] and rec_bbox[3] <= person_bbox[3]:
                            target_id = person_id
            except:
                print('except 실행')
                target_id = False

        print('target_id: ',target_id)
    
        # detection된 객체가 없다면 멈춤(target lost)
        if target_id not in track_id_list:
            key = 'stop'
        
        else:
            # update tracks(한 화면에 잡히는 모든 객체를 for문을 통해 반복한다. 우리는 하나의 객체만 추적해야 하므로 타겟 객체에 대해서면 모터 제어를 해야 한다.)
            for track in tracker.tracks:            
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                        
                bbox = track.to_tlbr()
                class_name = track.get_class()
                
                # 타겟 객체 id일때 모터 제어 실시
                if target_id == track.track_id:
                    # target bbox의 width, height
                    width_bbox = bbox[2] - bbox[0]
                    height_bbox = bbox[3] - bbox[1]

                    # target bbox의 center 좌표
                    cx = width_bbox/2 + bbox[0]
                    cy = height_bbox/2 + bbox[2]
                    # cv2.circle(frame, (cx,cy), 3, (0,0,255), 2)
                    """
                    depth값 활용
                    1. Target과의 거리[전진]
                    1) 적정거리: 1.5m ~ 2.0m --> linear.x = 1
                    2) 위험거리: ~1.5m       --> linear.x = 0
                    3) 추격거리: 2.0m~       --> linear.x += 0.2 (적정거리가 될 때까지)

                    2. Target의 중심점을 이용해 좌우 회전
                    1) 중심점 cx, cy는 아래와 같이 구할 수 있다.
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    cx = int(width/2 + bbox[0])
                    cy = int(height/2 + bbox[1])
                    좌우 판단이기 때문에 cx만 사용.

                    2) cx값 설정 중 주의 사항
                    target이 화면 밖으로 점점 나갈수록 bbox의 좌측 상단 x좌표는 음수 혹은 frame width(여기서는 640)보다 커질 수 있다.
                    즉, cx의 값을 설정할 때 bbox[0]값이 음수 또는 640을 초과하면 좌우 회전 즉시 실시해야 함.

                    depth camera를 통해 장애물 유무를 먼저 판단하고 없다면 Target과의 거리/방향 측정 후 최종 발행값 결정.
                    """
                    # 좌/우 회전 한곗값 설정
                    left_limit = frame.shape[1]//2 - 50
                    right_limit = frame.shape[1]//2 + 50

                    # 좌/우 회전 속도 증가 구간 설정
                    left_max = frame.shape[1]//4
                    right_max = (frame.shape[1]//4)*3

                    """
                    depth camera를 이용해 장애물과의 거리 결과(str)를 담은 변수 = depth_obstacle, 좌우 판단
                    depth camera를 이용해 Target과의 거리 결과(float)를 담은 변수 = depth_target
                    """

                    # depth 값
                    depth_target = 1800
                    depth_obstacle = 'None'

                    # 직진 안전 구간 최대/최소값
                    stable_max_dist = 2000
                    stable_min_dist = 1500

                    # 직진, depth_result라는 변수는 depth값 판단 결과에 따른 결괏값.
                    if depth_target < stable_min_dist: # 로봇과 사람의 거리가 1.5m 미만이라면 정지
                        key = 'stop!!'
                    else: # 로봇과 사람의 거리가 1.5m이상일 때만 구동
                        # 회피해야할 장애물이 있을때
                        if depth_obstacle == 'left':
                            key = 'turn_right'
                        elif depth_obstacle == 'right':
                            key = 'turn_left'
                        else: # 전진에 아무런 문제가 없다면 전진
                            # Target의 위치 파악(좌우 회전이 필요한 지)
                            if cx <= left_limit: 
                                key = 'go_turn_left' # bbox 중앙점 x좌푯값이 좌측 회전 한곗값(left_limit)보다 작으면 좌회전
                                if bbox[0] < 0 or cx < left_max: key = 'angular_speed_up'
                                elif turn > 1.2: turn = 1.2
                            elif cx >= right_limit: 
                                key = 'go_turn_right' # bbox 중앙점 x좌푯값이 우측 회전 한곗값(right_limit)보다 크면 우회전
                                if bbox[2] >= frame.shape[1] or cx > right_max: key = 'augular_speed_up'
                                elif turn > 1.2: turn = 1.2
                            else: # 좌/우 회전이 아니라면 직진, 거리에 따른 속도 제어
                                if stable_min_dist <= depth_target <= stable_max_dist:
                                    key = 'go' # 1.5 ~ 2.0m라면 전진
                                    speed = 0.1
                                else: # 2.0m 초과라면 거리에 따른 속도 증감
                                    speed_factor = depth_target - stable_max_dist # 안정 거리 최대값과 사람과의 거리의 차이, 이를 이용해 얼만큼 속도를 증감해야하는 지 정하는 요소
                                    """
                                    기본 컨셉은 speed_factor가 200 이상이면 속도 증가, 미만이면 속도 감소(대신 속도 최대값은 2.5, 최소값은 1.2로 설정한다)
                                    예를 들어 사람과 로봇의 거리가 2500이라면 speed_factor는 500이 된다. 로봇은 speed_factor이 200이 될 때까지 속도 증가(최댓값은 2.5로 설정함.)
                                    speed_factor이 200 미만이 되는 순간 속도 감소(최솟값은 1.2로 설정)
                                    그리고 speed_factor이 0이 되면 안정 구간 진입이므로 속도는 1로 설정됨.
                                    
                                    <아래는 로봇과 사람이 2.5m 떨어진 상황>
                                    ┌─┐
                                    └─┘<----위험(stop)---->|<---안정(go)--->|<---속도 증가(최댓값: 2.5)--->|<---속도 감소(최솟값: 1.2)--->|
                                    0  0                    1.5m            2.0m                          2.3m                          2.5m
                                    """
                                    if speed_factor >= 200: # speed up
                                        if speed < 2.5:
                                                key = 'linear_speed_up'
                                        else:
                                            speed = 2.5
                                            # key = 'not_change_speed'
                                    else: # speed down
                                        if speed > 1.2:
                                            key = 'linear_speed_down'
                                        else:
                                            speed = 1.2
                                            # key = 'not_change_speed'
        x,y,z,th,speed,turn = key_move(key,x,y,z,th,speed,turn)
        print('key: ', key)
        rate.sleep()

    #     for track in tracker.tracks:            
    #         if not track.is_confirmed() or track.time_since_update > 1:
    #             continue 
                        
    #     # draw bbox on screen
    #         color = colors[int(track.track_id) % len(colors)]
    #         color = [i * 255 for i in color]
    #         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    #         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
    #         cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

    #     # if enable info flag then print details about each track
    #         if FLAGS.info:
    #             print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
    #     print('track id list: ', track_id_list)
    #     # calculate frames per second of running detections
    #     fps = 1.0 / (time.time() - start_time)
    #     print("FPS: %.2f" % fps)
    #     result = np.asarray(frame)
    #     result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
    #     if not FLAGS.dont_show:
    #         cv2.imshow("Output Video", result)
        
    #     # if output flag is set, save video file
    #     if FLAGS.output:
    #         out.write(result)
    #     if cv2.waitKey(1) & 0xFF == ord('q'): break
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    print('start')
    try:
        app.run(main)

    except SystemExit:
        pass
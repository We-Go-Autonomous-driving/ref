#! /usr/bin/env python

import os

# 카메라 모듈 불러오기
from camera import *

# 모터 제어 발행 코드 불러오기
from scout_motor_pub import *

# key 변수로 모터 제어 함수 불러오기
from key_move import *

# RGB값을 이용한 주행 알고리즘 불러오기
from drive import drive

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

# 임포트 추가
# import pyrealsense2 as rs
from utils2 import *
from Default_dist import *

use_webcam = False

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
#flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video', './data/0527_4_follow_1.mp4', 'path to input video')
#flags.DEFINE_string('video', 0, 'path to input video')
#flags.DEFINE_string('video', './data/0527_1_far_away.mp4', 'path to input video')
#flags.DEFINE_string('video', './data/0527_3_left_right.mp4', 'path to input video')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

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
    #video_path = FLAGS.video

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

    # 주석처리 (삭제)
    # begin video capture
    '''try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)'''
    
    # realsense 스트림에서 읽어오기위한 코드 추가
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    # 로봇 모터 제어를 위한 초깃값 설정
    x = 0
    y = 0
    z = 0
    th = 0
    speed = 0.5
    turn = 1

    # 변수 추가
    cx, cy, h = 0, 0, 0
    frame_num = 0

    # realsense align 맞추기 추가
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Depth camera class 불러오기
    # dc = DepthCamera()
    default = Default_dist()

    # while video is running
    while not rospy.is_shutdown():
         # realsense 파이프라인에서 프레임 받아오기 추가
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        frame = aligned_frames.get_color_frame()

        #depth_frame = frames.get_depth_frame()
        #frame = frames.get_color_frame()

        depth_frame = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(frame.get_data())

        # 삭제
        #return_value, frame = vid.read()
        return_value = True
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        
        # 장애물 회피를 위한 ROI 디폴트 세팅하기 (현재는 10프레임만) 추가
        if frame_num < 11 :
            default.default_update(depth_frame)
            continue
        
        print('Frame #: ', frame_num)
        #frame_size = frame.shape[:2]
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
        
        #fps = 1.0 / (time.time() - start_time)
        #print("1-FPS: %.2f" % fps)

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

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

        #print('detections[0].feature : ',detections[0].feature)
        #Image.fromarray(detections[0].feature.astype(np.uint8)).show()
        #gg = cv2.cvtColor(detections[0].feature, cv2.COLOR_RGB2BGR)
        #cv2.imshow("feature", gg)
        #print('(detections[0].feature.shape : ', detections[0].feature.shape)
        #print('features.shape : ', features.shape)
        #print('type : ', type(features))
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
        tracker.update(detections, frame_num)
        
        # <st-mini 제어를 위한 Publisher code>
        go = scout_pub_basic()
        rate = rospy.Rate(200)
        go.update(x,y,z,th,speed,turn)
        go.sendMsg()

        # try: # 추적할 대상이 있다면
            # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            #print('track_id : ', track.track_id)
            bbox = track.to_tlbr()
            class_name = track.get_class()
            #print('bbox : ', bbox)

            # cx, cy 계산 추가
            w, h = int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
            cx, cy = int(w/2 + bbox[0]), int(h/2 + bbox[1])
            #print('cx, cy : ', cx, cy)
            
            # 사람과 로봇의 거리 person_distance
            person_distance = person_dist(depth_frame, cx, cy, h)

            # 직진 안전 구간 최대/최소값
            stable_max_dist = 2500
            stable_min_dist = 2000
            
            if person_distance < stable_min_dist:
                key = 'stop'
                x,y,z,th,speed,turn = key_move(key,x,y,z,th,speed,turn)
                print('key: ', key)
            else:
                """
                depth값 활용
                1. Target과의 거리[전진]
                1) 적정거리: 2.0m ~ 2.5m --> linear.x = 0.7
                2) 위험거리: ~2.0m       --> linear.x = 0
                3) 추격거리: 2.5m~       --> linear.x += 0.2 (적정거리가 될 때까지)

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

                max_speed = 0.9
                min_speed = 0.9

                """
                정지 조건(3/3): 장애물이 있을 때, 정확히는 정지가 아닌 회피 기동
                obstacle_detect 함수는 장애물이 있을때만 key에 값을 할당한다.
                """
                # 장애물 회피용 ROI distance로 left, right string 받아오기
                # key = obstacle_detect(default, depth_frame)
                cv2.rectangle(frame, (cx+10, cy-(h//5)+10), (cx-10, cy-(h//5)-10), (255, 0, 0), 5)
                # 장애물이 없다면 사람 따라가기
                # if not key:
                key = drive(bbox, cx, left_limit, left_max, right_limit, right_max, turn, frame, stable_min_dist, person_distance, stable_max_dist,speed,max_speed,min_speed)
                x,y,z,th,speed,turn = key_move(key,x,y,z,th,speed,turn)
                print('key: ', key)
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            #cv2.circle(frame, (cx, cy), 10, (255, 0, 0))
            print('speed: {}, turn: {}'.format(speed,turn))
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
    
        # 추적할 대상이 없다면 정지
        # except:
        #     key = 'stop'
            # x,y,z,th,speed,turn = key_move(key,x,y,z,th,speed,turn)
            # print('key: ', key)
        cv2.circle(frame, (320, 240), 10, (255, 255, 255))

        # x,y,z,th,speed,turn = key_move(key,x,y,z,th,speed,turn)
        # print('key: ', key)
        rate.sleep() 

        # 트레킹하는 person에서 가운데 거리 받아오기 추가
        # print('person distance : ', person_dist(depth_frame, cx, cy, h))
        
        # bbox center 20x20 roi
        '''
        box_center_roi = np.array((depth_frame[cy-10:cy+10, cx-10:cx+10]),dtype=np.float64)
        cv2.rectangle(frame, (cx-10, cy+10), (cx+10, cy-10), (255, 255, 255), 2)
        '''

        # 장애물 회피용 ROI distance로 left, right string 받아오기 추가
        # key = obstacle_detect(default, depth_frame)
        #box_center_roi = np.array((depth_frame[cy-10-(h/5):cy+10-(h/5), cx-10:cx+10]),dtype=np.float64)
        # cv2.rectangle(frame, (cx+10, cy-(h//5)+10), (cx-10, cy-(h//5)-10), (255, 0, 0), 5)
        
        # 아래 애들은 다 필요 없음
        # safe zone ROI
        #(240, 420) (400, 420)
        #(160, 480) (480, 480)
        safe_roi = np.array([[400, 400], [240, 400], [160, 480], [480, 480]])
        #safe_roi = np.array([[240, 420], [400, 420], [480, 160], [480, 480]])
        cv2.polylines(frame, [safe_roi], True, (255, 255, 255), 2)
        cv2.rectangle(frame, (205, 445), (195, 435), (255, 0, 0), 5)
        cv2.rectangle(frame, (245, 405), (235, 395), (255, 0, 0), 5)
        cv2.rectangle(frame, (405, 405), (395, 395), (255, 0, 0), 5)
        cv2.rectangle(frame, (445, 445), (435, 435), (255, 0, 0), 5)


        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        info = "time: %.2f ms" %(1000*(time.time() - start_time))
        #print(info)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # depth map을 칼라로 보기위함 
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
            #cv2.imshow("Output Video", depth_colormap)
        
        # if output flag is set, save video file
        #if FLAGS.output:
        #    out.write(result)
        if cv2.waitKey(400) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

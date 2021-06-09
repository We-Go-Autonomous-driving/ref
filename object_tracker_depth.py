#! /usr/bin/env python

# 로봇 제어 및 모터 제어를 위한 import
import rospy
from geometry_msgs.msg import Twist
import os

# 조명 제어를 위한 import
from scout_mini_ros.scout_bringup.light_control import *
from scout_msgs.msg import ScoutLightCmd

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
flags.DEFINE_string('weights', '/home/aiffel-dj40/st_mini/src/scout_mini_ros/scout_bringup/checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '0', 'path to input video or set to 0 for webcam') # '/home/aiffel-dj40/st_mini/src/scout_mini_ros/scout_bringup/data/video/0527_1_far_away.mp4'
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

# st-mini 속도, 방향 제어
class scout_pub_basic():
    def __init__(self):
        rospy.init_node("scout_pub_basic_name", anonymous=False)
        self.msg_pub = rospy.Publisher(
            '/cmd_vel',
            Twist,
            queue_size = 10
        )
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0

    def update(self,x,y,z,th,speed,turn):
        self.x = x
        self.y = y
        self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn

    def sendMsg(self):
        tt = Twist()
        tt.linear.x = self.x * self.speed
        tt.linear.y = self.y * self.speed
        tt.linear.z = self.z * self.speed
        tt.angular.x = 0
        tt.angular.y = 0
        tt.angular.z = self.th * self.turn
        self.msg_pub.publish(tt)

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

    # d435 depth camera
    class DepthCamera:
        def __init__(self):
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


            # Start streaming
            self.pipeline.start(config)

        def get_frame(self):
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            if not depth_frame or not color_frame:
                return False, None, None
            return True, depth_image, color_image

        def release(self):
            self.pipeline.stop()

    # Initialize Camera Intel Realsense
    dc = DepthCamera()

    out = None

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
    while not rospy.is_shutdown():
        return_value, depth_frame, frame = dc.get_frame() # depth camera에서 depth & RGB값 받기
        
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

        # <st-mini 제어를 위한 초기값 설정 & Publisher code>
        x = 0
        y = 0
        z = 0
        th = 0
        speed = 0.5
        turn = 0
        
        moveBindings = {
            'go':(1,0,0,0), # i
            'go_turn_right':(1,0,0,-1), # o
            'turn_left':(0,0,0,1), # j
            'turn_right':(0,0,0,-1), # l
            'go_turn_left':(1,0,0,1), # u
            'back':(-1,0,0,0), # ,
            'back_right':(-1,0,0,1), # .
            'back_left':(-1,0,0,-1), # m
            'parallel_go_right':(1,-1,0,0), # O
            'parallel_left':(0,1,0,0), # J
            'parallel_right':(0,-1,0,0), # L
            'parallel_go_left':(1,1,0,0), # U
            'parallel_back_right':(-1,-1,0,0), # >
            'parallel_back_left':(-1,1,0,0), # M
        }

        speedBindings={
            'total_speed_up':(1.1,1.1), # q
            'total_speed_down':(.9,.9),   # z
            'linear_speed_up':(1.1,1),   # w
            'linear_speed_down':(.9,1),    # x
            'angular_speed_up':(1,1.1), # e
            'angular_speed_down':(1,.9),  # c
        }
        
        # ROS Publisher class이용
        scout = scout_pub_basic()
        rate = rospy.Rate(200)   # rospy.Rate는 Hz를 나타냄. 1초에 몇 번 통신할 것인지 정할 수 있다. 높을수록 1초에 더 많은 통신을 한다.
        
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
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
            left_limit = 220
            right_limit = 420

            """
            depth camera를 이용해 장애물과의 거리 결과(str)를 담은 변수 = depth_obstacle, 좌우 판단
            depth camera를 이용해 Target과의 거리 결과(float)를 담은 변수 = depth_target
            """

            # 직진, depth_result라는 변수는 depth값 판단 결과에 따른 결괏값.
            if depth_target < 1500: # 로봇과 사람의 거리가 1.5m 미만이라면 정지
                key = 'stop!!'
            else: # 로봇과 사람의 거리가 1.5m이상일 때만 구동
                if depth_obstacle == 'left':
                    key = 'parallel_go_left'
                elif depth_obstacle == 'right':
                    key = 'parallel_go_right'
                else: # 전진에 아무런 문제가 없다면 전진
                    # Target의 위치 파악(좌우 회전이 필요한 지)
                    if bbox[0] <= 0:
                        key = 'go_turn_left' # bbox 좌측 상단 x좌푯값이 음수라면 좌회전
                    elif bbox[2] >= frame.shape[1]:
                        key = 'go_turn_right' # bbox 우측 하단 x 좌푯값이 영상width값 이상이면 우회전
                    else: # cx의 값을 이용해 좌/우 판단
                        if cx <= left_limit: key = 'go_turn_left' # bbox 중앙점 x좌푯값이 좌측 회전 한곗값(left_limit)보다 작으면 좌회전
                        elif cx >= right_limit: key = 'go_turn_right' # bbox 중앙점 x좌푯값이 우측 회전 한곗값(right_limit)보다 크면 우회전
                        else: # 좌/우 회전이 아니라면 직진, 거리에 따른 속도 제어
                            if 1500 =< depth_target <= 2000: key = 'go' # 1.5 ~ 2.0m라면 전진
                            else: # 2.0m 초과라면 속도를 점차 증가
                                while 2500 < depth_target:
                                    key = 'linear_speed_up'
                                while 2000 < depth_target <= 2500:
                                    key = 'linear_speed_down'

            # <구동 상태에 따른 조명 제어>
            if 'left' in key or 'right' in key:
                emergency() # 조명 빠르게 깜빡
            elif key = 'go':
                on()
            elif 'speed' in key:
                emergency()
            else:
                off()
            if target_lost: # target을 잃었다면 주기적인 깜빡임
                blink()
            
            # <key라는 변수를 이용해 st-mini 움직임 제어>
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]
            else:
                x = 0
                y = 0
                z = 0
                th = 0
            scout.update(x,y,z,th,speed,turn)
            scout.sendMsg()
            print('speed = {}, key: {}, x = {}, y = {}, z = {}, th = {}'.format(speed, key, x, y, z, th))
            rate.sleep()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

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
    print('start')
    try:
        app.run(main)

    except SystemExit:
        pass
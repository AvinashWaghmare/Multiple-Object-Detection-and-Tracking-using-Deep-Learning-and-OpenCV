from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes
from flask import Flask, render_template, Response, request, redirect, send_file, url_for, jsonify
from werkzeug.utils import secure_filename, send_from_directory
import subprocess
from deep_SORT import preprocessing
from deep_SORT import nn_matching
from deep_SORT.detection import Detection
from deep_SORT.tracker import Tracker
from tools import generate_detections as gdet
import time

global switch
switch = 1

app = Flask(__name__)

uploads_dir = os.path.join(app.instance_path, 'uploads')

os.makedirs(uploads_dir, exist_ok=True)


class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')


max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_dataset/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/walk.mp4')


codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

counter1 = []
counter2 = []
counter3 = []
counter4 = []


@app.route('/')
def index():
    return render_template('index.html')


def gen(vid):

    while True:
        _, img = vid.read()

        if img is None:
            print('Completed')
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        t1 = time.time()

        boxes, scores, classes, nums = yolo.predict(img_in)

        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

        i = int(0)
        i1 = int(0)
        i2 = int(0)
        i3 = int(0)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >1:
                continue
            bbox = track.to_tlbr()
            class_name= track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                        +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                        (255, 255, 255), 2)

            center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
            pts[track.track_id].append(center)

            height, width, _ = img.shape

            cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
            cv2.line(img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)


            center_y = int(((bbox[1])+(bbox[3]))/2)

            if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
                if class_name == 'car':
                    counter1.append(int(track.track_id))
                    i += 1

                if class_name == 'person':
                    counter2.append(int(track.track_id))
                    i1 += 1

                if class_name == 'bicycle':
                    counter3.append(int(track.track_id))
                    i2 += 1

                if class_name == 'truck':
                    counter4.append(int(track.track_id))
                    i3 += 1



        count1 = len(set(counter1))
        count2 = len(set(counter2))
        count3= len(set(counter3))
        count4 = len(set(counter4))


        cv2.putText(img, "car: " + str(count1), (int(20), int(100)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(img, "person: " + str(count2), (int(20), int(80)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(img, "bicycle: " + str(count3), (int(20), int(60)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(img, "truck: " + str(count4), (int(20), int(40)), 0, 5e-3 * 100, (0, 255, 0), 2)


        fps = 1. / (time.time() - t1)
        cv2.putText(img, "FPS: %f" % (fps), (int(20), int(15)), 0, 5e-3 * 100, (0, 255, 0), 2)

        #cv2.namedWindow("MOT_output", cv2.WINDOW_AUTOSIZE)
        #cv2.resizeWindow('MOT_output', 1900, 1300)
        #cv2.imshow('MOT_output', img)
        out.write(img)



        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        if cv2.waitKey(1) == ord('q'):
            break



    out.release()


@app.route('/webcam')
def webcam():
        global vid
        return Response(gen(vid), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, vid
    if request.method == 'POST':
        if request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                vid.release()
                cv2.destroyAllWindows()
            else:
                vid = cv2.VideoCapture(0)
                switch = 1
                out.release()
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

@app.route("/detect", methods=['POST'])
def detect():

    if not request.method == "POST":
        return
    video = request.files['video']
    file_name = video.filename
    video.save(os.path.join(uploads_dir, secure_filename(file_name)))
    print(video)

    subprocess.call('dir',shell=True)
    subprocess.call(["python", "detect.py", os.path.join(uploads_dir, secure_filename(file_name))], shell=True)

    #return os.path.join(uploads_dir, secure_filename(video.filename))
    obj= secure_filename(file_name)
    return obj


@app.route("/return-files", methods=['GET', 'POST'])
def return_file():

    obj = request.args.get('obj')
    loc = os.path.join("runs/detect", obj)
    print(loc)
    try:
        return send_file(os.path.join("runs/detect", obj), attachment_filename=obj)
        # return send_from_directory(loc, obj)
    except Exception as e:
        return str(e)







if __name__ == '__main__':

    app.run(debug=True)
    vid.release()
    cv2.destroyAllWindows()
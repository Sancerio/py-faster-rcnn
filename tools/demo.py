#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Updated by Limiardi Eka Sancerio
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from rpn.generate import im_proposals
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import skvideo.io
import argparse
import collections

CLASSES = ('__background__',
           'person', 'bicycle', 'car', 'motorcycle', 
           'airplane', 'bus','train', 'truck', 'boat', 
           'traffic light', 'fire hydrant', 'stop sign', 
           'parking meter', 'bench', 'bird', 'cat', 
           'dog', 'horse', 'sheep', 'cow', 
           'elephant', 'bear', 'zebra', 'giraffe', 
           'backpack', 'umbrella', 'handbag', 'tie', 
           'suitcase', 'frisbee', 'skis', 'snowboard', 
           'sports ball', 'kite', 'baseball bat', 'baseball glove', 
           'skateboard', 'surfboard', 'tennis racket', 'bottle', 
           'wine glass', 'cup', 'fork', 'knife', 
           'spoon', 'bowl', 'banana', 'apple', 
           'sandwich', 'orange', 'broccoli', 'carrot', 
           'hot dog', 'pizza', 'donut', 'cake', 
           'chair', 'couch', 'potted plant', 'bed', 
           'dining table', 'toilet', 'tv', 'book', 
           'mouse', 'remote', 'keyboard', 'cell phone', 
           'microwave', 'oven', 'toaster', 'sink', 
           'refrigerator', 'book', 'clock', 'vase', 
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'zf_faster_rcnn_coco.caffemodel')}

smoothing_window = 10

def vis_detections(im, class_name, dets, thresh=0.5):
    """Return list of tuples of (bounding box, score)."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print "no " + class_name+ " detected."
        return
    boxes_scores = []

    #print ("{:d} {:s}s detected").format(len(inds), class_name)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        boxes_scores.append((bbox,score))

    return boxes_scores

def det_objects(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] <= thresh)[0]
    if len(inds) == 0:
        #print "no " + class_name+ " detected."
        return
    boxes_scores = []

    #print ("{:d} {:s}s detected").format(len(inds), class_name)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        #print ("Score for {:s} {:d} is {:.3f} ").format(class_name, i, score)
        #print ("Detected coordinates : {:.1f}, {:.1f}, {:.1f}, and {:.1f}").format(bbox[0],bbox[1],bbox[2],bbox[3])
        boxes_scores.append((bbox,score))

    return boxes_scores

def iou(box1, box2):
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2

    area1 = (x21 - x11) * (y21 - y11)
    area2 = (x22 - x12) * (y22 - y12)

    xx1 = np.maximum(x11, x12)
    yy1 = np.maximum(y11, y12)
    xx2 = np.minimum(x21, x22)
    yy2 = np.minimum(y21, y22)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    iou = inter / (area1 + area2 - inter)
    
    return iou

def is_learnt_boxes(list_of_boxes, boxes_scores, thresh = 0.5):
    """Check Unique Unlearnt Boxes"""

    x1 = boxes_scores[0][0][0]
    y1 = boxes_scores[0][0][1]
    x2 = boxes_scores[0][0][2]
    y2 = boxes_scores[0][0][3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for box in list_of_boxes:
        for coordinates in box[0]:
            xx1 = np.maximum(x1, coordinates[0][0])
            yy1 = np.maximum(y1, coordinates[0][1])
            xx2 = np.minimum(x2, coordinates[0][3])
            yy2 = np.minimum(y2, coordinates[0][3])
            areas2 = ((coordinates[0][2]-coordinates[0][0]+1)*
                (coordinates[0][3]-coordinates[0][1]))
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas + areas2 - inter)
            if(iou >= thresh):
                return True

    return False

def demo(net, im, PrevClass):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    if im != None:
        object_scores, object_boxes, scores, boxes = im_detect(net, im)
        timer.toc()
        print ('\nDetection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        CONF_THRESH = 0.7

        NMS_THRESH = 0.1
        rslt = []
        #print scores.shape
        #print scores
        for cls_ind, cls in enumerate(CLASSES[1:]):

            #fine tuning class probability
            conf = CONF_THRESH
            #print cls
            if cls in PrevClass:
                conf -= 0.1
            
            if cls == 'cell phone':
                conf = 0.5
            if cls == 'book':
                conf = 0.5
            if cls == 'car':
                conf = 0.5

            if cls == 'person':
                conf = 0.9

            if cls == 'cup' :
                conf = 0.9

            if cls == 'clock' :
                conf = 0.9

            if cls == 'truck' :
                conf = 0.9

            if cls == 'refrigerator' :
                conf = 0.9

            if cls == 'cat' :
                conf = 0.9

            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            boxes_scores = vis_detections(im, cls, dets, thresh=conf)
            if boxes_scores is not None:
                rslt.append((boxes_scores, cls))
        
        return rslt,object_boxes

def find_obj(object_boxes, box ,cls):
    """Find The closest objects box"""
    if object_boxes == []:
        object_boxes.append(collections.deque(maxlen=smoothing_window))
        return 0

    for ind, exist_cls_box in enumerate(object_boxes):
        exist_cls, exist_box, _  = exist_cls_box[0]

        #print exist_cls
        if cls != exist_cls:
            continue
        exist_box = [x[1] for x in exist_cls_box]
        avg_box = np.mean(exist_box, axis = 0)

        if (abs(box-avg_box) < 150).all() or iou(box, avg_box) > 0.6:
            #print 'the same object!!'
            return ind


    # New objects detected
    ind += 1
    object_boxes.append(collections.deque(maxlen=smoothing_window))
    return ind

def crop_refine(frame, box, net):
    cropped_frame = frame[box[1]:box[3], box[0]:box[2]]
    #cv2.imshow('cropped', cropped_frame)
    refined, _ = demo(net,cropped_frame)
    #print refined
    if len(refined) == 0:
        return False

    for box_scores_refined, cls_refined in refined:
        for box_refined, score_refined in box_scores_refined:
            x1, y1, x2, y2 = box[0] + box_refined[0], box[1] + box_refined[1], \
                box[0] + box_refined[2], box[1] + box_refined[3]
            box_refined = x1, y1, x2, y2
            if not crop_refine(frame, box_refined, net):
                draw_bbox_class(frame, box_refined, cls_refined)
    return True

def draw_bbox_class(frame, box, cls):
    cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), (255,255,255), 2)
    cv2.putText(frame, cls,
        ( int ((box[0] + box[2])/2),
        int ((box[1] + box[3])/2) ),
        font, 1 , (0,152,0), 2)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='zf')
    parser.add_argument('--video', dest='video_mode', help='Video mode', default=False, type=bool)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        im_detect(net, im)

    import time
    # Start here
    vid_path = os.path.join(cfg.ROOT_DIR, 'tools', 'demo.mp4')
    img_path = os.path.join(cfg.ROOT_DIR, 'test.jpg')

    if args.video_mode:
        cap = skvideo.io.VideoCapture(vid_path)
    else:
        cap = cv2.VideoCapture(-1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    '''Faster R-CNN result'''
    detected = []

    '''Time measurement variables'''
    current = time.time()
    previous = current

    '''Tracker utility'''
    #keep reference of object trackers
    Trackers = []

    '''Class Heuristic'''
    PrevClass = []
    while (True):
        
        ret, frame = cap.read()

        if (ret == True):
            modified_frame = frame

            # VideoCapture from skvideo.io has different RGB encoding
            if args.video_mode:
                modified_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            key = cv2.waitKey(20) & 0xFF
            current = time.time() 

            if key == ord('q'):
                break

            #poll every 500 ms
            if  (current - previous) > 0.5:
                previous = current            
                detected,test = demo(net, modified_frame, PrevClass)

                #reset tracker
                Trackers = []

                #reset prevClass
                PrevClass = []

                for box_scores, cls in detected:
                    for box, score in box_scores:

                        #ignores small objects
                        if((box[2]-box[0])*(box[3]-box[1]) > 20000):
                            tuple_box = (box[0],box[1],box[2]-box[0],box[3]-box[1])
                            tracker = cv2.Tracker_create("MIL")
                            Trackers.append((cls,tracker))
                            PrevClass.append(cls)
                            ok = tracker.init(modified_frame,tuple_box)

                            #if not crop_refine(modified_frame, box, net):
                            draw_bbox_class(modified_frame,box,cls)

            else:
                for cls, tracker in Trackers:
                    ok, tuple_box = tracker.update(modified_frame)

                    box = tuple_box[0],tuple_box[1],\
                    tuple_box[0]+tuple_box[2],tuple_box[1]+tuple_box[3]

                    draw_bbox_class(modified_frame,box,cls)

            
            cv2.imshow('Video', modified_frame)

    cap.release()
    cv2.imwrite(img_path, frame)
    cv2.destroyAllWindows()

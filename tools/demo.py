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
import argparse
import collections

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        #print "no " + class_name+ " detected."
        return
    boxes_scores = []

    print ("{:d} {:s}s detected").format(len(inds), class_name)
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print ("Score for {:s} {:d} is {:.3f} ").format(class_name, i, score)
        boxes_scores.append((bbox,score))

    return boxes_scores

def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    if im != None:
        object_scores, object_boxes, scores, boxes, feature_maps = im_detect(net, im)
        features = feature_maps.reshape((256,36,64))


        #EXPERIMENT CHECK ACTIVATED CONVMAPS FEATURES
        activated = np.transpose(np.nonzero(features))
        #print activated[0].size
        #print activated[1].size
        #print activated.shape
        #print activated
        #print activated
        #print activated

        timer.toc()
        print ('\nDetection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.7*np.ones((1,len(CLASSES)),dtype=np.float32)
        #PERSON
        CONF_THRESH[0,15] = 0.9
        #BOTTLE
        CONF_THRESH[0,5] = 0.5

        NMS_THRESH = 0.2
        rslt = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            boxes_scores = vis_detections(im, cls, dets, thresh=CONF_THRESH[0,i])
            if boxes_scores is not None:
                rslt.append((boxes_scores, cls))

        return rslt,object_boxes

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
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
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
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        im_detect(net, im)

    import time
    # Start here
    vid = os.path.join(cfg.ROOT_DIR, 'tools', 'demo.avi')
    cap = cv2.VideoCapture(0)
    detected = []
    test = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    current = time.time()
    previous = current
    n = 0;
    object_boxes = collections.deque(maxlen=10)
    while (True):
        ret, frame = cap.read()

        key = cv2.waitKey(1) & 0xFF
        current = time.time() 

        if key == ord('q'):
            break
        if  (current - previous) > 1:
            previous = current            
            detected,test = demo(net, frame)
            object_boxes.append(test[1,:]);

        for x in detected:
            for y in x[0]:
                # Draw detected objects and the type
                cv2.rectangle(frame, (y[0][0], y[0][1]), 
                    (y[0][2], y[0][3]), (0,255,0), 3)        
                cv2.putText(frame, x[1],
                   ( int ((y[0][0] + y[0][2])/2),
                     int ((y[0][1] + y[0][3])/2) ),
                   font, 1 , (199,175,152), 2, cv2.CV_AA)

        n = len(object_boxes)
        sum = [];
        #for x in object_boxes:
        test = [0,0,0,0]
        #print object_boxes
        #for boxes in object_boxes:
            #test += boxes/n
            #print test
            #for y in boxes[0:2]:
                #print 'haha'
                #cv2.rectangle(frame, (y[0], y[1]),
                    #(y[2], y[3]), (0,0,255), 3)
        #print test
        #cv2.rectangle(frame, (int(test[0]), int(test[1])),
        #            (int(test[2]), int(test[3])), (0,0,255), 3)
        cv2.imshow('Video', frame)

    cap.release()
    cv2.destroyAllWindows()

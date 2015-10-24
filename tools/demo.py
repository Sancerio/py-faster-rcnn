#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
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
        print "no " + class_name+ " detected."
        return
    boxes_scores = []
    #scores = []
    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        boxes_scores.append((bbox,score))
        #scores.append(score)
        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=3.5)
        #     )

        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')
    return boxes_scores

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, image_name + '.jpg')
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    if im != None:
        #rbox, rscore = im_proposals(net, im)
        scores, boxes = im_detect(net, im)#, rbox)
        timer.toc()
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        rslt = []
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            boxes_scores = vis_detections(im, cls, dets, thresh=CONF_THRESH)
            if boxes_scores is not None:
                rslt.append((boxes_scores, cls))


    #if not rslt:
        print rslt
        return rslt

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
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
        _, _= im_detect(net, im)

    import time
    # Start here
    cap = cv2.VideoCapture(0)
    detected = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    current = time.time()
    previous = current
    #init = True
    while (True):
        ret, frame = cap.read()
        

        key = cv2.waitKey(1) & 0xFF
        current = time.time() 

        if key == ord('q'):
            break
        if  (current - previous) > 1:
            cv2.imwrite(cfg.ROOT_DIR + '/' + 'frame' + '.jpg', frame)
            image_path = [cfg.ROOT_DIR + '/' + 'frame' + '.jpg']
            #print image_path
            previous = current
            #if init:
                #init = False
        
            #get_windows(image_path)
            #print("Processed {} images in {:.3f} s".format(
                #len(image_path), time.time() - t))
            
            detected = demo(net, 'frame')

            #plt.show()

        for x in detected:
            #print x[1],x[0][0][0]
            #,x[1]
            for y in x[0]:
                #print y[0][0]
                #print y[1]
                cv2.rectangle(frame, (y[0][0], y[0][1]), 
                    (y[0][2], y[0][3]), (0,255,0), 3)        
                cv2.putText(frame, x[1],# + str(y[1]), 
                    ( int ((y[0][0] + y[0][2])/2), 
                      int ((y[0][1] + y[0][3])/2) ), 
                    font, 1 , (255,0,0), 2, cv2.CV_AA)

        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()

#!/usr/bin/env python

# --------------------------------------------------------
# Fast/er/ R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Generate RPN proposals."""

import _init_paths
import numpy as np
import cv2
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import im_proposals
import cPickle
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id
    cap = cv2.VideoCapture(1)

    # RPN test settings
    cfg.TEST.RPN_PRE_NMS_TOP_N = -1
    cfg.TEST.RPN_POST_NMS_TOP_N = 10
    TIME_OUT = 4
    cfg.TEST.NMS = 0.1
    cfg.TEST.RPN_NMS_THRESH = 0.0
    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    previous = time.time()
    avgbox = []
    boxes = []
    restart = True

    import time
    while (True):
        
        ret, frame = cap.read()

        if ret == True:

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break

            current = time.time()
            #if key == ord('c') :

            if key == ord('s'):#current - previous > 0.5 :
                boxes, scores = im_proposals(net, frame)
                high_conf = np.where(scores > 0.95)
                scores = scores[high_conf]
                boxes = boxes[high_conf,:]

                print scores

                if boxes != [] :
                    restart = False
                    tracker = cv2.Tracker_create("MIL")                    
                    avgbox = np.dot(np.transpose(scores), boxes)/np.sum(scores)
                    avgbox = avgbox[0]
                    tuple_box = (avgbox[0],avgbox[1],avgbox[2]-avgbox[0],avgbox[3]-avgbox[1])
                    ok = tracker.init(frame,tuple_box)

                previous = time.time()

            elif not restart:
                ok, tuple_box = tracker.update(frame)
                #print ok,tuple_box
                avgbox = tuple_box[0],tuple_box[1],\
                tuple_box[0]+tuple_box[2],tuple_box[1]+tuple_box[3]

                cv2.rectangle(frame, (int(avgbox[0]),int(avgbox[1])), (int(avgbox[2]),int(avgbox[3])), \
                (0,1,255), 2)
            
            if(current - previous > TIME_OUT):
                restart = True

            cv2.imshow("frame", frame)

    # import re
    # directory = 'data/msra/JPEGImages'

    # for im in os.listdir(directory):
    #     image = cv2.imread(os.path.join(directory, im))
    #     boxes, scores = im_proposals(net, image)

    #     high_conf = np.where(scores > 0.8)
    #     scores = scores[high_conf]
    #     boxes = boxes[high_conf,:]
        
    #     avgbox = np.dot(np.transpose(scores), boxes)/np.sum(scores)
    #     avgbox = avgbox[0]
    #     print avgbox
    #     gt = 'data/msra/Annotations'
    #     gt = os.path.join(gt, im.split('.',1)[0] + '.txt')
    #     with open(gt) as f:
    #         data = f.read()
    #     objs = re.findall('\d+', data)
    #     gt = int(objs[0]), int(objs[1]), int(objs[2]), int(objs[3])
    #     print gt
    #     #avgbox = np.sum(boxes, axis = 0)
    #     # for box in boxes:
    #     #     for box in box:
    #     #         cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), \
    #     #         (0,1,255), 1)
    #     cv2.rectangle(image, (gt[0],gt[1]), (gt[2],gt[3]), \
    #         (0,255,0), 2)
    #     cv2.rectangle(image, (avgbox[0],avgbox[1]), (avgbox[2],avgbox[3]), \
    #         (0,1,255), 2)
    #     cv2.imshow("frame", image)
    #     cv2.waitKey(0)
    # imdb = get_imdb(args.imdb_name)
    # imdb_boxes = imdb_proposals(net, imdb)

    # # output_dir = os.path.dirname(args.caffemodel)
    # output_dir = get_output_dir(imdb, net)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # rpn_file = os.path.join(output_dir, net.name + '_rpn_proposals.pkl')
    # with open(rpn_file, 'wb') as f:
    #     cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)
    # print 'Wrote RPN proposals to {}'.format(rpn_file)

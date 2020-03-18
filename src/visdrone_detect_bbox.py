from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import json
import math
import random
import numpy as np 
import skimage.io 
import matplotlib 
import matplotlib.pyplot as plt

import _init_paths
import time
import torch

from external.nms import soft_nms

from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory

from opts import opts
from detectors.detector_factory import detector_factory
from datasets.dataset.visdrone import VISDRONE

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

valid_ids = np.arange(2, dtype=np.int32)

def to_float(x):
    return float("{:.2f}".format(x))

def convert_eval_format(all_bboxes, img_dict, image_name, id):
    # import pdb; pdb.set_trace()
    detections = []

    img_dict['images'].append({
            'file_name': image_name + str(".jpg"),
            'height': 1,
            'width': 1,
            'id': image_name
        })

    for image_id in all_bboxes:
        for cls_ind in all_bboxes[image_id]:
            category_id = valid_ids[cls_ind - 1]
            for bbox in all_bboxes[image_id][cls_ind]:
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = bbox[4]
                bbox_out = list(map(to_float, bbox[0:4]))

                # detection = {
                #     "image_id": image_id,
                #     "category_id": int(category_id),
                #     "bbox": bbox_out,
                #     "score": float("{:.2f}".format(score))
                # }

                id = id + 1
                # line = line.split("\n")[0]
                # line = line.split(",")
                # line_list = []
                #print(line)
                # for i in range(len(line)):
                #     if i <= 3:
                #         line_list.append(int(float(line[i])))
            
                img_dict['annotations'].append({
                    'segmentation': [[]],
                    'area': bbox[2] * bbox[3], # approx the area
                    'iscrowd': 0,
                    'image_id': image_name,
                    'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                    'category_id': 1,
                    'id': id,
                })

                # if len(bbox) > 5:
                #     extreme_points = list(map(self._to_float, bbox[5:13]))
                #     detection["extreme_points"] = extreme_points
                # detections.append(detection)
    return img_dict

def demo(opt, img_dict):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
        opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            #cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            print(ls)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        id = 0
        results = {}
        for (image_name) in image_names:
            ret = detector.run(image_name)
            results[image_name] = ret['results']

            img_dict = convert_eval_format(results, img_dict, image_name, id)

    for item in class_names:
        img_dict['categories'].append({
            'supercategory': "person",
            'id': 1,
            'name': item
        })

    with open('ntu_validate.json', 'w') as f:
        json.dump(img_dict, f)

if __name__ == "__main__":
    #create a dictionary to store json stuff
    #'images' : filename, height, width, id
    #'type' : 'instances'
    #'annotations': imageid, bbox, category id, id, ignore = 0

    #Class Names
    opt = opts().init()

    class_names = ['person']

    img_dict = {}
    img_dict['images'] = []
    img_dict['annotations'] = []
    img_dict['categories'] = []

    #opt.demo = os.listdir("/home/coffeemix/Desktop/NTU/test_validate_images")

    demo(opt, img_dict)

    # id = 0
    # bbox_list = []
    # for txt in anno_dir:
    #     img_dict['images'].append({
    #             'file_name': str(txt.split(".")[0]) + str(".jpg"),
    #             'height': 1,
    #             'width': 1,
    #             'id': str(txt.split(".")[0])
    #         })
    #     read_txt = open("./training_labels/" + str(txt), "r")
    #     for line in read_txt:
    #         id = id + 1
    #         line = line.split("\n")[0]
    #         line = line.split(",")
    #         line_list = []
    #         #print(line)
    #         for i in range(len(line)):
    #             if i <= 3:
    #                 line_list.append(int(float(line[i])))
        
    #         img_dict['annotations'].append({
    #             'segmentation': [[]],
    #             'area': line_list[2] * line_list[3], # approx the area
    #             'iscrowd': 0,
    #             'image_id': str(txt.split(".")[0]),
    #             'bbox': line_list,
    #             'category_id': 1,
    #             'id': id,
    #         })

    # for item in class_names:
    #     img_dict['categories'].append({
    #         'supercategory': "person",
    #         'id': 1,
    #         'name': item
    #     })

    # print(img_dict)

    # with open('vis_train.json', 'w') as f:
    #     json.dump(img_dict, f)

    #print(img_dict)



    # if success:
    #     #save orig image
    #     cv2.imwrite('images/' + str(filename) + '.jpg', image)
    #     #save image with mask
    #     mask_file_no = 1
    #     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                             class_names, r['scores'])
    #     plt.savefig('annotated_images/annotated{}.jpg'.format(mask_file_no))
    #     mask_file_no += 1

    # for i in bbox:
    #     for j in i:
    #     #add to dictionary
    #         row_list = j.tolist() #converts a numpy array to list with commas
    #         print(row_list)
    #         img_dict['annotations'].append({
    #             'imageid': str(filename),
    #             'bbox': str(row_list),
    #             'category_id': str(category_id),
    #             'id': str(id_val),
    #         })
    #         id_val += 1
    # filename += 1

    # while success:
    #     count += 1
    #     sec = sec + frame_rate
    #     sec = round(sec,2)
    #     success, image = frames_frm_video(sec)
    #     if success:
    #         #save orig image
    #         cv2.imwrite('images/' + str(filename) + '.jpg', image)
    #         results = model.detect([image], verbose=1)
    #         r = results[0]

    #         #save image with mask
    #         visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                             class_names, r['scores'])
    #         plt.savefig('annotated_images/annotated{}.jpg'.format(mask_file_no))
    #         mask_file_no += 1

    #         bbox, classes = bbox_coords(r['class_ids'], r['rois'])

    #         img_dict['images'].append({
    #                 'file_name': str(filename) + '.jpg',
    #                 'height': str(image_height),
    #                 'width': str(image_width),
    #                 'id': str(filename)
    #             })
                
    #         for i in bbox:
    #             for j in i:
    #             #add to dictionary
    #                 row_list = j.tolist() #converts a numpy array to list with commas
    #                 img_dict['annotations'].append({
    #                     'imageid': str(filename),
    #                     'bbox': str(row_list),
    #                     'category_id': str(category_id),
    #                     'id': str(id_val),
    #                 })
    #                 id_val += 1
    #         filename += 1
    #     else:
    #         break
    
    # id = 0
    # for item in class_names:
    #     img_dict['categories'].append({
    #         'supercategory': 'none',
    #         'id': id,
    #         'name': item
    #     })
    #     id += 1
    # with open('result.json', 'w') as fp:
    #     json.dump(img_dict, fp)
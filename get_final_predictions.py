import json, csv
import os, sys
import numpy as np
from functools import reduce
from PIL import Image, ImageDraw, ImageFont, ImageOps
from matplotlib import path, transforms
import pickle

BASE_DIR = 'output/test/faster_rcnn_end2end/ResNet_50_faster_rcnn_iter_45000'
IMAGE_DIR = 'FISH/data/Images/test'
IMAGESET_DIR = 'FISH/data/ImageSets'
DETECTIONS_DIR = 'output/faster_rcnn_end2end/test/ResNet_50_faster_rcnn_iter_45000/'

TYPES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
MIN_SCORE = 0.02

def extract_pkl(filename):
    # Init
    results = {}
    IMAGE_NAMES = get_image_name_list()
    for image_name in IMAGE_NAMES:
        results[image_name] = {}

    classes = ('__background__', 'ALB', 'BET', 'DOL',
               'LAG', 'OTHER', 'SHARK', 'YFT', 'BAIT')
    with open(os.path.join(BASE_DIR,filename), 'rb') as f:
        data = pickle.load(f)
    for i in range(1, len(classes)):
        type = classes[i]
        f = open(os.path.join(IMAGESET_DIR, 'test.txt'), 'r')
        for j, line in enumerate(f):
            image_name = line[:-1] + '.jpg'
            objects = data[i][j]
            for obj in objects:
                bbox = list(obj[:-1].astype(float))
                score = obj[-1].astype(float)
                if not (type in results[image_name]):
                    results[image_name][type] = []
                object = {}
                object['bbox'] = bbox
                object['score'] = score
                results[image_name][type].append(object)
    return results

def write_csv(data):
    header = ['image'] + TYPES
    image_names = sorted(data)
    print(len(image_names))
    counter = 0
    with open('submission.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for image_name in image_names:
            counter +=1
            obj = data[image_name]
            row = [image_name]
            for type in header[1:]:
                row.append(obj[type]['score'])
            writer.writerow(row)
    print(counter)

def visualize(obj, image_name, path, flip=False):
    pil_image = Image.open(os.path.join(IMAGE_DIR, image_name))
    if flip:
        pil_image = ImageOps.mirror(pil_image)
    draw = ImageDraw.Draw(pil_image)
    counter = 1
    for type in obj.keys():
        if type == 'BAIT':
            continue
        bbox = obj[type]['bbox']
        score = obj[type]['score']
        x1, y1, x2, y2 = [int(coor) for coor in bbox]
        line_color = (0, 255, 255)

        draw.line((x1, y1, x2, y1), fill=line_color, width=2)
        draw.line((x1, y1, x1, y2), fill=line_color, width=2)
        draw.line((x1, y2, x2, y2), fill=line_color, width=2)
        draw.line((x2, y1, x2, y2), fill=line_color, width=2)

        text_color = (0, 255, 255)
        font = ImageFont.truetype("/Library/Fonts/arial.ttf", 15)
        draw.text((10, 20*counter), type+' '+str(score), text_color, font)
        counter += 1

    if not os.path.exists(path):
        os.makedirs(path)
    pil_image.save(os.path.join(path, image_name))


def postprocess(final_results):
    for image_name in final_results.keys():
        obj = final_results[image_name]
        for type in TYPES:
            if not type in obj:
                obj[type] = {}
                obj[type]['score'] = MIN_SCORE
                obj[type]['bbox'] = [0,0,1,1]
            elif obj[type]['score'] < MIN_SCORE:
                obj[type]['score'] = MIN_SCORE

    return final_results


def clip_boxes(box0, box1):
    path_coords = np.array([[box0[0, 0], box0[0, 1]],
                            [box0[1, 0], box0[0, 1]],
                            [box0[1, 0], box0[1, 1]],
                            [box0[0, 0], box0[1, 1]]])
    poly = path.Path(np.vstack((path_coords[:, 0], path_coords[:, 1])).T, closed=True)
    clip_rect = transforms.Bbox(box1)
    poly_clipped = poly.clip_to_bbox(clip_rect).to_polygons()
    if len(poly_clipped) == 0:
        return np.array([[0, 0], [0, 0]])
    else:
        poly_clipped = poly_clipped[0]
    return np.array([np.min(poly_clipped, axis=0), np.max(poly_clipped, axis=0)])


def suppress_bait(object):
    if not 'BAIT' in object.keys():
        return
    area = lambda x: (x[1][0]-x[0][0])*(x[1][1]-x[0][1])
    overlap = {}
    baits = object['BAIT']
    del object['BAIT']
    return # comment it if you want to use smart bait supprassion
    for bait in baits:
        if bait['score'] < 0.15:
            continue
        bait_bbox = np.array([[bait['bbox'][0], bait['bbox'][1]],
                              [bait['bbox'][2], bait['bbox'][3]]])
        for type in object.keys():
            new_obj_list = []
            for obj in object[type]:
                obj_bbox = np.array([[obj['bbox'][0], obj['bbox'][1]],
                                     [obj['bbox'][2], obj['bbox'][3]]])
                overlap_bbox = clip_boxes(obj_bbox, bait_bbox)
                overlap['overlap/object'] = area(overlap_bbox) / area(obj_bbox)
                overlap['overlap/bait'] = area(overlap_bbox) / area(bait_bbox)
                if overlap['overlap/object'] < 0.1 or overlap['overlap/bait'] < 0.8:
                    new_obj_list.append(obj)
            object[type] = new_obj_list


def get_max_recall(object):
    new_object = {}
    for type in object.keys():
        if type == 'BAIT':
            new_object[type] = object[type]
            continue
        max_obj = {}
        max_score = 0
        for obj in object[type]:
            if obj['score'] > max_score:
                max_score = obj['score']
                max_obj = obj
        if max_obj:
            new_object[type] = max_obj
    return new_object


def get_image_name_list():
    image_names = []
    for image_name in os.listdir(IMAGE_DIR):
        if image_name.endswith('.jpg'):
            image_names.append(image_name)
    return image_names


def process_results(filename):
    results = extract_pkl(filename)
    # Suppress bait
    for image_name in results.keys():
        obj = results[image_name]
        suppress_bait(obj)
        results[image_name] = get_max_recall(obj)

    # Add NoF
    for image_name in results.keys():
        obj = results[image_name]
        scores = [obj[type]['score'] for type in obj.keys()]
        obj['NoF'] = {}
        obj['NoF']['bbox'] = [0., 10., 1., 11.]
        if len(scores) == 0:
            obj['NoF']['score'] = 1.0
        else:
            not_scores = [1-score for score in scores]
            obj['NoF']['score'] = reduce(lambda x, y: x*y, not_scores)
        scores.append(obj['NoF']['score'])
        # Normalize
        sum = reduce(lambda x, y: x+y, scores)
        for type in obj.keys():
            if type == 'BAIT':
                continue
            obj[type]['score'] = obj[type]['score'] / sum

    #Postprocessing: set min score
    return postprocess(results)


# Main
if __name__ == '__main__':
    results = process_results(os.path.join(DETECTIONS_DIR,'detections.pkl'))
    flipped_results = process_results(os.path.join(DETECTIONS_DIR,'detections_flip.pkl'))

    #visualize Origin
    #for image_name in results.keys():
    #   obj = results[image_name]
    #   visualize(obj, image_name, 'vis_results')

    final_results = {}
    for image_name in results.keys():
        obj = results[image_name]
        flipped_obj = flipped_results[image_name]
        final_obj = obj.copy()
        for type in obj.keys():
            score = obj[type]['score']
            flipped_score = flipped_obj[type]['score']
            final_obj[type]['score'] = (score + flipped_score) / 2
        final_results[image_name] = final_obj

    # Save CSV
    write_csv(final_results)


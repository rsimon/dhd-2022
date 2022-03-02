import json
import os
from datetime import datetime
import pprint
import numpy as np
from shapely.geometry import Polygon
from pycocotools import coco
import cv2
from PIL import Image as PILImage
from io import BytesIO

import random
import requests
from math import trunc
# code for visualization of cocodatasets from https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
class CocoDataset():
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.colors = ['blue', 'purple', 'red', 'green', 'orange', 'salmon', 'pink', 'gold',
                       'orchid', 'slateblue', 'limegreen', 'seagreen', 'darkgreen', 'olive',
                       'teal', 'aquamarine', 'steelblue', 'powderblue', 'dodgerblue', 'navy',
                       'magenta', 'sienna', 'maroon']

        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()

        self.process_info()
        self.process_licenses()
        self.process_categories()
        self.process_images()
        self.process_segmentations()

    def display_info(self):
        print('Dataset Info:')
        print('=============')
        if self.info is None:
            return
        for key, item in self.info.items():
            print('  {}: {}'.format(key, item))

        requirements = [['description', str],
                        ['url', str],
                        ['version', str],
                        ['year', int],
                        ['contributor', str],
                        ['date_created', str]]
        for req, req_type in requirements:
            if req not in self.info:
                print('ERROR: {} is missing'.format(req))
            elif type(self.info[req]) != req_type:
                print('ERROR: {} should be type {}'.format(req, str(req_type)))
        print('')

    def display_licenses(self):
        print('Licenses:')
        print('=========')

        if self.licenses is None:
            return
        requirements = [['id', int],
                        ['url', str],
                        ['name', str]]
        for license in self.licenses:
            for key, item in license.items():
                print('  {}: {}'.format(key, item))
            for req, req_type in requirements:
                if req not in license:
                    print('ERROR: {} is missing'.format(req))
                elif type(license[req]) != req_type:
                    print('ERROR: {} should be type {}'.format(
                        req, str(req_type)))
            print('')
        print('')

    def display_categories(self):
        print('Categories:')
        print('=========')
        for sc_key, sc_val in self.super_categories.items():
            print('  super_category: {}'.format(sc_key))
            for cat_id in sc_val:
                print('    id {}: {}'.format(
                    cat_id, self.categories[cat_id]['name']))
            print('')

    def display_image(self, image_id, show_polys=True, show_bbox=True, show_crowds=True, use_url=False):
        print('Image:')
        print('======')
        if image_id == 'random':
            image_id = random.choice(list(self.images.keys()))

        # Print the image info
        image = self.images[image_id]
        for key, val in image.items():
            print('  {}: {}'.format(key, val))

        # Open the image
        if use_url:
            image_path = image['coco_url']
            response = requests.get(image_path)
            image = PILImage.open(BytesIO(response.content))

        else:
            image_path = os.path.join(self.image_dir, image['file_name'])
            image = PILImage.open(image_path)

        # Calculate the size and adjusted display size
        max_width = 600
        image_width, image_height = image.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height

        # Create list of polygons to be drawn
        polygons = {}
        bbox_polygons = {}
        rle_regions = {}
        poly_colors = {}
        bbox_categories = {}
        print('  segmentations ({}):'.format(
            len(self.segmentations[image_id])))
        for i, segm in enumerate(self.segmentations[image_id]):
            polygons_list = []
            if segm['iscrowd'] != 0:
                # Gotta decode the RLE
                px = 0
                x, y = 0, 0
                rle_list = []
                for j, counts in enumerate(segm['segmentation']['counts']):
                    if j % 2 == 0:
                        # Empty pixels
                        px += counts
                    else:
                        # Need to draw on these pixels, since we are drawing in vector form,
                        # we need to draw horizontal lines on the image
                        x_start = trunc(
                            trunc(px / image_height) * adjusted_ratio)
                        y_start = trunc(px % image_height * adjusted_ratio)
                        px += counts
                        x_end = trunc(trunc(px / image_height)
                                      * adjusted_ratio)
                        y_end = trunc(px % image_height * adjusted_ratio)
                        if x_end == x_start:
                            # This is only on one line
                            rle_list.append(
                                {'x': x_start, 'y': y_start, 'width': 1, 'height': (y_end - y_start)})
                        if x_end > x_start:
                            # This spans more than one line
                            # Insert top line first
                            rle_list.append(
                                {'x': x_start, 'y': y_start, 'width': 1, 'height': (image_height - y_start)})

                            # Insert middle lines if needed
                            lines_spanned = x_end - x_start + 1  # total number of lines spanned
                            full_lines_to_insert = lines_spanned - 2
                            if full_lines_to_insert > 0:
                                full_lines_to_insert = trunc(
                                    full_lines_to_insert * adjusted_ratio)
                                rle_list.append(
                                    {'x': (x_start + 1), 'y': 0, 'width': full_lines_to_insert, 'height': image_height})

                            # Insert bottom line
                            rle_list.append(
                                {'x': x_end, 'y': 0, 'width': 1, 'height': y_end})
                if len(rle_list) > 0:
                    rle_regions[segm['id']] = rle_list
            else:
                # Add the polygon segmentation
                for segmentation_points in segm['segmentation']:
                    segmentation_points = np.multiply(
                        segmentation_points, adjusted_ratio).astype(int)
                    polygons_list.append(
                        str(segmentation_points).lstrip('[').rstrip(']'))
            polygons[segm['id']] = polygons_list
            if i < len(self.colors):
                poly_colors[segm['id']] = self.colors[i]
            else:
                poly_colors[segm['id']] = 'white'

            bbox = segm['bbox']
            bbox_points = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                           bbox[0] + bbox[2], bbox[1] +
                           bbox[3], bbox[0], bbox[1] + bbox[3],
                           bbox[0], bbox[1]]
            bbox_points = np.multiply(bbox_points, adjusted_ratio).astype(int)
            bbox_polygons[segm['id']] = str(
                bbox_points).lstrip('[').rstrip(']')
            bbox_categories[segm['id']] = self.categories[segm['category_id']]
            # Print details
            print('    {}:{}:{}'.format(
                segm['id'], poly_colors[segm['id']], self.categories[segm['category_id']]))

        # Draw segmentation polygons on image
        html = '<div class="container" style="position:relative;">'
        html += '<img src="{}" style="position:relative;top:0px;left:0px;width:{}px;">'.format(
            image_path, adjusted_width)
        html += '<div class="svgclass"><svg width="{}" height="{}">'.format(
            adjusted_width, adjusted_height)

        if show_polys:
            for seg_id, points_list in polygons.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for points in points_list:
                    html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5" />'.format(
                        points, fill_color, stroke_color)

        if show_crowds:
            for seg_id, rect_list in rle_regions.items():
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                for rect_def in rect_list:
                    x, y = rect_def['x'], rect_def['y']
                    w, h = rect_def['width'], rect_def['height']
                    html += '<rect x="{}" y="{}" width="{}" height="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0.5; stroke-opacity:0.5" />'.format(
                        x, y, w, h, fill_color, stroke_color)

        if show_bbox:
            for seg_id, points in bbox_polygons.items():
                x, y = [int(i) for i in points.split()[:2]]
                html += '<text x="{}" y="{}" fill="yellow">{}</text>'.format(
                    x, y, bbox_categories[seg_id]["name"])
                fill_color = poly_colors[seg_id]
                stroke_color = poly_colors[seg_id]
                html += '<polygon points="{}" style="fill:{}; stroke:{}; stroke-width:1; fill-opacity:0" />'.format(
                    points, fill_color, stroke_color)

        html += '</svg></div>'
        html += '</div>'
        html += '<style>'
        html += '.svgclass { position:absolute; top:0px; left:0px;}'
        html += '</style>'
        return html

    def process_info(self):
        self.info = self.coco.get('info')

    def process_licenses(self):
        self.licenses = self.coco.get('licenses')

    def process_categories(self):
        self.categories = {}
        self.super_categories = {}
        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']

            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print("ERROR: Skipping duplicate category id: {}".format(category))

            # Add category to super_categories dict
            if super_category not in self.super_categories:
                # Create a new set with the category id
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {
                    cat_id}  # Add category id to the set

    def process_images(self):
        self.images = {}
        for image in self.coco['images']:
            image_id = image['id']
            if image_id in self.images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            else:
                self.images[image_id] = image

    def process_segmentations(self):
        self.segmentations = {}
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)


pp = pprint.PrettyPrinter(indent=4)
# image-path to the Images, which where annotated
imagepath = "/home/erik/DHd Workshop 2022/images/"
# input for the annotation file
with open('/home/erik/Downloads/annotations.json', 'r') as f:
  data = json.load(f)

now = datetime.now()
#Building up coco-annotation dataset
annotation={}
categorieIDs={}
annotation["info"]={}
annotation["info"]["date_created"]= now.strftime("%m/%d/%Y")
annotation["info"]["version"]="1"
annotation["info"]["description"]="Kucha Corpus, Version v1"
annotation["info"]["contributor"]="SAW-Leipzig"
annotation["info"]["url"]="kucha.saw-leipzig.de"
annotation["info"]["year"]=2020
annotation["licenses"]=[]
license={}
license["name"]="NONE"
license["url"]="Don't distribute"
license["id"]=1
annotation["licenses"].append(license)
annotation["annotations"]=[]
annotation["images"]=[]
annotation["categories"]=[]
annotationID=0
images = {}
#Start filling the Annotation into the coco-dataset
for key in data:
    #filling in images
    print("processing image: ",key)
    image = cv2.imread(imagepath + key)
    filename = imagepath + key
    images[len(images)] = len(images)
    print("imageID:", len(images))
    # we need this, because Detectron cannot read tiff
    if key.endswith("tiff") or key.endswith("tif"):
        new_path = "/media/erik/DATA/kucha_images/"
        filename = filename.split(".")[0] + ".jpg"
        cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 200])
    # create image entry in coco format
    image = {
        "license": 1,
        "coco_url": "",
        "flickr_url": "",
        "id": len(images),
        "file_name": filename,
        "date_captured": "2019-02-18 12:15:34",
        "height": image.shape[0],
        "width": image.shape[1]
    }
    annotation["images"].append(image)
    # accessing annotations
    for annotationKey in data[key]:
        print("found annotation", annotationKey)
        # process annotation only if polygon
        if 'polygon' in data[key][annotationKey]['target']['selector']['value']:
            for body in data[key][annotationKey]["body"]:
                # if tag is not in categories, a new category in cocoformat will be added
                if body['value'] not in categorieIDs:
                    category= {
                        "id": len(categorieIDs),
                        "name": body['value'],
                        "supercategory": "Kucha"
                    }
                    categorieIDs[body['value']] = len(categorieIDs)
                    annotation["categories"].append(category)
                # change polygon format to coco-format
                svg = data[key][annotationKey]['target']['selector']['value']
                points = svg.replace("<svg><polygon points=\"","").replace("\"></polygon></svg>","")
                points = points.split(" ")
                polygonPoints = []
                for point in points:
                    strCoord = point.split(",")
                    numCoord = [float(strCoord[0]), float(strCoord[1])]
                    polygonPoints.append(numCoord)
                print(polygonPoints)
                polygon = Polygon(polygonPoints)
                print(polygon)
                segmentation = []
                segmentation.append(np.array(polygon.exterior.coords).ravel().tolist())
                # print(segmentation)
                x, y, max_x, max_y = polygon.bounds
                width = max_x - x
                height = max_y - y
                bbox = (x, y, width, height)
                area = polygon.area
                # create annotation in coco-format
                anno = {
                    'segmentation': segmentation,
                    'iscrowd': 0,
                    'image_id': len(images),
                    'category_id': categorieIDs[body['value']],
                    'id': annotationID,
                    'bbox': bbox,
                    'area': area
                }
                print(anno)
                annotation["annotations"].append(anno)
                annotationID += 1

pp.pprint(annotation)
with open(os.path.join("train.json"), 'w') as fo:
    json.dump(annotation, fo, indent=2)
train_data_set = coco.COCO(os.path.join("train.json"))
cats = train_data_set.loadCats(train_data_set.getCatIds())
print(cats)

nms = set([cat['supercategory'] for cat in cats])
print('COCO categories: \n{}\n'.format(' '.join(nms)))
print('COCO supercategories: \n{}'.format(' '.join(nms)))
print("CatIDs: ",train_data_set.getCatIds())
imgIds = train_data_set.getImgIds();
imgs = train_data_set.loadImgs(imgIds)
imgid=0
annotation_path = 'train.json'
image_dir = ''
# part for data visualization of generated train.json in coco format
coco_dataset = CocoDataset(annotation_path, image_dir)
coco_dataset.display_info()
coco_dataset.display_licenses()
coco_dataset.display_categories()
print("-----------------------------------detected images:",imgIds)
html=''
for imgID in imgIds:
    print("processing picture: ",imgID)
    html = html + "<H1>"+str(imgID)+"</H1>"
    try:
        html = html + coco_dataset.display_image(imgID, use_url=False)
    except:
        print("could not process image ", imgID)
    #print(html)
with open(os.path.join("Results.tml"), 'w') as fo:
    fo.write(html)
fo.close()

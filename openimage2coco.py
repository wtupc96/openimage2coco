import os
import cv2
import pdb
import json
import shutil
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import argparse
from pycocotools import mask
from pycocotools.coco import COCO
from skimage import measure

try:
    import moxing as mox
    mox.file.shift('os', 'mox')
except:
    pass

WIDTH = 50

COCO_CLASS_NAMES_ID_DIC = {
        'person': '1', 'bicycle': '2', 'car': '3', 'motorcycle': '4', 'airplane': '5', 'bus': '6', 'train': '7', 'truck': '8', 'boat': '9', 'traffic light': '10', 
        'fire hydrant': '11', 'stop sign': '13', 'parking meter': '14', 'bench': '15', 'bird': '16', 'cat': '17', 'dog': '18', 'horse': '19', 'sheep': '20', 'cow': '21', 
        'elephant': '22', 'bear': '23', 'zebra': '24', 'giraffe': '25', 'backpack': '27', 'umbrella': '28', 'handbag': '31', 'tie': '32', 'suitcase': '33', 'frisbee': '34', 
        'skis': '35', 'snowboard': '36', 'sports ball': '37', 'kite': '38', 'baseball bat': '39', 'baseball glove': '40', 'skateboard': '41', 'surfboard': '42', 'tennis racket': '43', 'bottle': '44', 
        'wine glass': '46', 'cup': '47', 'fork': '48', 'knife': '49', 'spoon': '50', 'bowl': '51', 'banana': '52', 'apple': '53', 'sandwich': '54', 'orange': '55', 
        'broccoli': '56', 'carrot': '57', 'hot dog': '58', 'pizza': '59', 'donut': '60', 'cake': '61', 'chair': '62', 'couch': '63', 'potted plant': '64', 'bed': '65', 
        'dining table': '67', 'toilet': '70', 'tv': '72', 'laptop': '73', 'mouse': '74', 'remote': '75', 'keyboard': '76', 'cell phone': '77', 'microwave': '78', 'oven': '79', 
        'toaster': '80', 'sink': '81', 'refrigerator': '82', 'book': '84', 'clock': '85', 'vase': '86', 'scissors': '87', 'teddy bear': '88', 'hair drier': '89', 'toothbrush': '90'
}

COCO_OPENIMAGE_RELATED_CLASSES_NAME_DIC = {
        'person': ['Person'], 'bicycle': ['Stationary bicycle', 'Bicycle'], 'car': ['Car', 'Limousine', 'Van', 'Vehicle', 'Land vehicle', 'Ambulance', 'Cart', 'Golf cart'], 'motorcycle': ['Motorcycle'], 'airplane': ['Airplane'], 'bus': ['Bus'], 'train': ['Train'], 'truck': ['Truck'], 'boat': ['Boat', 'Barge', 'Gondola', 'Canoe', 'Jet ski', 'Submarine'], 'traffic light': ['Traffic light'], 
        'fire hydrant': ['Fire hydrant'], 'stop sign': ['Stop sign'], 'parking meter': ['Parking meter'], 'bench': ['Bench'], 'bird': ['Magpie', 'Woodpecker', 'Blue jay', 'Ostrich', 'Penguin', 'Raven', 'Chicken', 'Eagle', 'Owl', 'Duck', 'Canary', 'Goose', 'Swan', 'Falcon', 'Parrot', 'Sparrow', 'Turkey'], 'cat': ['Cat'], 'dog': ['Dog'], 'horse': ['Horse'], 'sheep': ['Sheep'], 'cow': ['Cattle', 'Bull'], 
        'elephant': ['Elephant'], 'bear': ['Bear', 'Brown bear', 'Panda', 'Polar bear'], 'zebra': ['Zebra'], 'giraffe': ['Giraffe'], 'backpack': ['Backpack'], 'umbrella': ['Umbrella'], 'handbag': ['Handbag'], 'tie': ['Tie'], 'suitcase': ['Suitcase'], 'frisbee': ['Flying disc'], 
        'skis': ['Ski'], 'snowboard': ['Snowboard'], 'sports ball': ['Ball', 'Football', 'Cricket ball', 'Volleyball', 'Tennis ball', 'Rugby ball', 'Golf ball'], 'kite': ['Kite'], 'baseball bat': ['Baseball bat'], 'baseball glove': ['Baseball glove'], 'skateboard': ['Skateboard'], 'surfboard': ['Surfboard'], 'tennis racket': ['Tennis racket'], 'bottle': ['Bottle'], 
        'wine glass': ['Wine glass'], 'cup': ['Coffee cup', 'Measuring cup'], 'fork': ['Fork'], 'knife': ['Kitchen knife', 'Knife'], 'spoon': ['Spoon'], 'bowl': ['Mixing bowl', 'Bowl'], 'banana': ['Banana'], 'apple': ['Apple'], 'sandwich': ['Sandwich', 'Hamburger', 'Submarine sandwich'], 'orange': ['Orange'], 
        'broccoli': ['Broccoli'], 'carrot': ['Carrot'], 'hot dog': ['Hot dog'], 'pizza': ['Pizza'], 'donut': ['Doughnut'], 'cake': ['Cake'], 'chair': ['WheelChair', 'Chair'], 'couch': ['Couch', 'Sofa bed', 'Loveseat', 'studio couch'], 'potted plant': ['Houseplant'], 'bed': ['Bed', 'Infant bed'], 
        'dining table': ['Table', 'Coffee table', 'Kitchen & dining room table'], 'toilet': ['Toilet'], 'tv': ['Television'], 'laptop': ['Laptop'], 'mouse': ['Computer mouse'], 'remote': ['Remote control'], 'keyboard': ['Computer keyboard'], 'cell phone': ['Mobile phone'], 'microwave': ['Microwave oven'], 'oven': ['Oven'], 
        'toaster': ['Toaster'], 'sink': ['Sink'], 'refrigerator': ['Refrigerator'], 'book': ['Book'], 'clock': ['Clock', 'Alarm clock', 'Digital clock', 'Wall clock'], 'vase': ['Vase'], 'scissors': ['Scissors'], 'teddy bear': ['Teddy bear'], 'hair drier': ['Hair dryer'], 'toothbrush': ['Toothbrush']
}

COCO_OPENIMAGE_RELATED_CLASSES_DIC = {
        'person': ['/m/01g317'], 'bicycle': ['/m/03kt2w', '/m/0199g'], 'car': ['/m/0k4j', '/m/01lcw4', '/m/0h2r6', '/m/07yv9', '/m/01prls', '/m/012n7d', '/m/018p4k', '/m/0323sq'], 'motorcycle': ['/m/04_sv'], 'airplane': ['/m/0cmf2'], 'bus': ['/m/01bjv'], 'train': ['/m/07jdr'], 'truck': ['/m/07r04'], 'boat': ['/m/019jd', '/m/01btn', '/m/02068x', '/m/0ph39', '/m/01xs3r', '/m/074d1'], 'traffic light': ['/m/015qff'], 
        'fire hydrant': ['/m/01pns0'], 'stop sign': ['/m/02pv19'], 'parking meter': ['/m/015qbp'], 'bench': ['/m/0cvnqh'], 'bird': ['/m/012074', '/m/01dy8n', '/m/01f8m5', '/m/05n4y', '/m/05z6w', '/m/06j2d', '/m/09b5t', '/m/09csl', '/m/09d5_', '/m/09ddx', '/m/0ccs93', '/m/0dbvp', '/m/0dftk', '/m/0f6wt', '/m/0gv1x', '/m/0h23m', '/m/0jly1'], 'cat': ['/m/01yrx'], 'dog': ['/m/0bt9lr'], 'horse': ['/m/03k3r'], 'sheep': ['/m/07bgp'], 'cow': ['/m/01xq0k1', '/m/0cnyhnx'], 
        'elephant': ['/m/0bwd_0j'], 'bear': ['/m/01dws', '/m/01dxs', '/m/03bj1', '/m/0633h'], 'zebra': ['/m/0898b'], 'giraffe': ['/m/03bk1'], 'backpack': ['/m/01940j'], 'umbrella': ['/m/0hnnb'], 'handbag': ['/m/080hkjn'], 'tie': ['/m/01rkbr'], 'suitcase': ['/m/01s55n'], 'frisbee': ['/m/02wmf'], 
        'skis': ['/m/071p9'], 'snowboard': ['/m/06__v'], 'sports ball': ['/m/018xm', '/m/01226z', '/m/02ctlc', '/m/05ctyq', '/m/0wdt60w', '/m/044r5d'], 'kite': ['/m/02zt3'], 'baseball bat': ['/m/03g8mr'], 'baseball glove': ['/m/03grzl'], 'skateboard': ['/m/06_fw'], 'surfboard': ['/m/019w40'], 'tennis racket': ['/m/0h8my_4'], 'bottle': ['/m/04dr76w'], 
        'wine glass': ['/m/09tvcd'], 'cup': ['/m/02p5f1q', '/m/07v9_z'], 'fork': ['/m/0dt3t'], 'knife': ['/m/058qzx', '/m/04ctx'], 'spoon': ['/m/0cmx8'], 'bowl': ['/m/03hj559', '/m/04kkgm'], 'banana': ['/m/09qck'], 'apple': ['/m/014j1m'], 'sandwich': ['/m/0l515', '/m/0cdn1', '/m/06pcq'], 'orange': ['/m/0cyhj_'], 
        'broccoli': ['/m/0hkxq'], 'carrot': ['/m/0fj52s'], 'hot dog': ['/m/01b9xk'], 'pizza': ['/m/0663v'], 'donut': ['/m/0jy4k'], 'cake': ['/m/0fszt'], 'chair': ['/m/01mzpv'], 'couch': ['/m/02crq1', '/m/03m3pdh', '/m/0703r8'], 'potted plant': ['/m/03fp41'], 'bed': ['/m/03ssj5', '/m/061hd_'], 
        'dining table': ['/m/04bcr3', '/m/078n6m', '/m/0h8n5zk'], 'toilet': ['/m/09g1w'], 'tv': ['/m/07c52'], 'laptop': ['/m/01c648'], 'mouse': ['/m/020lf'], 'remote': ['/m/0qjjc'], 'keyboard': ['/m/01m2v'], 'cell phone': ['/m/050k8'], 'microwave': ['/m/0fx9l'], 'oven': ['/m/029bxz'], 
        'toaster': ['/m/01k6s3'], 'sink': ['/m/0130jx'], 'refrigerator': ['/m/040b_t'], 'book': ['/m/0bt_c3'], 'clock': ['/m/01x3z', '/m/046dlr', '/m/06_72j', '/m/0h8mzrc'], 'vase': ['/m/02s195'], 'scissors': ['/m/01lsmm'], 'teddy bear': ['/m/0kmg4'], 'hair drier': ['/m/03wvsk'], 'toothbrush': ['/m/012xff']
}

def convert_dic(dic):
    new_dic = dict()
    for k, v in dic.items():
        for _v in v:
            if type(_v) == str:
                new_dic[_v] = k
            elif type(_v) == tuple:
                new_dic[_v[0]] = k
    return new_dic

COCO_OPENIMAGE_RELATED_CLASSES_DIC_CONVERT = convert_dic(COCO_OPENIMAGE_RELATED_CLASSES_DIC)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--im_root', type=str, required=True)
    parser.add_argument('--seg_im_root', type=str, required=True)
    parser.add_argument('--op_bbox_anno', type=str, default=None)
    parser.add_argument('--op_segm_anno', type=str, default=None)
    parser.add_argument('--op_cls_info', type=str, default=None, required=True)
    parser.add_argument('--which_set', type=str, default='train', choices=['train', 'test', 'val', 'all'])
    parser.add_argument('--not_conv_cls', default=False, action='store_true')

    args = parser.parse_args()
    return args


class OpenImage2COCO(object):
    def __init__(self,
                 image_root=None,
                 segm_image_root=None,
                 openimage_bbox_file=None, 
                 openimage_seg_file=None, 
                 openimage_cls_info_file=None,
                 which_set='val',
                 convert_cls_into_coco=True):

        if 'train' in which_set:
            filling_info = 'training'
        elif 'val' in which_set:
            filling_info = 'validation'
        elif 'test' in which_set:
            filling_info = 'testing'
        else:
            exit()

        self.info = {
                'description': 'This is the {} set of Open Images V5.'.format(filling_info),
                'url' : 'https://storage.googleapis.com/openimages/web/index.html',
                'version': '1.0',
                'year': 2019,
                'contributor': 'Google',
                'date_created': '2019-07-30'
            }
        self.which_set = which_set
        self.image_root = image_root
        self.segm_image_root = segm_image_root
        self.openimage_bbox_file = openimage_bbox_file
        self.openimage_seg_file = openimage_seg_file
        self.openimage_cls_info_file = openimage_cls_info_file
        self.convert_cls_into_coco = convert_cls_into_coco

        self.cls_id = 0
        self.cls_dic = dict()
        self.cls_id_dic = dict()

        cls_info = self.__read_csv(openimage_cls_info_file)

        for ci in cls_info:
            if convert_cls_into_coco:
                if ci[0] in COCO_OPENIMAGE_RELATED_CLASSES_DIC_CONVERT:
                    self.cls_dic.update({ci[0]: ci[1]})
                    self.cls_id_dic.update({ci[0]: COCO_CLASS_NAMES_ID_DIC[COCO_OPENIMAGE_RELATED_CLASSES_DIC_CONVERT[ci[0]]]})
                    self.cls_id += 1
            else:
                self.cls_dic.update({ci[0]: ci[1]})
                self.cls_id_dic.update({ci[0]: self.cls_id})
                self.cls_id += 1


    @staticmethod
    def __read_csv(name, to_start=0):
        return pd.read_csv(name, header=None, low_memory=False).values[to_start: ]


    def convert_bbox(self, save_path):
        self.save_path = save_path

        image_id = 0
        box_id = 0
        images_dic = dict()
        image_info = list()
        annos = list()
        categories = list()

        print('=' * 100)
        print('Reading bbox file...')
        bbox_annos = self.__read_csv(self.openimage_bbox_file, 1)

        print('Counting images...')
        counted_annos = list()
        for ba in tqdm(bbox_annos, ncols=WIDTH):
        # name_key: ImageID
            name_key = ba[0]
            cls_id = ba[2]

            if self.convert_cls_into_coco:
                if cls_id in COCO_OPENIMAGE_RELATED_CLASSES_DIC_CONVERT:
                    counted_annos.append(ba)
                    if not name_key in images_dic:
                        images_dic.update({name_key: image_id})
                        image_id += 1
            else:
                counted_annos.append(ba)
                if not name_key in images_dic:
                    # ImageID <==> image_id 
                    images_dic.update({name_key: image_id})
                    image_id += 1

        print('Getting image infos...')
        name_key_height_width_dic = dict()
        for name_key in tqdm(images_dic, ncols=WIDTH):
            im = cv2.imread(os.path.join(self.image_root, self.which_set, name_key + '.jpg'))
            height, width = im.shape[ :2]

            if name_key not in name_key_height_width_dic:
                name_key_height_width_dic[name_key] = (height, width)

            image = {
                    'file_name': name_key + '.jpg',
                    'height': height,
                    'width': width, 
                    'id': images_dic[name_key]
                }
            
            image_info.append(image) 

        print('Writing annotations...')
        for ba in tqdm(counted_annos, ncols=WIDTH):
            name_key = ba[0]
            height, width = name_key_height_width_dic[name_key]
                        
            bbox = [
                    float(ba[4]) * width, 
                    float(ba[6]) * height, 
                    (float(ba[5]) - float(ba[4])) * width, 
                    (float(ba[7]) - float(ba[6])) * height
            ]
            
            IsOccluded = int(ba[8])
            IsTruncated = int(ba[9])
            IsGroupOf = int(ba[10])
            IsDepiction = int(ba[11])
            IsInside = int(ba[12])

            LabelName = ba[2] 
            if LabelName in self.cls_id_dic:
                anno = {
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'image_id': images_dic[name_key],
                    'category_id': self.cls_id_dic[LabelName],
                    'iscrowd': 0, # ??? == isGroupOf?
                    'id': int(box_id),
                    'IsOccluded': IsOccluded,
                    'IsTruncated': IsTruncated,
                    'IsGroupOf': IsGroupOf,
                    'IsDepiction': IsDepiction,
                    'IsInside': IsInside
                }
                annos.append(anno)
                box_id += 1

        if self.convert_cls_into_coco:
            for k, v in COCO_CLASS_NAMES_ID_DIC.items():
                category = {
                        'supercategory': k,
                        'id': v,
                        'name': k
                }
                categories.append(category)
        else:
            for cat in self.cls_dic:
                category = {
                        'supercategory': self.cls_dic[cat],
                        'id': self.cls_id_dic[cat],
                        'name': self.cls_dic[cat]
                }
                categories.append(category)

        inputfile = {
                'info': self.info,
                'images': image_info,
                'type': 'instances',
                'annotations': annos,
                'categories': categories
        }

        print('Writing into JSON...')
        with open(save_path, 'w', encoding='utf8') as f:
            json.dump(inputfile,f)
        print('=' * 100)

    def visualize_bbox(self, sample_num=10, r_seed=3, g_seed=7, b_seed=19, save_path=None):
        if save_path:
            with open(save_path, 'r', encoding='utf8') as f:
                coco_format_file = json.load(f)
        else:
            with open(self.save_path, 'r', encoding='utf8') as f:
                coco_format_file = json.load(f)
        
        selected_images = random.choices(coco_format_file['images'], k=sample_num)
        
        for si in selected_images:
            im = cv2.imread(os.path.join(self.image_root, self.which_set, si['file_name']))

            for a in coco_format_file['annotations']:
                # print(a['image_id'], si['id'])
                if a['image_id'] == si['id']:
                    x, y, w, h = list(map(round, a['bbox']))
                    cate_id = int(a['category_id'])
                    color = ((r_seed * cate_id) % 255, (g_seed * cate_id) % 255, (b_seed * cate_id) % 255)
                    
                    cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
                    
                    cls_name = self.cls_dic[[x[0] for x in self.cls_id_dic.items() if x[1] == a['category_id']][0]]
                    if self.convert_cls_into_coco:
                        cls_name = convert_dic(COCO_OPENIMAGE_RELATED_CLASSES_NAME_DIC)[cls_name]
                    
                    cv2.putText(im, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, color=color, thickness=1)
            cv2.imshow(si['file_name'], im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def convert_segmentation(self, save_path):
        self.seg_save_path = save_path

        image_id = 0
        box_id = 0
        images_dic = dict()
        image_info = list()
        annos = list()
        categories = list()

        print('=' * 100)
        print('Reading segm file...')
        bbox_annos = self.__read_csv(self.openimage_seg_file, 1)

        print('Counting images...')
        counted_annos = list()
        for ba in tqdm(bbox_annos, ncols=WIDTH):
            # name_key: ImageID
            name_key = ba[1]
            cls_id = ba[2]

            if self.convert_cls_into_coco:
                if cls_id in COCO_OPENIMAGE_RELATED_CLASSES_DIC_CONVERT:
                    counted_annos.append(ba)
                    if not name_key in images_dic:
                        images_dic.update({name_key: image_id})
                        image_id += 1
            else:
                counted_annos.append(ba)
                if not name_key in images_dic:
                    # ImageID <==> image_id 
                    images_dic.update({name_key: image_id})
                    image_id += 1

        print('Getting image infos...')
        name_key_height_width_dic = dict()
        for name_key in tqdm(images_dic, ncols=WIDTH):
            im = cv2.imread(os.path.join(self.image_root, self.which_set, name_key + '.jpg'))
            height, width = im.shape[ :2]

            if name_key not in name_key_height_width_dic:
                name_key_height_width_dic[name_key] = (height, width)

            image = {
                    'file_name': name_key + '.jpg',
                    'height': height,
                    'width': width, 
                    'id': images_dic[name_key]
                }
            
            image_info.append(image) 

        print('Writing annotations...')
        for ba in tqdm(counted_annos, ncols=WIDTH):
            mask_key = ba[0]
            mask_img = cv2.imread(os.path.join(self.segm_image_root, self.which_set, mask_key), 0)
            
            name_key = ba[1]
            height, width = name_key_height_width_dic[name_key]

            mask_img = cv2.resize(mask_img, (width, height))
            
            mask_img[0,: ] = 0
            mask_img[-1,: ] = 0
            mask_img[:, 0] = 0
            mask_img[:, -1] = 0

            fortran_ground_truth_binary_mask = np.asfortranarray(mask_img)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            contours = measure.find_contours(mask_img, 0.5)

            bbox = [
                    float(ba[4]) * width, 
                    float(ba[6]) * height, 
                    (float(ba[5]) - float(ba[4])) * width, 
                    (float(ba[7]) - float(ba[6])) * height
            ]
            
            LabelName = ba[2]
            if LabelName in self.cls_id_dic:
                anno = {
                    'bbox': bbox,
                    'area': ground_truth_area.tolist(),
                    'image_id': images_dic[name_key],
                    'category_id': self.cls_id_dic[LabelName],
                    'iscrowd': 0,
                    'id': int(box_id),
                    'segmentation': []
                }

                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                    anno["segmentation"].append(segmentation)
                
                annos.append(anno)
                box_id += 1

        if self.convert_cls_into_coco:
            for k, v in COCO_CLASS_NAMES_ID_DIC.items():
                category = {
                        'supercategory': k,
                        'id': v,
                        'name': k
                }
                categories.append(category)
        else:
            for cat in self.cls_dic:
                category = {
                        'supercategory': self.cls_dic[cat],
                        'id': self.cls_id_dic[cat],
                        'name': self.cls_dic[cat]
                }
                categories.append(category)

        inputfile = {
                'info': self.info,
                'images': image_info,
                'type': 'instances',
                'annotations': annos,
                'categories': categories
        }

        print('Writing into JSON...')
        with open(save_path, 'w', encoding='utf8') as f:
            json.dump(inputfile,f)
        print('=' * 100)

    def visualize_segmentation(self, sample_num=10, r_seed=3, g_seed=7, b_seed=19, save_path=None):
        coco =  COCO(save_path if save_path else self.seg_save_path)
        selected_images = random.choices(coco.loadImgs(coco.getImgIds()), k=sample_num)
        
        specified_image = '6ba6e6afe9964a18.jpg'
        for i in coco.loadImgs(coco.getImgIds()):
            if i['file_name'] == specified_image:
                selected_images.append(i)
                break
            
        for si in selected_images:
            img_id = si['id']
            annos = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

            for a in annos:
                im = cv2.imread(os.path.join(self.image_root, self.which_set, si['file_name']))
                
                mask_img = coco.annToMask(a)
                im = im * np.stack([mask_img] * 3, axis=-1)

                x, y, w, h = list(map(round, a['bbox']))
                cate_id = int(a['category_id'])
                color = ((r_seed * cate_id) % 255, (g_seed * cate_id) % 255, (b_seed * cate_id) % 255)
                
                cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)

                cls_name = self.cls_dic[[x[0] for x in self.cls_id_dic.items() if x[1] == a['category_id']][0]]
                if self.convert_cls_into_coco:
                    cls_name = convert_dic(COCO_OPENIMAGE_RELATED_CLASSES_NAME_DIC)[cls_name]
                    
                cv2.putText(im, cls_name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, color=color, thickness=1)

                cv2.imshow(si['file_name'], im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    converter = OpenImage2COCO(image_root=args.im_root,
                               segm_image_root=args.seg_im_root,
                               openimage_bbox_file=args.op_bbox_anno,
                               openimage_seg_file=args.op_segm_anno,
                               openimage_cls_info_file=args.op_cls_info,
                               which_set=args.which_set
                )
    converter.convert_segmentation(save_path='openimage_coco_format_{}_segm.json'.format(args.which_set))
    converter.visualize_segmentation(save_path='openimage_coco_format_{}_segm.json'.format(args.which_set))

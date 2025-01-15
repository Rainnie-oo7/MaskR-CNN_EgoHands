import os.path as osp
import os
import glob
from types import NoneType
import torch
import cv2
import scipy
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import Empty
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops import masks_to_boxes
import torchvision
from triton.language import tensor
import json
from torchvision.utils import draw_bounding_boxes

def extract_poly(file_path):
    mat_data = scipy.io.loadmat(file_path)
    if 'polygons' not in mat_data:
        print(f"Fehler: 'polygons' nicht in {file_path}")
        return []
    data = mat_data['polygons']
    polys = [[ml, mr, yl, yr] for ml, mr, yl, yr in
             zip(data['myleft'][0], data['myright'][0], data['yourleft'][0], data['yourright'][0])]
    # polys = [np.array(polys, dtype=object)]   # Trick, um ndarray mit gemischt. Dimension zu umgehen (NumPy unterstü. keine gem. Dim. direkt!)
    # print(polys)    # representative Darstellung einer matlab-File

    return polys if polys else []

def polygons_to_binary_mask(polygon, width, height): #Maske könnte None sein

    # blk_img = Image.new("L", (width, height), 0)
    blk_img = np.zeros((720, 1280), dtype= np.uint8) #handeln wenn mask leer ist

    # polygon_list = polygon.flatten().tolist()
    polygon_list = polygon.reshape(-1,1,2)

    if len(polygon_list) == 0:
        return blk_img
    pts = np.int32(polygon_list)
    cv2.fillPoly(blk_img, [pts], [255])
    # ImageDraw.Draw(blk_img).polygon(polygon_list, outline=1, fill=1)
    # mask = np.array(blk_img)
    # return mask
    return blk_img

def beatboxing(mask):   #Weil die maske none sein könnte, hier eine Überprüfuing einbauen
    x_values = []
    y_values = []
    for item in mask:
        if len(item) == 2:
            x, y = item
            x_values.append(x)
            y_values.append(y)
        else:
            x_values.append(np.array([]))
            y_values.append(np.array([]))

    if any(i > 0 for i in x_values) and any(i > 0 for i in y_values):
        min_x = min(i for i in x_values if i > 0)
        max_x = max(i for i in x_values if i > 0)
        min_y = min(i for i in y_values if i > 0)
        max_y = max(i for i in y_values if i > 0)

        tupel1 = (min_x, min_y)
        tupel2 = (max_x, max_y)
        bbmatrix = np.array([tupel1, tupel2])
    # print("das ist bbmatrix:", bbmatrix)
        bbmatrix = bbmatrix.flatten()
        return bbmatrix
    else: return []

def mkboxes_n_relate(bbmatrixmyleft, bbmatrixmyright, bbmatrixyourleft, bbmatrixyourright):    #Etwas lang geworden #Wenn es ausschließlich None Type ist Weiterleitung an empty_tensor
    bool_portray = [True, True, True, True] # Specifies related Existance for [N]_Labels = [N]_Boxes later on requrement! #Wenn es nicht leer ist, aber ein Nonetype ist, Weiterleitung an empty_tensor
    empty_tensor = torch.tensor([], dtype = torch.float64)  # Tensor shape (4,) und mit 0 gefüllt
    if len(bbmatrixmyleft) != 0 and type(bbmatrixmyleft) != NoneType:
        tensor_boxesml = tv_tensors.BoundingBoxes(bbmatrixmyleft, format="XYXY", canvas_size=[720, 1280])
        bool_portray[0] = True
    else:
        tensor_boxesml = empty_tensor
        bool_portray[0] = False
    if len(bbmatrixmyright) != 0 and type(bbmatrixmyright) != NoneType:
        tensor_boxesmr = tv_tensors.BoundingBoxes(bbmatrixmyright, format="XYXY", canvas_size=[720, 1280])
        bool_portray[1] = True
    else:
        tensor_boxesmr = empty_tensor
        bool_portray[1] = False
    if len(bbmatrixyourleft) != 0 and type(bbmatrixyourleft) != NoneType:
        tensor_boxesyl = tv_tensors.BoundingBoxes(bbmatrixyourleft, format="XYXY", canvas_size=[720, 1280])
        bool_portray[2] = True
    else:
        tensor_boxesyl = empty_tensor
        bool_portray[2] = False
    if len(bbmatrixyourright) != 0 and type(bbmatrixyourright) != NoneType:
        tensor_boxesyr = tv_tensors.BoundingBoxes(bbmatrixyourright, format="XYXY", canvas_size=[720, 1280])
        bool_portray[3] = True
    else:
        tensor_boxesyr = empty_tensor
        bool_portray[3] = False

    return tensor_boxesml, tensor_boxesmr, tensor_boxesyl, tensor_boxesyr, bool_portray

def mktensorboxes(tensor_boxesml, tensor_boxesmr, tensor_boxesyl, tensor_boxesyr):
    fin_tensors_list = []  # Gebe nur die mit Werten befüllten\gültigen B-Boxes. Bsp: Es gibt 3 BB, gebe 3 BB aus, anstatt 4 und einer ist kaputt.
    count = 0

    for tensor in [tensor_boxesml, tensor_boxesmr, tensor_boxesyl, tensor_boxesyr]:  # Überprüfung Nicht-Leerheit jedes Tensors
        if tensor.size(0) != 0 and type(tensor) != NoneType:  # Wenn die erste Dimension nicht Länge 0 hat
            fin_tensors_list.append(tensor.squeeze(0))
            count = + 1
    for i in fin_tensors_list:
        if type(i) != torch.Tensor:
            print(f"Ja Keine gültigen Tensoren guck count {count} tensor in fin_tensors_liste ist kein Tensor.")
        if i.size(0) != 0:
            tensor_boxes = torch.stack(fin_tensors_list)
            if type(tensor_boxes) != NoneType:  # Hier muss Image gelöscht werden # Ganze Zeile in .mat soll übersprungen w\!
                return tensor_boxes
        else:
            print(f"Ja Keine gültigen Tensoren guck count {count} in mktensorboxes zum Stacken.")

def mktensorlabels(bool_portray):
    # tensor_labels = torch.tensor([1, 2, 3, 4])  # ^angepasst
    tmp = []
    for i, b in enumerate(bool_portray):
        if b == True:
            tmp.append(i+1)
    tensor_labels = torch.tensor(tmp)

    return tensor_labels

def mkarea(tensor_boxes):   #Nach #1130 #80 #480
    empty_tensor = torch.tensor([], dtype = torch.float64)  # Tensor shape (4,) und mit 0 gefüllt
    if type(tensor_boxes) == NoneType:
        print("type tensor_boxes ist NoneType!!!")
        return empty_tensor

    if tensor_boxes.size(0) == 0 and type(tensor_boxes) != NoneType:
        print("tensor_boxes ist leer. Der type tensor_boxes ist o.k.")
        return empty_tensor
    else:
        for i in tensor_boxes:
            if i.size(0) != 0 and type(i) != NoneType:
                # Es berechnet, weil es sich zm einen Tensor handelt, alle Rows bzw. : skalarhaft gleichzeitig...No need for append over list.
                area = (tensor_boxes[:, 3] - tensor_boxes[:, 1]) * (tensor_boxes[:, 2] - tensor_boxes[:, 0])
                return area

# def ensure_four_masks(my_left_mask, my_right_mask, your_left_mask, your_right_mask, height= 720, width= 1280):
#     # Bitweise IDs für die Masken definieren, sonst werden in der all-masks = ml + mr + yl + yr Addition alle Einsen zusammenaddiert, das führt zu 510, 765 etc
#     my_left_id = 1 << 0  # 0001 = 1
#     my_right_id = 1 << 1  # 0010 = 2
#     your_left_id = 1 << 2  # 0100 = 4
#     your_right_id = 1 << 3  # 1000 = 8
#
#     # Leere Maske als Platzhalter
#     empty_mask = torch.zeros((height, width), dtype=torch.int32)
#
#     # Überprüfen und fehlende Masken ersetzen
#     masks = [
#         my_left_mask *    my_left_id    if my_left_mask is not None else empty_mask * my_left_id,
#         my_right_mask *   my_right_id   if my_right_mask is not None else empty_mask * my_right_id,
#         your_left_mask *  your_left_id  if your_left_mask is not None else empty_mask * your_left_id,
#         your_right_mask * your_right_id if your_right_mask is not None else empty_mask * your_right_id,
#     ]
#
#     # Stapeln der Masken
#     # stacked_masks = torch.stack(masks)  # Shape: (4, height, width)   # gibt 2??, 510, 1020, 2040 aus wegen bit
#     all_masks = masks[0] | masks[1] | masks[2] | masks[3]
#
#     return all_masks

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, _, transforms=None):
        self.imgs = []
        self.polys = []
        self.transforms = transforms
        self.path = osp.normpath(osp.join(osp.dirname(__file__), "data"))
        self.invalid_indices = []

        for dir in os.listdir(self.path):
            full_dir_path = os.path.join(self.path, dir)
            if os.path.isdir(full_dir_path):
                files = sorted(os.listdir(full_dir_path))
                for file in files:
                    if file.startswith('frame'):
                        img_path = os.path.join(full_dir_path, file)
                        self.imgs.append(img_path)  # Append image to the list
                    elif file.startswith('polygons'):
                        file_path = os.path.join(full_dir_path, file)
                        # self.polys  = self.polys  + extract_poly(file_path)
                        polys = extract_poly(file_path)

                        # Daten validieren, bevor sie gespeichert werden
                        if polys and all(isinstance(p, list) and len(p) == 4 for p in polys):
                            self.polys += polys
                        else:
                            print(f"Ungültige Polygon-Daten in {file_path} übersprungen.")

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.imgs[idx])
            img = tv_tensors.Image(img)
            img = img.permute(2, 0, 1)


            my_left, my_right, your_left, your_right = self.polys[:][idx]
            # print(self.polys[:][2697]) #[array([], shape=(0, 0), dtype=uint8), array([], shape=(0, 0), dtype=uint8), array([], shape=(0, 0), dtype=uint8), array([], shape=(0, 0), dtype=uint8)]

            bbmatrixmyleft = np.array(beatboxing(my_left))
            bbmatrixmyright   = np.array(beatboxing(my_right))
            bbmatrixyourleft   = np.array(beatboxing(your_left))
            bbmatrixyourright = np.array(beatboxing(your_right))
            tensor_boxesml, tensor_boxesmr, tensor_boxesyl, tensor_boxesyr, bool_portray = mkboxes_n_relate(bbmatrixmyleft, bbmatrixmyright, bbmatrixyourleft, bbmatrixyourright)

            tensor_boxes = mktensorboxes(tensor_boxesml, tensor_boxesmr, tensor_boxesyl, tensor_boxesyr)
            ##


            # Masken mit IDs multiplizieren
            my_left_mask    = torch.tensor(polygons_to_binary_mask(my_left, 1280, 720), dtype=torch.int32)
            my_right_mask   = torch.tensor(polygons_to_binary_mask(my_right, 1280, 720), dtype=torch.int32)
            your_left_mask  = torch.tensor(polygons_to_binary_mask(your_left, 1280, 720), dtype=torch.int32)
            your_right_mask = torch.tensor(polygons_to_binary_mask(your_right, 1280, 720), dtype=torch.int32)
            # Bitweise IDs für die Masken definieren, sonst werden in der all-masks = ml + mr + yl + yr Addition alle Einsen zusammenaddiert, das führt zu 510, 765 etc
            my_left_id = 1 << 0  # 0001 = 1
            my_right_id = 1 << 1  # 0010 = 2
            your_left_id = 1 << 2  # 0100 = 4
            your_right_id = 1 << 3  # 1000 = 8

            my_left_mask = my_left_mask * my_left_id
            my_right_mask = my_right_mask * my_right_id
            your_left_mask = your_left_mask * your_left_id
            your_right_mask = your_right_mask * your_right_id

            # Bitweises OR        print(all_masks)
            all_masks = my_left_mask | my_right_mask | your_left_mask | your_right_mask

            # all_masks = ensure_four_masks(my_left_mask, my_right_mask, your_left_mask, your_right_mask, height= 720, width= 1280)
            # print(all_masks)
            # Number of Bounding Boxes
            obj_ids = torch.unique(all_masks)
            # first id is the missly inwith-in calculated background, so remove it  # obj_ids is tensor([0, 255, 510, 1020, 2040], dtype=torch.int32)
            valid_mask_ids = obj_ids[1:]
            masks = (all_masks == valid_mask_ids[:, None, None]).to(dtype=torch.uint8)
            # # masks = tv_tensors.Mask(masks)
            # masks = torch.tensor(masks)
            # print(masks)
            ##
            # Dynamische Labels zuweisen
            valid_labels = torch.arange(1, len(valid_mask_ids) + 1, dtype=torch.int64)

            # Mappen der Masken-IDs auf Labels
            mask_to_label = {mask_id.item(): label.item() for mask_id, label in zip(valid_mask_ids, valid_labels)}

            # Labels für jede Maske extrahieren
            tensor_labels = torch.tensor([mask_to_label[mask_id.item()] for mask_id in obj_ids if mask_id != 0])
            # tensor_labels = torch.as_tensor([1, 2, 3, 4], dtype=torch.int64)
            ##

            image_id = idx
            ##

            tensor_area = mkarea(tensor_boxes)
            ##

            iscrowd = torch.zeros((4,), dtype=torch.int64)
            tensor_iscrowd = iscrowd
            ##

            target =  {'boxes': tensor_boxes,  # Tensor der Form [N, 4] # Nr. 3
                'labels': tensor_labels,  # Tensor der Form [N]     #
                'masks': masks,  # Tensor der Form [N, H, W]
                'image_id': image_id,  # Tensor der Form [N]
                'area': tensor_area,  # Tensor der Form [N]
                'iscrowd': tensor_iscrowd}  # Tensor der Form [N]

            if self.transforms is not None:
                img, target = self.transforms(img, target)
            # print(target)
            return img, target
        except Exception as e:
            print(f"Fehler bei Index {idx}: {e}")
            self.invalid_indices.append({
                "index": idx,
                # "error": str(e),
            })  # Speichere den fehlerhaften Index
            return None, None

    def __len__(self):
        return len(self.imgs)




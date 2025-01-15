import torch
import torchvision
from networkx.algorithms.bipartite.basic import color
from torchvision.io import read_image
import cv2
from Mydatasetplayersjede import Mydataset
import numpy as np
from engine import train_one_epoch, evaluate
import utils
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms.v2 as T
from torchvision.transforms import functional as F
from get_model_instance_segmentation import *   #hier is: from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torch.utils.data import Subset

# class CustomSubset(Subset):
#     def __init__(self, dataset, indices):
#         super().__init__(dataset, indices)
#         self.invalid_indices = getattr(dataset, 'invalid_indices', [])



def get_transform(train):
    transforms = []
    if train:
        pass
        # transforms.append(T.RandomHorizontalFlip(0.5)),
        # transforms.append(T.RandomRotation((-30, +30)))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)
###################APPROACH 1#########################start from a model pre-trained on COCO Common Objects in Context
# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 5  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#Approach 2 eh:

"""
#Approach 2 wechselt das fasterrcn durch mobilenet aus!!!
###################APPROACH 2######################### - Modifying the model to add a different backbone
#
# # load a pre-trained model for classification and return
# # only the features
# backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
# # ``FasterRCNN`` needs to know the number of
# # output channels in a backbone. For mobilenet_v2, it's 1280
# # so we need to add it here
# backbone.out_channels = 1280
#
# # let's make the RPN generate 5 x 3 anchors per spatial
# # location, with 5 different sizes and 3 different aspect
# # ratios. We have a Tuple[Tuple[int]] because each feature
# # map could potentially have different sizes and
# # aspect ratios
# anchor_generator = AnchorGenerator(
#     sizes=((32, 64, 128, 256, 512),),
#     aspect_ratios=((0.5, 1.0, 2.0),)
# )
#
# # let's define what are the feature maps that we will
# # use to perform the region of interest cropping, as well as
# # the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be [0]. More generally, the backbone should return an
# # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
# # feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(
#     featmap_names=['0'],
#     output_size=7,
#     sampling_ratio=2
# )
#
# # put the pieces together inside a Faster-RCNN model
# model = FasterRCNN(
#     backbone,
#     num_classes=2,
#     rpn_anchor_generator=anchor_generator,
#     box_roi_pool=roi_pooler
# )
"""

#########################################

#hol TEST:

"""
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
dataset = Mydataset(torch.utils.data.Dataset, get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

# For Training
images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)  # Returns losses and detections
print("nikolaus")
print(output)

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)  # Returns predictions
print("drei mal schwarzer kater")
print(predictions[0])
"""
"""
#Nachdem Testing:

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 5
# use our dataset and defined transformations
dataset = Mydataset(torch.utils.data.Dataset, get_transform(train=True))
dataset_test = Mydataset(torch.utils.data.Dataset, get_transform(train=False))


# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:1850])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-250:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)


# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 4

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# Traceback (most recent call last):
#   File "/home/User/PycharmProjects/Project/main.py", line 184, in <module>
#     evaluate(model, data_loader_test, device=device)
#   File "/home/User/miniconda3/envs/tutorial/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
#     return func(*args, **kwargs)
#   File "/home/User/PycharmProjects/Project/engine.py", line 97, in evaluate
#     coco = get_coco_api_from_dataset(data_loader.dataset)
#   File "/home/User/PycharmProjects/Project/coco_utils.py", line 183, in get_coco_api_from_dataset
#     return convert_to_coco_api(dataset)
#   File "/home/User/PycharmProjects/Project/coco_utils.py", line 162, in convert_to_coco_api
#     ann["segmentation"] = coco_mask.encode(masks[i].numpy())
# IndexError
# IndexError: index 3 is out of bounds for dimension 0 with size 3
# 
# Traceback (most recent call last):
#     evaluate(model, data_loader_test, device=device)
#   File "/home/User/miniconda3/envs/tutorial/lib/python3.10/site-packages/torch/utils/_contextlib.py", in decorate_context
#     return func(*args, **kwargs)
#   File "/home/User/PycharmProjects/Project/engine.py", in evaluate
#     coco = get_coco_api_from_dataset(data_loader.dataset)
#   File "/home/User/PycharmProjects/Project/coco_utils.py", in get_coco_api_from_dataset
#     return convert_to_coco_api(dataset)
#   File "/home/User/PycharmProjects/Project/coco_utils.py", in convert_to_coco_api
#     ann["segmentation"] = coco_mask.encode(masks[i].numpy())
# IndexError


print("That's it!")
torch.save(model, "model.pth")
torch.save(model.state_dict(), "model_weights.pth")
"""

##

"""
Abgeschlossen, funktioniert (Objekt Detektion) Kleinere Areas 32² (<1024 px²)haben eine bessere Detektion erbracht 0.85 als größere Areas >96² (>9216²)
"""

# Anwendung unbekanntesBild


# # Speichern
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }, "checkpoint.pth")
#

# # Laden
# checkpoint = torch.load("checkpoint.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

image = read_image("data/CHESS_COURTYARD_B_T/frame_1035.jpg")
eval_transform = get_transform(train=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load("model.pth")
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cuda')))
model.eval()  # Wechsel in den Evaluierungsmodus, falls nötig

"""
with torch.no_grad():
    x = eval_transform(image)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x, ])
    for pred in predictions:
        print("Boxen:", pred['boxes'])
        print("Masken:", pred['masks'].shape)  # Shape: (N, 720, 1280)
        print("Labels:", pred['labels'])
    pred = predictions[0]


image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]
label_colors = {
    1: "red",
    2: "blue",
    3: "yellow",
    4: "green",
}
pred_labels = pred["labels"].tolist()  # Konvertiere Tensor zu Liste
pred_boxes = pred["boxes"].long()
box_colors = [label_colors[label] for label in pred_labels]
# pred_labels = [f"{label}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]

output_image = draw_bounding_boxes(image, boxes=pred_boxes, labels=[f"Label {l}" for l in pred_labels], colors=box_colors)

masks = (pred["masks"] > 0.05).squeeze(1)
output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
plt.show()


"""
def process_frame(frame, model, device):
    # label_colors = {
    #     1: (0, 0, 255), #ROT
    #     2: (0, 255, 255), #gelb
    #     3: (0, 255, 0), #green
    #     4: (255, 0, 0)  #blau
    # }

    # Convert the frame to a tensor
    frame_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(frame_tensor)

    # Extract predictions
    predictions = outputs[0]
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    masks = predictions['masks'].cpu().numpy()
    # box_colors = [label_colors[label] for label in labels]

    # Filter results based on a confidence threshold
    confidence_threshold = 0.5
    valid_indices = scores > confidence_threshold
    boxes = boxes[valid_indices]
    labels = labels[valid_indices]
    masks = masks[valid_indices]
    # Draw predictions on the frame
    for box, mask in zip(boxes, masks):
        x1, y1, x2, y2 = map(int, box)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Apply the mask with transparency
        mask = mask[0] > 0.5  # Convert to binary mask
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        # for i in mask:
        colored_mask[mask] = (0, 255, 0)
            # colored_mask[mask[0] if i == '1' in range(len(labels)) else mask[1]] = (0, 0, 255)
            # colored_mask[mask[1] if i == '2' in range(len(labels)) else mask[2]] = (0, 255, 0)
            # colored_mask[mask[2] if i == '3' in range(len(labels)) else mask[3]] = (255, 0, 0)
            # colored_mask[mask[3] if i == '4' in range(len(labels)) else mask[0]] = (255, 255, 0)
        frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    return frame


# Load the video
video_path = 'video2.webm'
output_path = 'output_video2.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process and write the frame
    processed_frame = process_frame(frame, model, device)
    # out.write(processed_frame)

# Release resources
cap.release()
# out.release()
# cv2.destroyAllWindows()

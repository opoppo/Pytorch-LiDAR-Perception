import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop
from flashtorch.activmax import GradientAscent
import matplotlib.pyplot as plt
import torch
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
import os
from tqdm import tqdm

cfg.merge_from_file('./maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_FPN_1x.yaml')
model=build_detection_model(cfg)    # loaded from the checkpoint model
model=model.backbone

path='/home/wangfa/Workspace/jupiter/maskrcnn-benchmark/datasets/coco/train2014/'

for root,dirs,files in os.walk(path):
    for file in tqdm(files):
        backprop = Backprop(model)
        image = load_image(os.path.join(root,file))
        input_ = apply_transforms(image)

        target_class = 0
        backprop.visualize(input_,target_class,guided=True)


        # model = models.vgg16(pretrained=True)
        # g_ascent= GradientAscent(model.features)
        # conv5_1=model.features[24]
        # conv5_1_filters= [45,271,363,409]
        #
        # g_ascent.visualize(conv5_1,conv5_1_filters,title='vgg16: conv5_1')
        # plt.savefig('/home/wangfa/Desktop/output/'+file)
        plt.show()

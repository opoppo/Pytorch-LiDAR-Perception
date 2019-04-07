import torch
import cv2
from maskrcnn_benchmark.modeling.detector.generalized_rcnn import GeneralizedRCNN
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg

# the_model = GeneralizedRCNN(cfg)
# the_model.load_state_dict(torch.load('./model_0068420.pth'))

the_model = torch.load('./toonnx.pth')
the_model.eval().cuda()
pp = cv2.imread('./0.jpg')
pp = torch.FloatTensor(pp)
pp = pp[None, :, :, :].permute(0, 3, 1, 2).cuda()

torch.onnx.export(the_model, pp, "alexnet.onnx", verbose=True)

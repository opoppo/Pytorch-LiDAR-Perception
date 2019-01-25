# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize

import cv2
import shutil
from maskrcnn_benchmark.engine.bBox_2D import bBox_2D


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        # print(targets,'============================')
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            # print(output.size(),'=========oo')
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        # print(images.tensors.size(),'=====================================')
        x = images.tensors.permute(0,2, 3, 1)
        images = x.cpu().detach().numpy().copy()
        # print(emptyImage.shape,type(emptyImage))
        # emptyImage = cv2.resize(emptyImage, (200, 200), interpolation=cv2.INTER_CUBIC)

        del x
        for j,(im,tar,out) in enumerate( zip(images,targets,output)):
            overlay_boxes(im,out,'output')
            overlay_boxes(im,tar,'targets')
            # print(out.shape)

            # cv2.imshow('scan', im)
            # print(im.shape,'======')
            # torch.save(im,'%d'%i)
            # shutil.rmtree('./result')
            # os.mkdir('./result')
            cv2.imwrite('./result/%d.jpg' % i, im)
            # k=cv2.waitKey()
            # if k == 27:  # Esc for exiting
            #     cv2.destroyAllWindows()
            #     os._exit(1)

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def overlay_boxes(image, predictions,anntype):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    oriens = predictions.get_field("rotations")
    boxes = predictions.bbox

    # print('\noriens:',oriens.size(),'boxes:',boxes.size(),'==========\n')

    for box, orien in zip(boxes, oriens):
        box = box.squeeze_().detach().cpu().numpy()
        alpha=torch.atan2(orien[:][0],orien[:][1])*180/3.1415926
        alpha = alpha.squeeze_().detach().cpu().numpy()
        # print(alpha,anntype,'====')
        # top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        top_left, bottom_right = box[:2], box[2:]
        l = top_left[1] - bottom_right[1]
        w = top_left[0] - bottom_right[0]
        xc = (top_left[0] + bottom_right[0]) / 2
        yc = (top_left[1] + bottom_right[1]) / 2

        box = bBox_2D(l, w, xc, yc, alpha)
        box.bBoxCalcVertxex()
        color={'targets':(155,255,255),'output':(155,255,55)}
        cv2.line(image, box.vertex1, box.vertex2, color[anntype], 2, cv2.LINE_AA)
        cv2.line(image, box.vertex2, box.vertex4, color[anntype], 2, cv2.LINE_AA)
        cv2.line(image, box.vertex3, box.vertex1, color[anntype], 2, cv2.LINE_AA)
        cv2.line(image, box.vertex4, box.vertex3, color[anntype], 2, cv2.LINE_AA)

    return image



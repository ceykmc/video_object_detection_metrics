# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from utility import iou


def compute_tp_and_fp_on_single_image(predicts, ground_truths, iou_threshold=0.5):
    """
    :param predicts: numpy array,
        two dimension numpy array, each row has format [score, x1, y1, x2, y2], shape is m x 5
        where m corresponds to predict rectangle number
    :param ground_truths: numpy array,
        two dimension numpy array, each row has format [x1, y1, x2, y2], shape is n x 4
        where n corresponds to ground truth rectangle number
    :param iou_threshold:
    :return:  True Positive, False Positive and corresponding scores
    """
    # 根据confidence从大到小依次匹配
    predict_scores = predicts[:, 0]
    predict_boxes = predicts[:, 1:]
    predict_boxes = predict_boxes[np.argsort(predict_scores)[::-1]]
    predict_scores = predict_scores[np.argsort(predict_scores)[::-1]]

    m, n = predicts.shape[0], ground_truths.shape[0]
    ground_truth_matched_flag = np.zeros(shape=(m, ), dtype=np.int32)  # record if the ground truth has been matched
    true_positive = np.zeros(shape=(n, ), dtype=np.int32)
    false_positive = np.zeros(shape=(n, ), dtype=np.int32)

    for i in range(m):
        predict_box = predict_boxes[i]
        overlaps = np.array([iou(predict_box, ground_truth) for ground_truth in ground_truths])
        if len(overlaps) == 0:
            false_positive[i] = 1
            continue
        max_overlap = np.max(overlaps)
        max_overlap_index = np.argmax(overlaps).item()
        if max_overlap > iou_threshold and ground_truth_matched_flag[max_overlap_index] is 0:
            ground_truth_matched_flag[max_overlap_index] = 1
            true_positive[i] = 1
        else:
            false_positive[i] = 1
    return true_positive, false_positive, predict_scores


def compute_tp_and_fp(all_predicts, all_ground_truths, iou_threshold=0.5):
    """
    :param all_predicts: list, predict result on all test images,
        for predict result on i'th test image, it has format:
            [[score, x1, y1, x2, y2], [score, x1, y1, x2, y2], ... , [score, x1, y1, x2, y2]], total p_i elements
        where p_i corresponds to predict rectangle number on the i'th test image
        all_predicts has n elements, where n corresponds to the number of test images
    :param all_ground_truths: list, ground truth on all test images,
        for ground truth on i'th test image, it has format:
            [[x1, y1, x2, y2], [x1, y1, x2, y2], ... , [x1, y1, x2, y2]], total g_i elements
        where g_i corresponds to ground truth number on the i'th test image
        all_ground_truths has n elements, where n corresponds to the number of test images
    :param iou_threshold:
    :return: True Positive, False Positive and scores
    """
    assert len(all_predicts) == len(all_ground_truths)

    all_tp, all_fp, all_scores = list(), list(), list()
    for i in range(len(all_predicts)):
        predicts = np.array(all_predicts[i])
        ground_truths = np.array(all_ground_truths[i])
        tp, fp, scores = compute_tp_and_fp_on_single_image(
            predicts, ground_truths, iou_threshold)
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_scores.extend(scores)
    all_scores = np.array(all_scores, dtype=np.float32)
    sorted_indices = np.argsort(all_scores)[::-1]
    all_tp = np.array(all_tp, dtype=np.int32)[sorted_indices]
    all_fp = np.array(all_fp, dtype=np.int32)[sorted_indices]
    all_scores = all_scores[sorted_indices]

    return all_tp, all_fp, all_scores


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_average_precision(all_predicts, all_ground_truths, iou_threshold=0.5, plot_ap=False):
    """
    :param all_predicts: list, predict result on all test images,
        for predict result on i'th test image, it has format:
            [[score, x1, y1, x2, y2], [score, x1, y1, x2, y2], ... , [score, x1, y1, x2, y2]], total p_i elements
        where p_i corresponds to predict rectangle number on the i'th test image
        all_predicts has n elements, where n corresponds to the number of test images
    :param all_ground_truths: list, ground truth on all test images,
        for ground truth on i'th test image, it has format:
            [[x1, y1, x2, y2], [x1, y1, x2, y2], ... , [x1, y1, x2, y2]], total g_i elements
        where g_i corresponds to ground truth number on the i'th test image
        all_ground_truths has n elements, where n corresponds to the number of test images
    :param iou_threshold:
    :param plot_ap: draw average precision curves
    :return: True Positive, False Positive and scores
    """
    tp, fp, scores = compute_tp_and_fp(all_predicts, all_ground_truths, iou_threshold)

    positive_number = sum([len(e) for e in all_ground_truths])
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(positive_number)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    ap = voc_ap(recall, precision, False)

    if plot_ap:
        precision, recall, _ = precision_recall_curve(1 - fp, scores)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(ap))
        plt.show()

    return ap


def main():
    pass


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

"""
In fragment error, we evaluate the integrity of the detections along a trajectory.
Particularly, the results of a stable detector should be consistent(always report
    as a target or always not).
It should not frequently change its status throughout the trajectory.

Paper: On The Stability of Video Detection and Tracking.
       Section 3.2.1
"""

import numpy as np

from utility import iou


def single_frame_match(predicts, ground_truths, predict_score_threshold, iou_threshold):
    """
    :param predicts: numpy array,
        two dimension numpy array, each row has format [score, x1, y1, x2, y2], shape is m x 5
        where m corresponds to predict rectangle number
    :param ground_truths: numpy array,
        two dimension numpy array, each row has format [x1, y1, x2, y2], shape is n x 4
        where n corresponds to ground truth rectangle number
    :param predict_score_threshold:
        any predict whose score is less than \predict_score_threshold will be ignored
    :param iou_threshold:
    :return: return a list, whose length is equal to \ground_truths,
        each element in the list represents if the corresponding ground_truth is matched,
        value 0 represent for unmatched, value 1 represent for matched.
    """
    predict_scores = predicts[:, 0]
    predict_boxes = predicts[:, 1:]
    predict_boxes = predict_boxes[np.where(predict_scores >= predict_score_threshold)]
    m, n = predicts.shape[0], ground_truths.shape[0]

    if m == 0:  # there are no predict boxes to be matched
        return [0] * n

    predict_box_used_flag = [0] * m
    ground_truth_matched_flag = [0] * n
    for i in range(n):
        ground_truth = ground_truths[i]
        overlaps = np.array([iou(predict_box, ground_truth) for predict_box in predict_boxes])
        max_overlap = np.max(overlaps)
        max_overlap_index = np.argmax(overlaps).item()
        if max_overlap > iou_threshold and predict_box_used_flag[max_overlap_index] is 0:
            predict_box_used_flag[max_overlap_index] = 1
            ground_truth_matched_flag[i] = 1
    return ground_truth_matched_flag


def main():
    pass


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import math
import statistics
from collections import defaultdict
import numpy as np

from utility import iou


def single_frame_match(predicts, ground_truths, predict_score_threshold, iou_threshold):
    """
    :param predicts: numpy array,
        two dimension numpy array, each row has format [score, x1, y1, x2, y2], shape is m x 5
        where m corresponds to predict rectangle number
    :param ground_truths: numpy array,
        two dimension numpy array, each row has format [person_id, x1, y1, x2, y2], shape is n x 4
        where n corresponds to ground truth rectangle number
    :param predict_score_threshold:
        any predict whose score is less than \predict_score_threshold will be ignored
    :param iou_threshold:
    :return: return a list, whose length is equal to \ground_truths,
        each element in the list is the matched predict position(exclude score)
        if there is not matched predict for some ground truths, the corresponding element is a empty list.
        for example, something like:
            [[x1, y1, x2, y2], [], [x1, y1, x2, y2]]
        where the second element is an empty list, meaning the second ground truth has no matched predict
    """
    m, n = predicts.shape[0], ground_truths.shape[0]
    assert n > 0, "there must be at least one ground truth"

    if m == 0:  # there are no predict boxes to be matched
        return [[]] * n

    predict_scores = predicts[:, 0]
    predict_boxes = predicts[:, 1:]
    predict_boxes = predict_boxes[np.where(predict_scores >= predict_score_threshold)]

    predict_box_used_flag = [0] * m
    ground_truth_matched_predict = [[]] * n
    for i in range(n):
        ground_truth = ground_truths[i][1:]  # exclude person id
        overlaps = np.array([iou(predict_box, ground_truth) for predict_box in predict_boxes])
        max_overlap = np.max(overlaps)
        max_overlap_index = np.argmax(overlaps).item()
        if max_overlap > iou_threshold and predict_box_used_flag[max_overlap_index] is 0:
            predict_box_used_flag[max_overlap_index] = 1
            ground_truth_matched_predict[i] = predict_boxes[max_overlap_index]
    return ground_truth_matched_predict


def compute_scale_and_ratio_error(matched_record):
    error_s, error_r = list(), list()
    m = len(matched_record)
    for i in range(m):
        assert len(matched_record[i]) == 2
        if len(matched_record[i][1]) == 0:  # there is no matched predict
            continue
        ground_truth = matched_record[i][0]
        predict = matched_record[i][1]
        g_c_x, g_c_y, g_w, g_h = [(ground_truth[2] - ground_truth[0]) / 2,
                                  (ground_truth[3] - ground_truth[1]) / 2,
                                  ground_truth[2] - ground_truth[0],
                                  ground_truth[3] - ground_truth[1]]
        p_c_x, p_c_y, p_w, p_h = [(predict[2] - predict[0]) / 2,
                                  (predict[3] - predict[1]) / 2,
                                  predict[2] - predict[0],
                                  predict[3] - predict[1]]
        e_s = math.sqrt((p_w * p_h) / (g_w * g_h))
        e_r = (p_w / p_h) / (g_w / g_h)
        error_s.append(e_s)
        error_r.append(e_r)
    if len(error_s) <= 1:
        return 0
    std_s = statistics.stdev(error_s)
    std_r = statistics.stdev(error_r)
    return std_s + std_r


def center_position_error(video_predicts, video_ground_truths, predict_score_threshold=0.4, iou_threshold=0.5):
    """
    :param video_predicts: list
        predict result on each frame of the video, has \m elements,
            where m equal to the frame number in the video
        each element represents predict results on corresponding frame,
            has format [[score, x1, y1, x2, y2], [score, x1, y1, x2, y2], ... , [score, x1, y1, x2, y2]]
            may have zero predict result, corresponding format is []
    :param video_ground_truths: list
        ground truth on each frame of the video, has \m elements,
            where m equal to the frame number in the video
        each element represents predict results on corresponding frame,
            has format [[person_id, x1, y1, x2, y2], [person_id, x1, y1, x2, y2], ... , [person_id, x1, y1, x2, y2]]
            may have zero predict result, corresponding format is []
    :param predict_score_threshold: float
    :param iou_threshold: float
    :return: dict
        center position error for each trajectory
    """
    assert len(video_predicts) == len(video_ground_truths)

    # step 1: get each trajectory match record
    trajectories_matched_record = defaultdict(list)
    m = len(video_predicts)
    for i in range(m):
        if len(video_ground_truths[i]) == 0:
            continue
        predicts = np.array(video_predicts[i])
        ground_truths = np.array(video_ground_truths[i])
        ground_truth_matched_predict = \
            single_frame_match(predicts, ground_truths, predict_score_threshold, iou_threshold)
        assert len(ground_truth_matched_predict) == ground_truths.shape[0]
        n = len(ground_truth_matched_predict)
        for j in range(n):
            person_id = ground_truths[j][0]
            trajectories_matched_record[person_id].append(
                (ground_truths[j][1:], ground_truth_matched_predict[j]))
    # step 2: compute each trajectory scale and ratio error
    trajectories_scale_and_ratio_error = dict()
    for person_id, match_record in trajectories_matched_record.items():
        trajectories_scale_and_ratio_error[person_id] = \
            compute_scale_and_ratio_error(match_record)
    return trajectories_scale_and_ratio_error


def main():
    pass


if __name__ == "__main__":
    main()

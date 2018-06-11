# -*- coding: utf-8 -*-

"""
In fragment error, we evaluate the integrity of the detections along a trajectory.
Particularly, the results of a stable detector should be consistent(always report
    as a target or always not).
It should not frequently change its status throughout the trajectory.

Paper: On The Stability of Video Detection and Tracking.
       Section 3.2.1
"""

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
        each element in the list represents if the corresponding ground_truth is matched,
        value 0 represent for unmatched, value 1 represent for matched.
    """
    n = ground_truths.shape[0]
    assert n > 0, "there must be at least one ground truth"

    if len(predicts) == 0:  # there are no predict boxes to be matched
        return [0] * n

    predict_scores = predicts[:, 0]
    predict_boxes = predicts[:, 1:]
    predict_boxes = predict_boxes[np.where(predict_scores >= predict_score_threshold)]

    m = predict_boxes.shape[0]
    if m == 0:  # there are no predict boxes to be matched
        return [0] * n

    predict_box_used_flag = [0] * m
    ground_truth_matched_flag = [0] * n
    for i in range(n):
        ground_truth = ground_truths[i][1:]  # exclude person id
        overlaps = np.array([iou(predict_box, ground_truth) for predict_box in predict_boxes])
        max_overlap = np.max(overlaps)
        max_overlap_index = np.argmax(overlaps).item()
        if max_overlap > iou_threshold and predict_box_used_flag[max_overlap_index] is 0:
            predict_box_used_flag[max_overlap_index] = 1
            ground_truth_matched_flag[i] = 1
    return ground_truth_matched_flag


def compute_fragment_error(matched_record):
    # As a special case, we define fragment error of a trajectory with length one to be 0.
    if len(matched_record) == 1:
        return 0
    f_k = 0
    for i in range(1, len(matched_record)):
        if matched_record[i] != matched_record[i - 1]:
            f_k += 1
    return f_k / (len(matched_record) - 1)


def fragment_error(video_predicts, video_ground_truths, predict_score_threshold=0.4, iou_threshold=0.5):
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
        fragment error for each trajectory
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
        ground_truth_matched_flag = \
            single_frame_match(predicts, ground_truths, predict_score_threshold, iou_threshold)
        assert len(ground_truth_matched_flag) == ground_truths.shape[0]
        n = len(ground_truth_matched_flag)
        for j in range(n):
            person_id = ground_truths[j][0]
            trajectories_matched_record[person_id].append(ground_truth_matched_flag[j])
    # step 2: compute each trajectory fragment error
    trajectories_fragment_error = dict()
    for person_id, match_record in trajectories_matched_record.items():
        trajectories_fragment_error[person_id] = compute_fragment_error(match_record)
    return trajectories_fragment_error


def main():
    pass


if __name__ == "__main__":
    main()

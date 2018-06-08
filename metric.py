# -*- coding: utf-8 -*-

import os

from mean_average_precision import average_precision


def read_predict_file(file_path):
    predicts = list()
    with open(file_path) as f:
        for line in f:
            predict = [float(e) for e in line.rstrip().split(' ')]
            predicts.append(predict)
    return predicts


def read_ground_truth_file(file_path):
    ground_truths = list()
    with open(file_path) as f:
        for line in f:
            ground_truth = [float(e) for e in line.rstrip().split(' ')]
            ground_truths.append(ground_truth)
    return ground_truths


def read_video_predicts_and_ground_truths(folder):
    predict_file_names = [name for name in os.listdir(folder) if '_predict.txt' in name]
    ground_truth_file_names = [name for name in os.listdir(folder) if '_ground_truth.txt' in name]

    video_predicts, video_ground_truths = list(), list()
    for predict_file_name in predict_file_names:
        predict_file_path = os.path.join(folder, predict_file_name)
        video_predicts.append(read_predict_file(predict_file_path))
    for ground_truth_file_name in ground_truth_file_names:
        ground_truth_file_path = os.path.join(folder, ground_truth_file_name)
        video_ground_truths.append(read_ground_truth_file(ground_truth_file_path))

    return video_predicts, video_ground_truths


def main():
    folder = r'E:\python_project_data\video_object_detection_metrics\test'
    video_predicts, video_ground_truths = read_video_predicts_and_ground_truths(folder)
    ap = average_precision(video_predicts, video_ground_truths)
    print('average precision: {:.2f}'.format(ap))


if __name__ == "__main__":
    main()

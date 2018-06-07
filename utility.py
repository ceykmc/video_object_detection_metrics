# -*- coding: utf-8 -*-


def iou(box_a, box_b):
    """
    :param box_a: list, box rectangle, with (x1, y1, x2, y2) format
    :param box_b: list, box rectangle, with (x1, y1, x2, y2) format
    :return: intersection of two bounding boxes over their union (IoU)
    """
    assert len(box_a) == len(box_b) and len(box_a) == 4

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 > x1 and y2 > y1:
        intersection = (x2 - x1) * (y2 - y1)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = box_a_area + box_b_area - intersection
        return intersection / union
    return 0

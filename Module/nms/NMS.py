import numpy as np
from shapely.geometry import Polygon


def calculate_area(bbox):
    # 输入numpy类型的bbox计算面积
    # 先算第一条边
    x1_len = np.abs(bbox[:, 2] - bbox[:, 0])
    y1_len = np.abs(bbox[:, 3] - bbox[:, 1])
    len1 = np.sqrt((x1_len**2) + (y1_len**2))
    # 算第二条边
    x2_len = np.abs(bbox[:, 4] - bbox[:, 2])
    y2_len = np.abs(bbox[:, 5] - bbox[:, 3])
    len2 = np.sqrt((x2_len**2) + (y2_len**2))
    areas = len1 * len2
    # areas = np.around(areas)
    return areas


def test_cal(bbox):
    res = []
    for elm in bbox:
        elm = elm.reshape((4, 2))
        area = Polygon(elm).intersection(Polygon(elm)).area
        res.append(area)
    return res


def intersection(g, p):
    g = np.asarray(g)
    p = np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def nms(pred_box, thresh):
    # 输入一张图片预测出的检测框，维度为：(N, 9),最后一个为预测的score
    boxs = pred_box.copy()
    scores = pred_box[:, 8]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        boxs = boxs[order]
        i = order[0]
        keep.append(i)

        ious = []
        max_box = boxs[0]
        for box in boxs[1:]:
            iou = calculate_area(max_box, box)
            ious.append(iou)
        ious = np.array(ious)
        index = np.where(ious <= thresh)[0]



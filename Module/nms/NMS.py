import numpy as np
from shapely.geometry import Polygon


# def calculate_area(bbox):
#     # 输入numpy类型的bbox计算面积
#     # 先算第一条边
#     x1_len = np.abs(bbox[:, 2] - bbox[:, 0])
#     y1_len = np.abs(bbox[:, 3] - bbox[:, 1])
#     len1 = np.sqrt((x1_len**2) + (y1_len**2))
#     # 算第二条边
#     x2_len = np.abs(bbox[:, 4] - bbox[:, 2])
#     y2_len = np.abs(bbox[:, 5] - bbox[:, 3])
#     len2 = np.sqrt((x2_len**2) + (y2_len**2))
#     areas = len1 * len2
#     # areas = np.around(areas)
#     return areas
#
#
# def test_cal(bbox):
#     res = []
#     for elm in bbox:
#         elm = elm.reshape((4, 2))
#         area = Polygon(elm).intersection(Polygon(elm)).area
#         res.append(area)
#     return res


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
    # boxs = pred_box.copy()
    scores = pred_box[:, 8]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        boxs = pred_box[:, :8].copy()
        boxs = boxs[order]
        i = order[0]
        keep.append(i)

        ious = []
        max_box = boxs[0]
        for box in boxs[1:]:
            iou = intersection(max_box, box)
            ious.append(iou)
        ious = np.array(ious)
        index = np.where(ious <= thresh)[0]
        order = order[index+1]

    return keep


def cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1)*(y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size>0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter/(areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def non_max_suppression(boxes, scores, threshold):
    """执行non-maximum suppression并返回保留的boxes的索引.
    boxes: [N, (y1, x1, y2, x2)].注意(y2, x2)可以会超过box的边界.
    scores: box的分数的一维数组.
    threshold: Float型. 用于过滤IoU的阈值.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # 计算box面积
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # 获取根据分数排序的boxes的索引(最高的排在对前面)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # 选择排在最前的box，并将其索引加到列表中
        i = ixs[0]
        pick.append(i)
        # 计算选择的box与剩下的box的IoU
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # 确定IoU大于阈值的boxes. 这里返回的是ix[1:]之后的索引，
        # 所以为了与ixs保持一致，将结果加1
        remove_ixs = np.where(iou > threshold)[0] + 1
        # 将选择的box和重叠的boxes的索引删除.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


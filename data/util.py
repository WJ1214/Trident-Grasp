import numpy as np
from PIL import Image
import random
import cv2


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


# 这个函数之后可能有用
def flip_bbox(bbox, size, x_flip=False, y_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        bbox[:, 1::2] = H - bbox[:, 1::2]
    if x_flip:
        # x_max = W - bbox[:, 1]
        # x_min = W - bbox[:, 3]
        # bbox[:, 1] = x_min
        # bbox[:, 3] = x_max
        bbox[:, 0::2] = W - bbox[:, 0::2]
    return bbox


def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        allow_outside_center=True, return_param=False):
    """Translate bounding boxes to fit within the cropped area of an image.

    This method is mainly used together with image cropping.
    This method translates the coordinates of bounding boxes like
    :func:`data.util.translate_bbox`. In addition,
    this function truncates the bounding boxes to fit within the cropped area.
    If a bounding box does not overlap with the cropped area,
    this bounding box will be removed.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_slice (slice): The slice of y axis.
        x_slice (slice): The slice of x axis.
        allow_outside_center (bool): If this argument is :obj:`False`,
            bounding boxes whose centers are outside of the cropped area
            are removed. The default value is :obj:`True`.
        return_param (bool): If :obj:`True`, this function returns
            indices of kept bounding boxes.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`, returns an array :obj:`bbox`.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`bbox, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **index** (*numpy.ndarray*): An array holding indices of used \
            bounding boxes.

    """

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
            .all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y_{offset}, x_{offset})`.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.

    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


# ---------------------------------------------------------------#

def random_image_flip(img, x_random=False, y_random=False, return_param=True):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            HWC format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[::-1, :, :]
    if x_flip:
        img = img[:, ::-1, :]
    img = img.copy()
    if return_param:
        return img, {'x_flip': x_flip, 'y_flip': y_flip}
    else:
        return img


def numpy_box2list(bbox):
    # 输入一张图片的相应numpy格式抓取框
    res = []
    for elm in bbox:
        box = []
        vex1 = list(elm[:2])
        vex2 = list(elm[2:4])
        vex3 = list(elm[4:6])
        vex4 = list(elm[6:8])
        box.append(vex1)
        box.append(vex2)
        box.append(vex3)
        box.append(vex4)
        res.append(box)
    return res


def tvtsf2np(img):
    # 将一张经过torchvision.transform转换过的image转换为opencv格式的numpy
    img = img.numpy() * 255
    img = img.astype('uint8')
    img = np.transpose(img, (1, 2, 0))
    return img


def show_bbox_image(image, bbox):
    if not isinstance(image, np.ndarray):
        image = tvtsf2np(image)
    if not isinstance(bbox, list):
        bbox = numpy_box2list(bbox)
    # 不copy会报错
    img = image.copy()
    for i, elm in enumerate(bbox):
        elm = np.array(elm, np.int32)
        cv2.line(img, tuple(elm[0]), tuple(elm[1]), (255, 0, 0), 2)
        cv2.line(img, tuple(elm[1]), tuple(elm[2]), (0, 255, 0), 2)
        cv2.line(img, tuple(elm[2]), tuple(elm[3]), (255, 0, 0), 2)
        cv2.line(img, tuple(elm[3]), tuple(elm[0]), (0, 255, 0), 2)
        img = cv2.putText(img, '{num}'.format(num=i), tuple(elm[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('1', img)
    cv2.waitKey(0)


def show_batch_image(image, bbox):
    for img, box in zip(image, bbox):
        img = tvtsf2np(img)
        box = box.numpy()
        box = numpy_box2list(box)
        show_bbox_image(img, box)


def calculate_x_angle(bbox):
    # 输入一个list格式的抓取框
    vex1 = bbox[0]
    vex2 = bbox[1]
    arr_a = np.array([(vex2[0] - vex1[0]), (vex2[1] - vex1[1])])  # 向量a
    axis_x = np.array([1, 0])  # 向量b
    cos_value = (float(arr_a.dot(axis_x)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(axis_x.dot(axis_x))))
    return np.arccos(cos_value) * (180 / np.pi)


def calculate_y_angle(bbox):
    # 输入一个list格式的抓取框
    vex1 = bbox[0]
    vex2 = bbox[1]
    arr_a = np.array([(vex2[0] - vex1[0]), (vex2[1] - vex1[1])])  # 向量a
    axis_y = np.array([0, 1])  # 向量b
    cos_value = (float(arr_a.dot(axis_y)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(axis_y.dot(axis_y))))
    return np.arccos(cos_value) * (180 / np.pi)


def calculate_angle(bbox):
    # 输入一个list类型的bbox,返回其应当顺时针旋转的角度
    x_angle = calculate_x_angle(bbox)
    y_angle = calculate_y_angle(bbox)
    angle_res = 0
    if x_angle > 90 and y_angle < 90:
        angle_res = x_angle - 90
    if x_angle < 90 and y_angle < 90:
        angle_res = 90 + x_angle
    if x_angle < 90 and y_angle > 90:
        angle_res = 90 - x_angle
    if x_angle > 90 and y_angle > 90:
        angle_res = 270 - x_angle
    if x_angle == 90:
        angle_res = 0
    if x_angle == 0:
        angle_res = 90
    return angle_res


def box_add0(bbox, num):
    length = bbox.shape[0]
    need = num - length
    zeros = np.zeros((need, 8))
    box = np.vstack((bbox, zeros))
    return box


def angle_add0(angles, num):
    length = angles.shape[0]
    need = num - length
    zeros = np.zeros(need)
    angle = np.hstack((angles, zeros))
    return angle


def box_remove0(bbox):
    # 输入一个numpy类型的bbox，返回numpy类型
    zeros = np.zeros(8)
    length = bbox.shape[0]
    for i, box in enumerate(bbox):
        if (box == zeros).all():
            length = i
            break
    bbox = bbox[:length]
    return bbox, length


class RandomCenterCrop(object):
    def __init__(self, x_offset=50, y_offset=50):
        self.x_offset = x_offset
        self.y_offset = y_offset

    def __call__(self, img, bbox):
        # 输入一张opencv读取的numpy类型的image
        # 输入该张image在数据集中对应的numpy类型抓取框数组
        random_x = np.random.randint(-self.x_offset, self.x_offset, 1)[0]
        random_y = np.random.randint(-self.y_offset, self.y_offset, 1)[0]
        # 使用copy()函数，防止改变数据集原本存储的self.data值
        bbox = bbox.copy()
        img = img.copy()
        # 随机裁剪图像
        img = img[80 + random_y:400 + random_y, 160 + random_x:480 + random_x, :]
        # 随机翻转图像
        img, param = random_image_flip(img, True, True, True)
        x_flip = param.get('x_flip')
        y_flip = param.get('y_flip')
        H = img.shape[0]
        W = img.shape[1]
        # 按照图像随机裁剪的数值将bbox缩放至图像内，后随机翻转
        bbox[:, 0::2] = bbox[:, 0::2] - 160 - random_x
        bbox[:, 1::2] = bbox[:, 1::2] - 80 - random_y
        bbox = flip_bbox(bbox, (H, W), x_flip, y_flip)
        need_del = []
        for i, box in enumerate(bbox):
            if (box < 0).any():
                need_del.append(i)
            if (box >= 320).any():
                need_del.append(i)
        bbox = np.delete(bbox, need_del, 0)
        zeros = np.zeros((len(need_del), 8))
        bbox = np.vstack((bbox, zeros))
        return img, bbox

# ---------------------------------------------------------------------------------#


def random_translate(img, bboxes, p=0.5):
    # 随机平移
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w_img, h_img))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return img, bboxes


def random_crop(img, bboxes, p=0.5):
    # 随机裁剪
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return img, bboxes


# 随机水平反转
def random_horizontal_flip(img, bboxes, p=0.5):
    if random.random() < p:
        _, w_img, _ = img.shape
        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
    return img, bboxes


# 随机垂直反转
def random_vertical_flip(img, bboxes, p=0.5):
    if random.random() < p:
        h_img, _, _ = img.shape
        img = img[::-1, :, :]
        bboxes[:, [1, 3]] = h_img - bboxes[:, [3, 1]]
    return img, bboxes


# 随机顺时针旋转90
def random_rot90_1(img, bboxes=None, p=0.5):
    '''
    :param img: nparray img
    :param bboxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
    :param p: 随机比例
    :return:
    '''
    # 顺时针旋转90度
    if random.random() < p:
        h, w, _ = img.shape
        trans_img = cv2.transpose(img)
        new_img = cv2.flip(trans_img, 1)
        if bboxes is None:
            return new_img
        else:
            # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
            # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
            # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
            bboxes[:, [0, 2]] = h - bboxes[:, [0, 2]]
            return new_img, bboxes
    else:
        if bboxes is None:
            return img
        else:
            return img, bboxes


# 随机逆时针旋转
def random_rot90_2(img, bboxes=None, p=0.5):
    '''
    :param img: nparray img
    :param bboxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
    :param p: 随机比例
    :return:
    '''
    # 逆时针旋转90度
    if random.random() < p:
        h, w, _ = img.shape
        trans_img = cv2.transpose(img)
        new_img = cv2.flip(trans_img, 0)
        if bboxes is None:
            return new_img
        else:
            # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
            # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
            # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
            bboxes[:, [1, 3]] = w - bboxes[:, [1, 3]]
            return new_img, bboxes
    else:
        if bboxes is None:
            return img
        else:
            return img, bboxes


# 随机对比度和亮度 (概率：0.5)
def random_bright(img, bboxes, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        mean = np.mean(img)
        img = img - mean
        img = img * random.uniform(lower, upper) + mean * random.uniform(lower, upper)  # 亮度
        img = img / 255.
    return img, bboxes


# 随机变换通道
def random_swap(im, bboxes, p=0.5):
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    if random.random() < p:
        swap = perms[random.randrange(0, len(perms))]
        print
        swap
        im[:, :, (0, 1, 2)] = im[:, :, swap]
    return im, bboxes


# 随机变换饱和度
def random_saturation(im, bboxes, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        im[:, :, 1] = im[:, :, 1] * random.uniform(lower, upper)
    return im, bboxes


# 随机变换色度(HSV空间下(-180, 180))
def random_hue(im, bboxes, p=0.5, delta=18.0):
    if random.random() < p:
        im[:, :, 0] = im[:, :, 0] + random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] = im[:, :, 0][im[:, :, 0] > 360.0] - 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] = im[:, :, 0][im[:, :, 0] < 0.0] + 360.0
    return im, bboxes


# 随机旋转0-90角度
def random_rotate_image_func(image):
    # 旋转角度范围
    angle = np.random.uniform(low=0, high=90)
    return misc.imrotate(image, angle, 'bicubic')


def random_rot(image, bboxes, angle, center=None, scale=1.0, ):
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if bboxes is None:
        for i in range(image.shape[2]):
            image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h), flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_CONSTANT)
        return image
    else:
        box_x, box_y, box_label, box_tmp = [], [], [], []
        for box in bboxes:
            box_x.append(int(box[0]))
            box_x.append(int(box[2]))
            box_y.append(int(box[1]))
            box_y.append(int(box[3]))
            box_label.append(box[4])
        box_tmp.append(box_x)
        box_tmp.append(box_y)
        bboxes = np.array(box_tmp)
        ####make it as a 3x3 RT matrix
        full_M = np.row_stack((M, np.asarray([0, 0, 1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        ###make the bboxes as 3xN matrix
        full_bboxes = np.row_stack((bboxes, np.ones(shape=(1, bboxes.shape[1]))))
        bboxes_rotated = np.dot(full_M, full_bboxes)

        bboxes_rotated = bboxes_rotated[0:2, :]
        bboxes_rotated = bboxes_rotated.astype(np.int32)

        result = []
        for i in range(len(box_label)):
            x1, y1, x2, y2 = bboxes_rotated[0][2 * i], bboxes_rotated[1][2 * i], bboxes_rotated[0][2 * i + 1], \
                             bboxes_rotated[1][2 * i + 1]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
            x1, x2 = min(w, x1), min(w, x2)
            y1, y2 = min(h, y1), min(h, y2)
            one_box = [x1, y1, x2, y2, box_label[i]]
            result.append(one_box)
        return img_rotated, result


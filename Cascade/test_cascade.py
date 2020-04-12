import argparse
import os
import time
import cv2
import numpy as np


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = cv2.resize(
            image, (round(w * scale), round(h * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def process_folder(path, out_path):
    for im in os.listdir(path):
        image = cv2.imread(os.path.join(path, im))
        start = time.time()
        image, *_ = resize_image(image, 400, 900, False)
        bbs, confs = lp.detectMultiScale2(image, minSize=(5, 5))
        print('detection time: ', time.time() - start)
        for bb, c in zip(bbs, confs):
            if c > 20:
                x, y, w, h = bb
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(out_path, im), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cascade inference')
    parser.add_argument('--in_path', default='',
                        help='path to folder with test images')
    parser.add_argument('--out_path', default='',
                        help='path to folder to save images in')
    parser.add_argument('--cascade_path', default='',
                        help='path to cascade .xml file')
    args = parser.parse_args()
    lp = cv2.CascadeClassifier()
    lp.load(args.cascade_path)
    process_folder(args.in_path, args.out_path)

import cv2 as cv
import numpy as np
from collections import deque


def normalize_image_cutoff(image, cutoff=2.5, ceil=255):  # aus Pixelmanipulationen 4.4.1
    val_min = np.percentile(image, cutoff)
    val_max = np.percentile(image, 100.0 - cutoff)  # die Max und min Werte im Bild
    val_range = val_max - val_min
    img_normalized = (image - val_min) / val_range * ceil
    # auf 0 bis 255
    img_normalized[img_normalized < 0] = 0
    img_normalized[img_normalized >= ceil] = ceil - 0.1
    # damit die Werte, die rausgeschnitten werden sollen dargestellt werden können
    return img_normalized


def corner_matrix_create(img, spacing, k, boxsize):
    gradients = np.gradient(img, spacing)
    dst = np.zeros(img.shape + (3,), 'float')
    dst[..., 0] = gradients[0] * gradients[0]
    dst[..., 1] = gradients[1] * gradients[0]
    dst[..., 2] = gradients[1] * gradients[1]

    filtered = cv.blur(dst, (boxsize, boxsize))

    a = filtered[..., 0]
    b = filtered[..., 1]
    c = filtered[..., 2]

    return a * c - b * b - k * (a + c) * (a + c)


def relaxation_algorithm(img, cs, cd):
    kernel = np.array([[1., 1., 1.],
                       [1., 0., 1.],
                       [1., 1., 1.]])
    dp = cv.filter2D(img, -1, kernel) * (cs - cd) + cd
    a = img * (1. + dp)
    b = (1. - img) * (1. - dp)
    return a / (a + b)


def feature_extract(img, values, discard):
    maxy, maxx = img.shape
    features = {}
    for i in values:
        features[i] = []

    for y in range(0, maxy, 3):
        for x in range(0, maxx, 3):
            val = img[y, x]
            if val in values:
                count, img, _, rect = cv.floodFill(img, np.array([[]]), (x, y),
                                                   discard, 0, 0)
                features[val].append([(x, y), count, rect])
    return features


def is_square(bound):
    return 0.7 * bound[3] <= bound[2] <= 1.5 * bound[3]


def area(blob):
    return blob[2][2] * blob[2][3]


def tolerance(a, b, bound1, bound2=None):
    if bound2 is None:
        bound2 = bound1
        bound1 = 1. / bound2
    return bound1 * b < a < bound2 * b


def getBBox(bounds, width, height):
    leftMost = width - 1
    topMost = height - 1
    rightMost = bottomMost = 0

    if len(bounds) == 0:
        return [0, 0, 0, 0]

    for i in bounds:
        bound = i[2]
        if leftMost > bound[0]:
            leftMost = bound[0]
        if topMost > bound[1]:
            topMost = bound[1]
        if rightMost < bound[0] + bound[2]:
            rightMost = bound[0] + bound[2]
        if bottomMost < bound[1] + bound[3]:
            bottomMost = bound[1] + bound[3]
    return [leftMost, topMost, rightMost, bottomMost]


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera!!!")
    exit()

count = 0
while True:
    ret, frame = cap.read()
    height, width, col = frame.shape

    if not ret:
        print("Can't receive frame (stream end?)")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    normalized = normalize_image_cutoff(gray, 10.0).astype(np.uint8)

    corners = corner_matrix_create(normalized, 3, 0.06, 10)
    z = np.zeros_like(gray)
    z[corners > corners.max() * 0.1] = 1
    z[corners < corners.min() * 0.1] = 2
    features2 = feature_extract(z, np.array([1, 2]), 0)

    leftMost = width - 1
    topMost = height - 1
    rightMost = bottomMost = 0

    leftMost, topMost, rightMost, bottomMost = getBBox(
            features2[2] + features2[1], width, height)
    cut = gray[topMost:bottomMost + 1, leftMost:rightMost + 1]
    normalized_strong = normalize_image_cutoff(cut, 50.0).astype(np.uint8)
    normalized_strong_copy = normalize_image_cutoff(cut, 50.0).astype(np.uint8)
    # relaxed = relaxation_algorithm(normalized / 255.0, 0.5, -0.5)
    # cv.imshow('frame2', (relaxed * 255.0).astype(np.uint8))
    # relaxed = relaxation_algorithm(relaxed, 0.5, -0.5)
    # relaxed = relaxation_algorithm(relaxed, 0.5, -0.5)
    # relaxed = relaxation_algorithm(relaxed, 0.5, -0.5)
    # # relaxed = cv.blur(relaxed, (15, 15))
    # relaxed = normalize_image_cutoff(relaxed, 50.0, 1.0)
    # cv.imshow('frame5', (relaxed * 255.0).astype(np.uint8))
    features = feature_extract(normalized_strong, np.array([0]), 255)
    features = [data for data in features[0] if is_square(data[2])]
    features.sort(
            key=lambda data: data[2][2] * data[2][3],
            reverse=True)
    resulting_squares = []

    for i in range(len(features) - 2):  # sliding window of 3
        a1 = area(features[i])
        a2 = area(features[i + 1])
        a3 = area(features[i + 2])

        if 0.8 * a2 < a1 < 1.3 * a2 and 0.8 * a3 < a1 < 1.3 * a3:
            resulting_squares.append(features[i])
            resulting_squares.append(features[i + 1])
            resulting_squares.append(features[i + 2])
            j = i + 3
            while j < len(features):
                if not tolerance(a3, area(features[j]), 0.8, 1.3):
                    break
                resulting_squares.append(features[j])
                j += 1
            break

    for i in resulting_squares:
        bound = i[2]
        frame = cv.rectangle(frame, (bound[0] + leftMost, bound[1] + topMost),
                             (bound[0] + bound[2] + leftMost, bound[1] + bound[3] + topMost),
                             (0, 255, 0), 3)


    leftMost2, topMost2, rightMost2, bottomMost2 = getBBox(
        resulting_squares, width, height)
    print(len(resulting_squares), leftMost, topMost, rightMost, bottomMost, resulting_squares)
    frame = cv.rectangle(frame, (leftMost + leftMost2, topMost + topMost2), (rightMost2 + leftMost, bottomMost2 + topMost), (125, 0, 125), 3)
    subheight, subwidth = normalized_strong_copy.shape
    ydif = bottomMost2 - topMost2
    xdif = rightMost2 - leftMost2
    if ydif > xdif:
        rightMost2 = min(rightMost2 + xdif // 2, subwidth - 1)
        leftMost2 = max(leftMost2 - xdif // 2, 0)
    else:
        topMost2 = max(topMost2 - ydif // 2, 0)
        bottomMost2 = min(bottomMost2 + ydif // 2, subheight - 1)
    if cut.shape[0] <= 0 or cut.shape[1] <= 0:
        cut = np.array([[0]], dtype=np.uint8)
    cut = normalized_strong_copy[topMost2:bottomMost2 + 1, leftMost2:rightMost2 + 1]
    cv.imshow('frame5', cut)
    # cv.imshow('frame5', normalized_strong)
    # corners_r = relaxation_algorithm(normalize_image_cutoff(corners, 0, 1.0), 0.5, -0.5)
    # frame_copy = np.copy(frame)
    # frame_copy[corners_r > corners_r.max() * 0.9] = [0, 0, 255]
    # frame_copy[corners_r < corners_r.min() * 0.1] = [255, 0, 0]
    # cv.imshow('corners2', frame_copy)

    # print(features)
    frame[corners > corners.max() * 0.1] = [0, 0, 255]
    frame[corners < corners.min() * 0.1] = [255, 0, 0]
    cv.imshow('corners', frame)
    if cv.waitKey(1) == ord('q'):
        break

    count += 30
    cap.set(cv.CAP_PROP_POS_FRAMES, count)


cap.release()
cv.destroyAllWindows()

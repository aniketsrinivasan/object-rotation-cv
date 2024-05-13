import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

METHOD_KP = 1
METHOD_CIRC = 2


# read_images(img1, img2) consumes paths to two images, reads them, and applies augmentation
#   and processing steps. Returns processed images.
def read_images(img1, img2) -> tuple:
    # Set the optimal kernel size for Gaussian blurring (cleans unwanted noise):
    BLUR_KERNEL_SIZE = 13
    # Set best value for blockSize and C for adaptive thresholding:
    ADAPTIVE_BLOCKSIZE = 11
    ADAPTIVE_C = 2

    print(f"Reading images {img1}, {img2}...")

    # Reading in img1 and applying Gaussian blur:
    img1_gray = cv.imread(img1, cv.IMREAD_GRAYSCALE)
    img1_gray = cv.medianBlur(img1_gray, BLUR_KERNEL_SIZE)

    # Reading in img2 and applying Gaussian blur:
    img2_gray = cv.imread(img2, cv.IMREAD_GRAYSCALE)
    img2_gray = cv.medianBlur(img2_gray, BLUR_KERNEL_SIZE)

    # Applying image transformation to detect keypoints more reliably:
    gray1 = cv.adaptiveThreshold(img1_gray, 255,
                                 cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                 ADAPTIVE_BLOCKSIZE, ADAPTIVE_C)
    gray2 = cv.adaptiveThreshold(img2_gray, 255,
                                 cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                 ADAPTIVE_BLOCKSIZE, ADAPTIVE_C)

    print(f"Images read successfully.")
    return gray1, gray2


# find_circle(img) finds the "best" circle in the center of the object. This is used to
#   eventually find a rotation axis.
def find_circle(img, display=False):
    print(f"Identifying key circle in {img}...")
    # BLUR_KERNEL_SIZE for medianBlur in image augmentation:
    BLUR_KERNEL_SIZE = 25

    image = cv.imread(img)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blurred = cv.medianBlur(image_gray, BLUR_KERNEL_SIZE)

    # Parameters used for identifying the circle(s) using HoughCircles:
    MIN_DIST = 400
    param1 = 30
    param2 = 50         # smaller value => more false circles
    minRadius = 100     # smallest circle radius
    maxRadius = 250     # largest circle radius

    circles = cv.HoughCircles(image_blurred,
                              cv.HOUGH_GRADIENT,
                              1, MIN_DIST,
                              param1=param1,
                              param2=param2,
                              minRadius=minRadius,
                              maxRadius=maxRadius)

    if circles is not None:
        # first circle in "circles" in the form [x, y, r]
        print(f"Key circle identified.")
        circles = np.uint16(np.around(circles))

        # Displaying the circles if display=True:
        if display:
            for i in circles[0, :]:
                cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 3)
                plt.imshow(image)
                plt.title("The 'key circle(s)' in the image (at least one at center of object)")
                plt.show()
        return circles[0][0]
    else:
        # We have not found any circles, so the algorithm cannot continue.
        raise Exception("No circles found.")


# find_center(img) finds the rotation axis (as a point) of the object in img. Returns
#   this rotation axis point.
def find_center(img, display=False) -> list[int]:
    circle = find_circle(img, display)
    return [int(circle[0]), int(circle[1])]


# find_keypoints(img1, img2) finds the keypoints of two images to simulate finding the
#   presence of an "object".
def find_keypoints(img1, img2, display=False) -> tuple[list, list]:
    print(f"Calculating keypoints...")
    # Reading in images and applying necessary transforms:
    gray1, gray2 = read_images(img1, img2)

    # Using the ORB detector to find keypoints in both images:
    orb = cv.ORB_create(nfeatures=100, scoreType=cv.ORB_FAST_SCORE)
    img1_keypoints = orb.detect(gray1, None)
    img2_keypoints = orb.detect(gray2, None)

    if display:
        img1_display = cv.drawKeypoints(gray1, img1_keypoints, None, color=(0, 255, 0), flags=0)
        img2_display = cv.drawKeypoints(gray2, img2_keypoints, None, color=(0, 255, 0), flags=0)

        plt.imshow(img1_display), plt.title("Initial image, processed with keypoints"), plt.show()
        plt.imshow(img2_display), plt.title("Second image, processed with keypoints"), plt.show()

    img1_keypoints = cv.KeyPoint_convert(img1_keypoints)
    img2_keypoints = cv.KeyPoint_convert(img2_keypoints)

    print(f"Keypoints identified successfully.")
    return img1_keypoints, img2_keypoints


# find_keypoint_average(keypoints) finds the "center" point of all the keypoints by computing
#   the coordinate-wise mean.
def find_keypoint_average(keypoints) -> tuple[int, int]:
    # Initializing variables:
    x_avg, x_sum = 0, 0
    y_avg, y_sum = 0, 0
    keypoints_len = len(keypoints)

    # Iterating over keypoints and summing x- and y-coordinates:
    for keypoint in keypoints:
        x_sum += keypoint[0]
        y_sum += keypoint[1]

    # Calculating average:
    x_avg = int(x_sum / keypoints_len)
    y_avg = int(y_sum / keypoints_len)

    return x_avg, y_avg


# find_points_grid(x1, y1) creates a "grid" of points in a lattice, around the
#   initial point (x1, y1) provided.
def find_keypoints_grid(img=None, keypoints=None) -> np.ndarray:
    # Setting the "spread" of the grid (in pixels):
    GRID_SPREAD = 50

    if (img is None) and (keypoints is not None):
        x1, y1 = find_keypoint_average(keypoints)
    elif (img is not None) and (keypoints is None):
        # circ is in the form [x, y, r]
        circ = find_center(img)
        x1 = circ[0]
        y1 = circ[1]
    else:
        raise NotImplementedError("Both image and keypoints are None.")

    # Creating the grid of points:
    grid = np.array([
        [x1, y1],
        [x1+GRID_SPREAD, y1], [x1, y1+GRID_SPREAD], [x1+GRID_SPREAD, y1+GRID_SPREAD],
        [x1-GRID_SPREAD, y1], [x1, y1-GRID_SPREAD], [x1-GRID_SPREAD, y1-GRID_SPREAD],
        [x1+GRID_SPREAD, y1-GRID_SPREAD], [x1-GRID_SPREAD, y1+GRID_SPREAD],
        [x1+2*GRID_SPREAD, y1], [x1-2*GRID_SPREAD, y1],
        [x1, y1+2*GRID_SPREAD], [x1, y1-2*GRID_SPREAD]
    ])

    return grid


# mask_crop_resize(mask, bbox) takes a mask and a bbox in the format [xywh] and returns
#   a cropped and resized version of the mask. A new center position is also calculated.
def mask_crop_resize(mask, bbox, center):
    RESIZE_DIM = 512
    cropped_mask = mask[bbox[1]:bbox[1]+bbox[3],
                        bbox[0]:bbox[0]+bbox[2]]

    # Calculating the "new" center of the cropped mask:
    new_center = [center[0]-bbox[0], center[1]-bbox[1]]

    # Reshape as float32 entries to allow cv.resize to work:
    cropped_mask = cropped_mask.reshape((bbox[3], bbox[2])).astype('float32')
    resized_mask = cv.resize(cropped_mask,
                             (RESIZE_DIM, RESIZE_DIM),
                             interpolation=cv.INTER_LINEAR)

    # Updating the center after resizing:
    new_center[0] = (RESIZE_DIM / bbox[2]) * new_center[0]
    new_center[1] = (RESIZE_DIM / bbox[3]) * new_center[1]

    return resized_mask, new_center, RESIZE_DIM

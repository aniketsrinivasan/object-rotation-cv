from utils import (find_keypoints, find_center, model_get_masks, model_get_mask_bbox,
                   find_keypoints_grid, mask_crop_resize)
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# ~~~~~~~~~~~~~~~~ FOR USER TO MODIFY ~~~~~~~~~~~~~~~~
# Change the image paths here to the two images used:
IMG_PATH_1 = "images/template_images/screenshot_2024-05-02_17-22-50.jpg"
IMG_PATH_2 = "images/test_images/screenshot_2024-05-02_17-23-01.jpg"

# If you would like the result saved into the "results" directory (recommended):
SAVE_RESULTS = True
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# find_difference(original_img, rotated_img) calculates a numeric difference between the two
#   images. Works only when both images are grayscale. Works best when both are binary.
# requires:     original_img, rotated_img are grayscale (monochrome)
def find_difference(original_img, rotated_img) -> float:
    absdiff = abs(original_img - rotated_img)
    diff_measure = np.sum(absdiff)
    return diff_measure


# find_best_angle(original_img, rotated_img, rotation_point, img_dimension) finds the angle of
#   rotation of original_img that "minimizes" the difference between both images.
# requires:     original_img, rotated_img are grayscale (monochrome)
def find_best_angle(original_img, rotated_img, rotation_point, img_dimension):
    # The iteration step for angles (can be negative, too):
    ANGLE_STEP = 1

    # Initializing variables:
    best_angle = None               # angle has not been found yet
    best_diff = float("inf")        # set to infinity initially
    diffs = []                      # list of differences (to be returned)
    i = 0                           # iterator
    while abs(i) < 360:
        # Rotating the image by i degrees around rotation_point:
        matrix = cv.getRotationMatrix2D(center=(rotation_point[0], rotation_point[1]),
                                        angle=i,
                                        scale=1.0)
        this_rotation = cv.warpAffine(original_img, matrix,
                                      dsize=(img_dimension, img_dimension))
        this_diff = find_difference(this_rotation, rotated_img)

        # Updating best:
        if this_diff < best_diff:
            best_diff = this_diff
            best_angle = i

        # Storing list of difference values (for future analysis):
        diffs.append(this_diff)

        i += ANGLE_STEP

    return best_angle, best_diff, diffs


# Consumes two paths, and returns the initial image (at path1) with a bounding box, as well
#   as the predicted angle of rotation.
def main(path1, path2):
    # Create keypoints of the images:
    keypoints1, keypoints2 = find_keypoints(path1, path2)

    # Create keypoint grids for each image:
    keypoints1_grid = find_keypoints_grid(keypoints1)
    keypoints2_grid = find_keypoints_grid(keypoints2)

    # Finding the centers of the object in each image:
    obj_center1 = find_center(path1)
    obj_center2 = find_center(path2)

    # Getting masks and setting best mask:
    masks1 = model_get_masks(IMG_PATH_1, keypoints1_grid)
    mask1 = masks1[0]       # best mask from masks1
    masks2 = model_get_masks(IMG_PATH_2, keypoints2_grid)
    mask2 = masks2[0]       # best mask from masks2

    # Create bounding boxes:
    bbox1 = model_get_mask_bbox(masks1)
    bbox2 = model_get_mask_bbox(masks2)

    # Processing the masked images by cropping and resizing:
    mask1_processed, mask1_center, _dim = mask_crop_resize(mask1, bbox1, obj_center1)
    mask2_processed, mask2_center, _dim = mask_crop_resize(mask2, bbox2, obj_center2)

    print(f"Attempting to find best angle...")
    # Compare mask1_processed and mask2_processed using the comparator function:
    best_angle, best_diff, diffs = find_best_angle(mask1_processed, mask2_processed, mask1_center, _dim)
    print(f"Best angle: {best_angle}. Best diff: {best_diff}.")

    # Creating an image and binary difference image to plot:
    rot_matrix = cv.getRotationMatrix2D(center=(mask1_center[0], mask1_center[1]),
                                    angle=best_angle,
                                    scale=1.0)
    best_rotation = cv.warpAffine(mask1_processed, rot_matrix,
                                  dsize=(_dim, _dim))

    diff_img = abs(mask2_processed - best_rotation)
    # Plotting the two masks:
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(best_rotation, cmap="gray")
    axarr[0, 1].imshow(mask2_processed, cmap="gray")
    axarr[1, 0].imshow(diff_img, cmap="gray")
    plt.show()

    image1 = cv.imread(IMG_PATH_1)
    image2 = cv.imread(IMG_PATH_2)

    cv.rectangle(image1,
              (bbox1[0], bbox1[1]),
              (bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]),
              color=(255, 0, 0),
              thickness=3)
    cv.rectangle(image2,
                 (bbox2[0], bbox2[1]),
                 (bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]),
                 color=(255, 0, 0),
                 thickness=3)

    plt.imshow(image1, cmap="gray")
    plt.title("Initial object with identified bounding box")
    plt.show()

    cv.imwrite("results/result1.jpg", image1)
    cv.imwrite("results/result2.jpg", image2)

    # Returning image with bounding box, and angle of rotation:
    return image1, best_angle


if __name__ == '__main__':
    main(IMG_PATH_1, IMG_PATH_2)

from utils import (find_keypoints, find_center, model_get_masks, model_get_mask_bbox,
                   find_keypoints_grid, mask_crop_resize)
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from tqdm import tqdm

# Methods: there are two methods implemented here.
#   METHOD_KP:      keypoint analysis to find the object
#   METHOD_CIRC:    circle identification to find the object
METHOD_KP = 1
METHOD_CIRC = 2

# ~~~~~~~~~~~~~~~~ FOR USER TO MODIFY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Names of the two images (without the .jpg):
TEMPLATE_IMG = "template3"
TEST_IMG = "screenshot_2024-05-02_17-23-22"

# Change the image paths here to the two images used:
IMG_PATH_1 = f"images/template_images/{TEMPLATE_IMG}.jpg"
IMG_PATH_2 = f"images/test_images/{TEST_IMG}.jpg"

# Define chosen method (as described above):
method = METHOD_KP

# If you would like to see a graphic output as the program runs, set to True.
DISPLAY = True
# If you would like the result saved into the "results" directory (recommended):
SAVE_RESULTS = False

# To generate results for all files in "images" systemically:
#   note: ignores the TEMPLATE_IMG and TEST_IMG and generates all results.
#         ignores SAVE_RESULTS and automatically saves all results.
GENERATE_ALL = False
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
#       path1:  template image
#       path2:  test image
def create_results(path1, path2, test_name=None, template_name=None):
    # Finding mask using the KEYPOINTS method:
    if method == METHOD_KP:
        # Create keypoints of the images:
        keypoints1, keypoints2 = find_keypoints(path1, path2, DISPLAY)

        # Create keypoint grids for each image:
        keypoints1_grid = find_keypoints_grid(keypoints=keypoints1)
        keypoints2_grid = find_keypoints_grid(keypoints=keypoints2)

    # Finding mask using the CIRCLE method otherwise:
    else:
        # Read the image in using CV:

        # Create keypoint grids for each image:
        keypoints1_grid = find_keypoints_grid(img=path1)
        keypoints2_grid = find_keypoints_grid(img=path2)

    # Finding the centers of the object in each image:
    obj_center1 = find_center(path1, DISPLAY)
    obj_center2 = find_center(path2, DISPLAY)

    # Getting masks and setting best mask:
    masks1 = model_get_masks(path1, keypoints1_grid, DISPLAY)
    mask1 = masks1[0]       # best mask from masks1
    masks2 = model_get_masks(path2, keypoints2_grid, DISPLAY)
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

    if DISPLAY:
        # Plotting the two masks:
        f, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(best_rotation, cmap="gray")
        axarr[0, 1].imshow(mask2_processed, cmap="gray")
        axarr[1, 0].imshow(diff_img, cmap="gray")
        plt.show()

    image1 = cv.imread(path1)
    image2 = cv.imread(path2)

    # Drawing the rectangle, adding the angle to the image:
    cv.rectangle(image2,
                 (bbox2[0], bbox2[1]),
                 (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]),
                 color=(255, 0, 0),
                 thickness=3)
    cv.putText(image2,
               f"Initial image. Rotation estimate: {best_angle}",
               (50, 60), cv.FONT_HERSHEY_SIMPLEX,
               1.5, (255, 255, 255), 5)

    if DISPLAY:
        # Displaying the resultant image:
        plt.imshow(image2, cmap="gray")
        plt.title("Initial object with identified bounding box")
        plt.show()

    # Saving the file(s):
    if SAVE_RESULTS and (test_name is not None) and (template_name is not None):
        if not os.path.exists(f"results/method_{method}/{template_name}_result.jpg"):
            cv.rectangle(image1,
                         (bbox1[0], bbox1[1]),
                         (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]),
                         color=(255, 0, 0),
                         thickness=3)
            cv.imwrite(f"results/method_{method}/{template_name}_result.jpg", image1)
        if not os.path.exists(f"results/method_{method}/{template_name}/{test_name}_result.jpg"):
            cv.imwrite(f"results/method_{method}/{template_name}/{test_name}_result.jpg", image2)

    # Returning image with bounding box, and angle of rotation:
    return image1, best_angle


if __name__ == '__main__':
    # GENERATE_ALL will ignore the single template and test image.
    if GENERATE_ALL:
        SAVE_RESULTS = True
        for template_img in tqdm(os.listdir("images/template_images")):
            template_name = template_img.split(".")[0]
            img_path_1 = f"images/template_images/{template_name}.jpg"
            print(f"Processing {template_img}.")

            for test_img in tqdm(os.listdir("images/test_images")):
                test_name = test_img.split(".")[0]
                img_path_2 = f"images/test_images/{test_name}.jpg"
                print(f"Processing {test_img} for {template_img}.")

                try:
                    create_results(img_path_1, img_path_2, test_name, template_name)
                except Exception as e:
                    continue

    else:
        create_results(IMG_PATH_1, IMG_PATH_2)

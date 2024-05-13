import numpy as np
import cv2 as cv
from segment_anything import SamPredictor, build_sam_vit_b
from huggingface_hub import hf_hub_download
import torch
from torchvision.ops import masks_to_boxes
import matplotlib.pyplot as plt


# show_mask(mask, ax, random_color) displays a mask on an image using pyplot. Returns nothing.
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# model_build_SAM() builds the SamPredictor model used and returns it.
def model_build_SAM() -> SamPredictor:
    print("Building the SamPredictor...")
    # Downloading the predictor:
    #   note: checks by default whether predictor pre-exists in cache.
    chkpt_path = hf_hub_download("ybelkada/segment-anything",
                                 "checkpoints/sam_vit_b_01ec64.pth")
    predictor = SamPredictor(build_sam_vit_b(checkpoint=chkpt_path))
    print("SamPredictor built.")
    return predictor


# model_get_masks(img, keypoints, predictor) gets an array of masks predicted in
#   image img using predictor, with keypoints. Returns this array of masks.
def model_get_masks(img_path, keypoints, display=True) -> np.ndarray:
    print(f"Getting masks for {img_path}...")
    # Building predictor:
    predictor = model_build_SAM()

    # Reading image:
    img = cv.imread(img_path)

    # Setting current image to predict:
    predictor.set_image(img)

    # Initializing labels (we only have one class):
    labels = np.array([1 for _ in range(len(keypoints))])

    # Running the SAM predictor:
    masks, scores, logits = predictor.predict(
        point_coords=keypoints,
        point_labels=labels,
        multimask_output=True,
    )
    print(f"Masks identified for {img_path}.")

    if not display:
        return masks

    # If display=True, then we will display this mask using show_mask:
    plt.imshow(img)
    show_mask(masks[0], plt.gca())
    plt.scatter(keypoints[:, 0], keypoints[:, 1], marker="o", color="red")
    plt.title("Object identified with mask (highlighted blue) and keypoints_grid (red)")
    plt.show()

    return masks


# model_get_mask_bbox(masks) takes an array of masks and returns a bounding box in
#   the form [x, y, w, h] for the best mask in masks.
def model_get_mask_bbox(masks) -> list:
    # Converting to tensor:
    masks_tensor = torch.from_numpy(masks)
    boxes = np.array(masks_to_boxes(masks_tensor))

    # Getting bounding box in the [xywh] form:
    #   note: we are choosing mask "0" from masks, which is the "best" one.
    rect_x, rect_y = int(boxes[0][0]), int(boxes[0][1])
    rect_width = int(boxes[0][2] - rect_x)
    rect_height = int(boxes[0][3] - rect_y)

    return [rect_x, rect_y, rect_width, rect_height]

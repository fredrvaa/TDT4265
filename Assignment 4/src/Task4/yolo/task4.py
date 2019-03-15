import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from drawing_utils import read_classes, draw_boxes, scale_boxes

# GRADED FUNCTION: yolo_filter_boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """ Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        box_confidence -- np.array of shape (19, 19, 5, 1)
        boxes -- np.array of shape (19, 19, 5, 4)
        box_class_probs -- np.array of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
            then get rid of the corresponding box
    
    Returns:
        scores -- np.array of shape (None,), containing the class probability score for selected boxes
        boxes -- np.array of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- np.array of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Step 1: Compute box scores
    scores = box_confidence * box_class_probs
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    max_box_scores = np.amax(scores, axis=-1)
    box_classes= np.argmax(scores, axis=-1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = max_box_scores >= threshold

    # Step 4: Apply the mask to scores, boxes and classes
    scores = max_box_scores[filtering_mask]
    boxes = boxes[filtering_mask]
    classes = box_classes[filtering_mask]
    
    return scores, boxes, classes

def iou(prediction_box, gt_box):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    #Used supplied code from https://gist.github.com/hukkelas/74f0420e64309a46f9751629dda710da
    #as this gives correct output.

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1)*(yi2 - yi1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[3] - box1[1])*(box1[2]- box1[0])
    box2_area = (box2[3] - box2[1])*(box2[2]- box2[0])
    union_area = (box1_area + box2_area) - inter_area
    
    # compute the IoU
    iou = inter_area / union_area

    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
        scores -- np.array of shape (None,), output of yolo_filter_boxes()
        boxes -- np.array of shape (None, 4), output of yolo_filter_boxes() 
            that have been scaled to the image size (see later)
        classes -- np.array of shape (None,), output of yolo_filter_boxes()
        max_boxes -- integer, maximum number of predicted boxes you'd like
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. 
    Note also that this function will transpose the shapes of scores, boxes, classes. 
    This is made for convenience.
    """
    sorted_indeces = scores.argsort()[::-1]

    sorted_scores = scores[sorted_indeces]
    sorted_boxes = boxes[sorted_indeces]
    sorted_classes = classes[sorted_indeces]

    nms_indices = []
    checked_boxes_indices = []

    # Use iou() to get the list of indices corresponding to boxes you keep
    for index, highest_box in enumerate(sorted_boxes):
        if index in checked_boxes_indices:
            continue

        nms_indices.append(index)

        if len(nms_indices) == max_boxes:
            break

        for box_index, box in enumerate(sorted_boxes[index+1:]):
            if iou(highest_box, box) >= iou_threshold:
                checked_boxes_indices.append(index+box_index+1)

    # Use index arrays to select only nms_indices from scores, boxes and classes
    scores = sorted_scores[nms_indices]
    boxes = sorted_boxes[nms_indices]
    classes = sorted_classes[nms_indices]

    return scores, boxes, classes

    
    

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
        yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 np.array:
                        box_confidence: tensor of shape (None, 19, 19, 5, 1)
                        boxes: tensor of shape (None, 19, 19, 5, 4)
                        box_class_probs: tensor of shape (None, 19, 19, 5, 80)
        image_shape -- np.array of shape (2,) containing the input shape, in this notebook we use 
            (608., 608.) (has to be float32 dtype)
        max_boxes -- integer, maximum number of predicted boxes you'd like
        score_threshold -- real value, if [ highest class probability score < threshold], 
            then get rid of the corresponding box
        iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
        scores -- np.array of shape (None, ), predicted score for each box
        boxes -- np.array of shape (None, 4), predicted box coordinates
        classes -- np.array of shape (None,), predicted class for each box
    """
    
    ### START CODE HERE ### 
    
    # Retrieve outputs of the YOLO model (≈1 line)
        
    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    
    # Scale boxes back to original image shape.

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    
    ### END CODE HERE ###
    
    return scores, boxes, classes

print("----TEST TO SEE EXPECTED OUTPUT TASK 4a) | yolo_filter_boxes----")
#DO NOT EDIT THIS CODE
np.random.seed(0)
box_confidence = np.random.normal(size=(19, 19, 5, 1), loc=1, scale=4)
boxes = np.random.normal(size=(19, 19, 5, 4), loc=1, scale=4)
box_class_probs = np.random.normal(size=(19, 19, 5, 80), loc=1, scale=4)
scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
print("scores[2] = " + str(scores[2]))
print("boxes[2] = " + str(boxes[2]))
print("classes[2] = " + str(classes[2]))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))

print("----TEST TO SEE EXPECTED OUTPUT TASK 4b) | iou----")

#DO NOT EDIT THIS CODE
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4) 
print("iou = " + str(iou(box1, box2)))

print("--------------------------------------------------------")



print("----TEST TO SEE EXPECTED OUTPUT TASK 4b) | yolo_non_max_suppression----")

#DO NOT EDIT THIS CODE
np.random.seed(0)
scores = np.random.normal(size=(54,), loc=1, scale=4)
boxes = np.random.normal(size=(54,4), loc=1, scale=4)
classes = np.random.normal(size=(54,), loc=1, scale=4)
scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
print("scores[2] = " + str(scores[2]))
print("boxes[2] = " + str(boxes[2]))
print("classes[2] = " + str(classes[2]))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))

print("--------------------------------------------------------")



print("----TEST TO SEE EXPECTED OUTPUT TASK 4c) | yolo_eval----")

#DO NOT EDIT THIS CODE
np.random.seed(0)
yolo_outputs = (np.random.normal(size=(19, 19, 5, 1,), loc=1, scale=4),
                np.random.normal(size=(19, 19, 5, 4,), loc=1, scale=4),
                np.random.normal(size=(19, 19, 5, 80,), loc=1, scale=4))
scores, boxes, classes = yolo_eval(yolo_outputs)
print("scores[2] = " + str(scores[2]))
print("boxes[2] = " + str(boxes[2]))
print("classes[2] = " + str(classes[2]))
print("scores.shape = " + str(scores.shape))
print("boxes.shape = " + str(boxes.shape))
print("classes.shape = " + str(classes.shape))

print("--------------------------------------------------------")

# DO NOT CHANGE
image = Image.open("test.jpg")
box_confidence = np.load("box_confidence.npy")
boxes = np.load("boxes.npy")
box_class_probs = np.load("box_class_probs.npy")
yolo_outputs = (box_confidence, boxes, box_class_probs)



# DO NOT CHANGE
image_shape = (720., 1280.)

#DO NOT EDIT THIS CODE
out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, image_shape)

#DO NOT EDIT THIS CODE
# Print predictions info
print('Found {} boxes'.format(len(out_boxes)))
# Draw bounding boxes on the image
draw_boxes(image, out_scores, out_boxes, out_classes)
# Display the results in the notebook
imshow(image)
import matplotlib.pyplot as plt
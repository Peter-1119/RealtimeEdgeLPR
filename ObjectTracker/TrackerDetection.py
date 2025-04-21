import numpy as np
import colorsys
import random

def calc_center_with_label(out_boxes, label_index):
    targetBoxes = out_boxes[out_boxes[:, 4] == label_index]
    centers = []
    for x1, y1, x2, y2, category in targetBoxes:
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        centers.append(center)
        
    return np.array(centers), targetBoxes

def calc_center_without_label(out_boxes):
    centers = []
    for x1, y1, x2, y2, category in out_boxes:
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        centers.append(center)
        
    return np.array(centers)

def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    
    #colors = [(255,99,71) if c==(255,0,0) else c for c in colors ]  # 单独修正颜色，可去除
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors

    
def get_tracking_object(tracker, centers, boundingBox):
    tracker.Update(centers, boundingBox)
    boxes = []
    for track in tracker.tracks:
        box = track.box
        boxes.append([track.track_id, track.alive, box[0], box[1], box[2], box[3], track.label])
    
    return np.array(boxes), tracker.delete_tracks_id
    
# sample
# centers, boxes, number = calc_center(result[:, :4], result[:, 5], result[:, 4], 0, score_limit = 0.5)
# boxes = get_tracking_object(self.tracker, centers, boxes)

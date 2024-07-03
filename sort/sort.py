import numpy as np

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialize the SORT tracker.

        Args:
            max_age (int): Maximum number of frames to keep a track active without association.
            min_hits (int): Minimum number of associate detections before the track is confirmed.
            iou_threshold (float): Intersection over union threshold for associating track and detection.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0

    def update(self, detections):
        """
        Update the SORT tracker with the current frame's detections.

        Args:
            detections (numpy.ndarray): A 2D numpy array of detections in the format [x1, y1, x2, y2, confidence].

        Returns:
            numpy.ndarray: A 1D numpy array of track IDs, one for each detection.
        """
        self.frame_count += 1
        # Update tracks
        tracks = self.tracks.copy()
        for track in tracks:
            if len(detections) > 0:
                # Compute IoU between track and detections
                iou_max = 0
                iou_argmax = -1
                for i, detection in enumerate(detections):
                    iou = self._iou(track.to_tlbr(), detection)
                    if iou > iou_max:
                        iou_max = iou
                        iou_argmax = i
                # Associate track with detection
                if iou_max > self.iou_threshold:
                    track.update(detections[iou_argmax])
                    detections = np.delete(detections, iou_argmax, axis=0)
                else:
                    track.mark_missed()
            # Remove old tracks
            if track.time_since_update > self.max_age:
                self.tracks.remove(track)
        # Create new tracks for unmatched detections
        for detection in detections:
            self.tracks.append(Track(detection))
        # Return track IDs
        return np.array([track.track_id for track in self.tracks])

    def _iou(self, bb_test, bb_gt):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            bb_test (numpy.ndarray): Bounding box in format [x1, y1, x2, y2].
            bb_gt (numpy.ndarray): Bounding box in format [x1, y1, x2, y2].

        Returns:
            float: Intersection over Union in the range [0, 1].
        """
        # Coordinates of the intersection rectangle
        x1 = max(bb_test[0], bb_gt[0])
        y1 = max(bb_test[1], bb_gt[1])
        x2 = min(bb_test[2], bb_gt[2])
        y2 = min(bb_test[3], bb_gt[3])
        # Compute the area of intersection rectangle
        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        # Compute the area of both bounding boxes
        bb1_area = (bb_test[2] - bb_test[0] + 1) * (bb_test[3] - bb_test[1] + 1)
        bb2_area = (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1)
        # Compute the intersection over union
        iou = inter_area / float(bb1_area + bb2_area - inter_area)
        return iou

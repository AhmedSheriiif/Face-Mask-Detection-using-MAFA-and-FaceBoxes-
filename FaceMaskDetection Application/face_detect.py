from face_utils import get_facebox_coords, load_faceboxes


class FaceDetector:
    def __init__(self):
        self.score_thresh = 0.8
        self.net = load_faceboxes()


    def detect_face_in_frame(self, frame):
        dets = get_facebox_coords(frame, self.net)
        faces = []
        locations = []
        for i, det in enumerate(dets):
            xmin = int(round(det[0]))
            ymin = int(round(det[1]))
            xmax = int(round(det[2]))
            ymax = int(round(det[3]))
            score = det[4]

            if score > self.score_thresh:
                faces.append(frame[ymin:ymax, xmin:xmax])
                locations.append((xmin, ymin, xmax, ymax))

        return faces, locations

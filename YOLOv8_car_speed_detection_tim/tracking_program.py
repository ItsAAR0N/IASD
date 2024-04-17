import math


class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        # objects_rect passed by function call
        # objects_rect contain (x1, y1, x2, y2)
        for rect in objects_rect:

            # x1, y1, x2, y2 -> x, y, w, h
            x, y, w, h = rect

            # '//' mean division and take the floor
            # e.g. 3 / 2 = 1.5, 3 // 2 = 1

            # cx and cy is the horizontal an vertical center
            # They together for the center coord of the bounding box

            # x = x1, x+w = x2
            # e.g. x1 = 3, x2 = 4
            # We know the horizontal center of x1 and x2 is 3.5
            # = [3 + (3+1)] / 2
            # Thus:
            # (x + w) // 2 is not correct!!!
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            # If the bounding boxes is too close to the same object which has been
            # bounded by the rectangel already, we treat it as same box id!!!
            same_object_detected = False
            for id, pt in self.center_points.items():
                # pt[0] is previous cx bounded by bounding box in self.center_points
                # pt[1] is previous cy bounded by bounding box in self.center_points
                dist = math.hypot(cx - pt[0], cy - pt[1])


                # Since the new object detected with the center too close to a 
                # previously detected object
                # We will treat it as same id!!!
                # But we also append it to the objects_bbs_ids!!!
                # And overwrite the same id with new cx cy
                if dist < 35:
                    self.center_points[id] = (cx, cy)
#                   print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    # Break out of the closest for-loop:
                    break

            # New object is detected we assign the unique ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            # Replace the id with multiple center with only the latest center!!!
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
"""
This file consists the code for construction of scene graph without the concept of vectorization.
"""

import itertools
import numpy as np

class SceneGraphGeneration():
    def __init__(self, boxes, labels):
        self.counter = 0
        self.predicted_boxes = boxes
        self.predicted_labels = labels
        self.processed_objects = []
        self.object_classes = ['background', 'player', 'score', 'alien', 'shield', 'satellite', 'bullet', 'lives']
        
    def are_objects_within_threshold(self, obj1, obj2, threshold = 17):
        """
        Check if objects are within a certain threshold distance.

        Parameters:
        - obj1 (dict): First object.
        - obj2 (dict): Second object.
        - threshold (float): Threshold distance.

        Returns:
        - bool: True if objects are within the threshold, False otherwise.
        """
        if (obj1['name'] == 'player' and obj2['name'] not in ['lives', 'score']):
            return True
        elif obj1['name'] == 'lives' or obj2['name'] == 'lives' or obj1['name'] == 'score' or obj2['name'] == 'score':
            return False
        
        centroid1 = np.array([obj1['x'] + obj1['w'] / 2, obj1['y'] + obj1['h'] / 2])
        centroid2 = np.array([obj2['x'] + obj2['w'] / 2, obj2['y'] + obj2['h'] / 2])
        distance = np.linalg.norm(centroid1 - centroid2)
        return distance <= threshold
    
    def is_to_the_left_of(self, obj1, obj2):
        """
        Check if obj1 is to the left of obj2.
        """
        return obj1['x'] + obj1['w'] < obj2['x']

    def is_to_the_right_of(self, obj1, obj2):
        """
        Check if obj1 is to the right of obj2.
        """
        return obj1['x'] > obj2['x'] + obj2['w']

    def is_in_front_of(self, obj1, obj2):
        """
        Check if obj1 is in front of obj2.
        """
        return obj1['y'] + obj1['h'] < obj2['y']

    def is_behind(self, obj1, obj2):
        """
        Check if obj1 is behind obj2.
        """
        return obj1['y'] > obj2['y'] + obj2['h']
    
    def generate_relationships(self, obj1, obj2):
        """
        Generate relationships between two objects.

        Parameters:
        - obj1 (dict): First object.
        - obj2 (dict): Second object.

        Returns:
        - dict: Relationship information.
        """
        if self.are_objects_within_threshold(obj1, obj2):
            if self.is_to_the_left_of(obj1, obj2):
                relationship_id = 1
                return {'predicate': 'to the left of', 'object': obj1, 'relationship_id': relationship_id, 'subject': obj2}
            elif self.is_to_the_right_of(obj1, obj2):
                relationship_id = 2
                return {'predicate': 'to the right of', 'object': obj1, 'relationship_id': relationship_id, 'subject': obj2}
            elif self.is_in_front_of(obj1, obj2):
                relationship_id = 3
                return {'predicate': 'in front of', 'object': obj1, 'relationship_id': relationship_id, 'subject': obj2}
            elif self.is_behind(obj1, obj2):
                relationship_id = 4
                return {'predicate': 'behind', 'object': obj1, 'relationship_id': relationship_id, 'subject': obj2}
        return None
    
    def process_objects(self):
        """
        Process a list of objects and generate object information.

        Parameters:
        - boxes (np.ndarray): Bounding box coordinates for multiple objects.
        - labels (np.ndarray): Labels for multiple objects.

        Returns:
        - list: List of object information.
        """
        for box, label in zip(self.predicted_boxes, self.predicted_labels):
            if label not in [7, 2]:
                obj = {
                    "name": self.object_classes[label],
                    "object_id": int(label),
                    "node_id": int(self.counter),
                    "h": round(float(box[3] - box[1]), 2),
                    "w": round(float(box[2] - box[0]), 2),
                    "y": round(float(box[1]), 2),
                    "x": round(float(box[0]), 2)
                }
                self.counter += 1
                self.processed_objects.append(obj)

        return self.processed_objects
    
    def generate_scene_graph(self):
        """
        Generate a scene graph from a list of objects.

        Parameters:
        - boxes (np.ndarray): Bounding box coordinates for multiple objects.
        - labels (np.ndarray): Labels for multiple objects.

        Returns:
        - dict: Scene graph information.
        """
        objects = self.process_objects()
        relationships = []

        # Generate relationships for each pair of objects
        for obj1, obj2 in itertools.product(objects, repeat=2):
            relationship_info = self.generate_relationships(obj1, obj2)
            if relationship_info:
                relationships.append(relationship_info)

        return {"relationships": relationships}
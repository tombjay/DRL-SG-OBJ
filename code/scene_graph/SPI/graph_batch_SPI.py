"""
This file consists the code for the traditional construction of scene graph without the vectorization concept.
"""

import cv2
import numpy as np
import itertools
import torch
from torch_geometric.data import Data, Batch
import time

def generate_bounding_boxes(frames):
    # Modify the function to accept a batch of images
    # Process all images in the batch simultaneously to generate bounding boxes and labels

    # Initialize lists to store boxes and labels for each image in the batch
    batch_boxes = []
    batch_labels = []
    width, height = 160, 210

    object_colors = {
    "player": [[50, 132, 50]], "score": [[50, 132, 50]],
    "alien": [[134, 134, 29]], "shield": [[181, 83, 40]],
    "satellite": [[151, 25, 122]], "bullet": [[142, 142, 142]],
    "lives": [[162, 134, 56]]
    }

    object_id_mapping = {"player": 1, "score": 2, "alien": 3, "shield": 4, "satellite": 5, "bullet": 6, "lives": 7}
    
    frames = frames.cpu().numpy()
    image_resized = cv2.resize(frames, (width, height), cv2.INTER_AREA)

    # Initialize lists to store boxes and labels
    boxes = []
    labels = []

    # Loop through object colors and create bounding boxes
    for obj_class, color in object_colors.items():
        # Adjust coordinates based on object class
        minx, miny, maxx, maxy, closing_dist = get_object_coordinates(obj_class)

        # Find pixels with the specified color range
        masks = [cv2.inRange(image_resized[miny:maxy, minx:maxx, :], np.array(color), np.array(color)) for color in color]

        mask = sum(masks)

        # Perform closing operation
        closed = closing(mask, square(closing_dist))

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Write bounding box coordinates and class label to the lists
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y = x + minx, y + miny

            # Append box coordinates and label to the lists
            boxes.append([x, y, x + w, y + h])
            labels.append(object_id_mapping[obj_class])

    return boxes, labels

def get_object_coordinates(obj_class):
    # Default coordinates
    minx, miny, maxx, maxy, closing_dist = 0, 0, 160, 210, 3

    # Adjust coordinates based on object class
    if obj_class == "player":
        miny, maxy = 180, 195

    elif obj_class == "score":
        maxy, closing_dist = 30, 12
    
    return minx, miny, maxx, maxy, closing_dist

def closing(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def square(size):
    return np.ones((size, size), dtype=np.uint8)


# Scene Graph Generation Class.
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
    
    
# GraphProcessor Class
class GraphProcessor:
    def __init__(self, scene_graphs):
        """
        Initialize the GraphProcessor.

        Parameters:
        - scene_graphs (dict): Scene graphs data.
        """
        self.scene_graphs = scene_graphs
        self.unique_nodes = {}
        self.nodes = []
        
    def process_scene_graphs(self):
        """
        Process scene graphs and generate nodes, edges.
        
        Returns:
        - list: list containing edges (list)
        - dict: dictionary of one-hot vectors of size(max number of relation types).
        """
        edges = []
        relation_ids = set()
        
        for relationship in self.scene_graphs['relationships']:
            subject_node = relationship["subject"]
            object_node = relationship["object"]
            
            # Add subject node to the dict if not present.
            if subject_node["node_id"] not in self.unique_nodes:
                self.unique_nodes[subject_node["node_id"]] = subject_node
            
            # Add object node to the dictionary if not present
            if object_node["node_id"] not in self.unique_nodes:
                self.unique_nodes[object_node["node_id"]] = object_node
            
            self.nodes = list(self.unique_nodes.values())
            sorted_nodes = sorted(self.nodes, key=lambda x: x['node_id'])
            self.nodes = sorted_nodes
            sorted_node_id = [d['node_id'] for d in sorted_nodes]
            
        for relationship in self.scene_graphs['relationships']:
            subject_node = relationship["subject"]
            object_node = relationship["object"]
            sorted_subject_id = sorted_node_id.index(subject_node["node_id"])
            sorted_object_id = sorted_node_id.index(object_node["node_id"])
            
            # Add edge connecting subject to object
            edges.append((sorted_subject_id, sorted_object_id, {"predicate": relationship["predicate"]}, relationship["relationship_id"]))
            
            # Collect unique relation ids to determine size of one-hot encoding
            relation_ids.add(relationship["relationship_id"])
        
        relation_ids = [1, 2, 3, 4]
        num_classes = max(relation_ids)
        one_hot_encoded = torch.eye(num_classes)[[number - 1 for number in relation_ids]]
        one_hot_dict = dict(zip(relation_ids, one_hot_encoded.tolist()))
            
        return edges, one_hot_dict
    
    def create_pytorch_geometric_data(self):
        """
        Create PyTorch Geometric Data object with node and edge attributes.

        Returns:
        - torch_geometric.data.Data: PyTorch Geometric Data object.
        """
        
        processed_edges, one_hot_dict = self.process_scene_graphs() # Convert unique nodes and edges to PyTorch tensors
        # nodes = list(self.unique_nodes.values())
        nodes = self.nodes
        node_features = torch.tensor([[node["x"], node["y"], node["w"], node["h"], node["object_id"]] for node in nodes], dtype=torch.float) # Append nodes with bbox info as node features.
        player_node = next((node["node_id"] for node in nodes if node["name"] == 'player'), None)  
        target_nodes = [node["node_id"] for node in nodes if node["object_id"] in {3, 5}] # Collecting alien & satellite nodes.
        edge_connection = torch.tensor([[edge[0], edge[1]] for edge in processed_edges], dtype=torch.long).t()
        edge_attributes = []
        
        for edge in processed_edges:
            predicate_vector = torch.tensor(one_hot_dict.get(edge[3]))
            if edge[1] == player_node :
                if edge[0] in target_nodes:
                    player_edge = torch.tensor([1.0])
                    edge_attribute = torch.cat((player_edge, predicate_vector), dim=0)
                else :
                    player_edge = torch.tensor([0.0])
                    edge_attribute = torch.cat((player_edge, predicate_vector), dim=0)
            
            else: 
                player_edge = torch.tensor([0.0])
                edge_attribute = torch.cat((player_edge, predicate_vector), dim=0)
            edge_attributes.append(edge_attribute)
        
        if not edge_attributes:
            # Create a one-hot encoding of length 5 as default value
            default_encoding = torch.zeros(5)
            edge_attributes.append(default_encoding)
        
        edge_attributes_stacked = torch.stack(edge_attributes, dim=0)
            
        # Create PyTorch Geometric Data object
        # data = Data(x=node_features, edge_index=edge_connection)
        data = Data(x=node_features, edge_index=edge_connection, edge_attr=edge_attributes_stacked)
        return data


def graph_batch(frames):
    scene_graphs = []
    batch_graphs = []
    batch_boxes, batch_labels = generate_bounding_boxes(frames)

    # Process the batch of scene graphs    
    # for i in range(len(frames)):
    scene_graph_gen = SceneGraphGeneration(batch_boxes, batch_labels)
    scene_graph = scene_graph_gen.generate_scene_graph()

    # Process the batch of graph data
    # for i in range(len(frames)):
    graph_processor = GraphProcessor(scene_graph)
    processed_graphs = graph_processor.create_pytorch_geometric_data()
    batch_graphs.append(processed_graphs)
    
    batched_graph_obs = Batch.from_data_list(batch_graphs)
    return(batched_graph_obs)

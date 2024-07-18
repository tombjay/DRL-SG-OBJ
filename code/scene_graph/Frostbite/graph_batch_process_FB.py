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

    objects_colors = {
            "WhitePlate": [[214,214,214]], 
            "BluePlate": [[84,138,210]],
            "Bird": [[132,144,252]],
            "hud_objs": [[132,144,252]],
            "house": [[142,142,142],[0,0,0]],
            "greenfish": [[111,210,111]],
            "crab": [[213,130,74]], 
            "clam": [[210,210,64]], 
            "bear": [[111,111,111]], 
            "player": [[162, 98,33], [198,108,58], [142,142,142],[162,162,42]] 
        }

    object_id_mapping = {"background": 0,"player": 1, "WhitePlate": 2, "BluePlate": 3, "Bird": 4, "house": 5, "greenfish": 6, "crab": 7, 
                         "clam": 8, "bear": 9, "frostbite": 10, "hud_objs": 11}
    
    frames = frames.cpu().numpy()
    for frame in frames:

        # Initialize lists to store boxes and labels
        boxes = []
        labels = []

        # Loop through object colors and create bounding boxes
        for obj_class, color in objects_colors.items():
            # Adjust coordinates based on object class
            minx, miny, maxx, maxy, closing_dist = get_object_coordinates(obj_class)

            # Find pixels with the specified color range
            masks = [cv2.inRange(frame[miny:maxy, minx:maxx, :], np.array(color), np.array(color)) for color in color]

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
                
        batch_boxes.append(boxes)
        batch_labels.append(labels)

    return batch_boxes, batch_labels

def get_object_coordinates(obj_class):
    # Default coordinates
    minx, miny, maxx, maxy, closing_dist = 0, 0, 160, 210, 3

    # Adjust coordinates based on object class
    if obj_class == "WhitePlate":
        maxy = 185
    elif obj_class == "player":
        miny = 60
    elif obj_class == "house":
        minx, miny, maxx, maxy, closing_dist = 84, 13, 155, 64, 1
    elif obj_class == "Bird":
        miny, maxy, closing_dist = 75, 180, 5
    elif obj_class == "bear":
        miny, maxy = 13, 75
    elif obj_class in ["crab", "clam", "frostbite"]:
        miny, maxy = 75, 180
    
    return minx, miny, maxx, maxy, closing_dist

def closing(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def square(size):
    return np.ones((size, size), dtype=np.uint8)


# Scene Graph Generation Class.
class SceneGraphGeneration():
    def __init__(self, batch_boxes, batch_labels):
        self.predicted_boxes = batch_boxes
        self.predicted_labels = batch_labels
        self.object_classes = ['background','player', 'WhitePlate', 'BluePlate', 'Bird', 'house', 'greenfish', 'crab', 
                  'clam', 'bear', 'frostbite', 'hud_objs']
        
    def generate_relationships(self, obj1, obj2):
        """
        Generate relationships between two objects.

        Parameters:
        - obj1 (dict): First object.
        - obj2 (dict): Second object.

        Returns:
        - dict: Relationship information.
        """
        obj1_name = obj1['name'].lower()
        obj2_name = obj2['name'].lower()

        # Define the conditions for within_threshold
        player_condition = (obj1_name == 'player' or obj2_name == 'player')
        white_plate_condition = (obj1_name == 'whiteplate' and obj2_name == 'whiteplate')
        blue_plate_condition = (obj1_name == 'blueplate' and obj2_name == 'blueplate')

        # Check if any of the conditions are met
        within_threshold = np.any([player_condition, white_plate_condition, blue_plate_condition])
        
        if within_threshold:
            # Calculate all positional relationships using NumPy operations
            to_left = obj1['x'] + obj1['w'] < obj2['x']
            to_right = obj1['x'] > obj2['x'] + obj2['w']
            in_front = obj1['y'] + obj1['h'] < obj2['y']
            behind = obj1['y'] > obj2['y'] + obj2['h']

            # Create a mask for each relationship condition
            relationship_mask = np.array([to_left, to_right, in_front, behind])

            # Define corresponding relationship IDs and predicates
            relationship_ids = np.array([1, 2, 3, 4])
            predicates = ['to the left of', 'to the right of', 'in front of', 'behind']

            # Find the index of the first True value in the relationship mask
            valid_index = np.argmax(relationship_mask)

            # If any relationship is true, return the corresponding relationship info
            if np.any(relationship_mask):
                relationship_id = relationship_ids[valid_index]
                return {'predicate': predicates[valid_index], 'object': obj1, 'relationship_id': relationship_id, 'subject': obj2}

        return None
    
    def process_objects(self):
        """
        Process a list of objects and generate object information.

        Returns:
        - list: List of object information.
        """
        # Filter out objects with label 11 and extract relevant information
        filtered_indices = np.where(np.array(self.predicted_labels) != 11)[0]
        object_classes = np.array(self.object_classes)
        object_labels = object_classes[np.array(self.predicted_labels)[filtered_indices]]
        object_ids = np.array(self.predicted_labels)[filtered_indices].astype(int)
        node_ids = filtered_indices.astype(int)
        heights = np.round(np.array(self.predicted_boxes)[filtered_indices, 3] - np.array(self.predicted_boxes)[filtered_indices, 1], 2)
        widths = np.round(np.array(self.predicted_boxes)[filtered_indices, 2] - np.array(self.predicted_boxes)[filtered_indices, 0], 2)
        ys = np.round(np.array(self.predicted_boxes)[filtered_indices, 1], 2)
        xs = np.round(np.array(self.predicted_boxes)[filtered_indices, 0], 2)

        # Create a list of dictionaries containing object information
        processed_objects = [{
            "name": name,
            "object_id": object_id,
            "node_id": node_id,
            "h": height,
            "w": width,
            "y": y,
            "x": x
        } for name, object_id, node_id, height, width, y, x in zip(
            object_labels, object_ids, node_ids, heights, widths, ys, xs
        )]

        return processed_objects
    
    
    def generate_scene_graph(self):
        """
        Generate a scene graph from a list of objects.

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
    def __init__(self, scene_graph):
        """
        Initialize the GraphProcessor.

        Parameters:
        - scene_graphs (dict): Scene graphs data.
        """
        self.scene_graph = scene_graph
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

        # Extracting nodes and relationships
        for relationship in self.scene_graph['relationships']:
            subject_node = relationship["subject"]
            object_node = relationship["object"]

            # Add subject node to the dict if not present.
            if subject_node["node_id"] not in self.unique_nodes:
                self.unique_nodes[subject_node["node_id"]] = subject_node

            # Add object node to the dictionary if not present
            if object_node["node_id"] not in self.unique_nodes:
                self.unique_nodes[object_node["node_id"]] = object_node

        # Extract nodes from the unique nodes dictionary and sort them
        self.nodes = sorted(list(self.unique_nodes.values()), key=lambda x: x['node_id'])
        node_ids = np.array([node['node_id'] for node in self.nodes])

        # Initialize dictionary to store one-hot vectors
        one_hot_dict = {}

        for relationship in self.scene_graph['relationships']:
            subject_node = relationship["subject"]
            object_node = relationship["object"]

            # Find indices of subject and object nodes in the sorted node list
            subject_index = np.where(node_ids == subject_node["node_id"])[0][0]
            object_index = np.where(node_ids == object_node["node_id"])[0][0]

            # Add edge connecting subject to object
            edges.append((subject_index, object_index, {"predicate": relationship["predicate"]}, relationship["relationship_id"]))

            # Collect unique relation ids to determine size of one-hot encoding
            relation_ids.add(relationship["relationship_id"])

        # Generate one-hot encoding for relations
        relation_ids = list(relation_ids)
        num_classes = max(relation_ids)
        one_hot_encoded = torch.eye(num_classes)[[number - 1 for number in relation_ids]]
        one_hot_dict = {relation_id: one_hot_encoded[i].tolist() for i, relation_id in enumerate(relation_ids)}

        return edges, one_hot_dict
    
    def create_pytorch_geometric_data(self):
        """
        Create PyTorch Geometric Data object with node and edge attributes.

        Returns:
        - torch_geometric.data.Data: PyTorch Geometric Data object.
        """
        
        processed_edges, one_hot_dict = self.process_scene_graphs()  # Convert unique nodes and edges to PyTorch tensors
        nodes = self.nodes
        node_features = []

        # Iterate through nodes
        for node in nodes:
            # Extract existing features
            features = [node["x"], node["y"], node["w"], node["h"], node["object_id"]]

            # Determine if the object is friendly or enemy
            if node["object_id"] in [1, 2, 3, 5, 6]:
                features.append(1)  # Add 1 for friendly objects
            elif node["object_id"] in [4, 7, 8, 9]:
                features.append(2)  # Add 2 for enemy objects

            # Append node features to the list
            node_features.append(features)

        # Convert the node features list to a torch tensor
        node_features = torch.tensor(node_features, dtype=torch.float)

        player_node = next((node["node_id"] for node in nodes if node["name"] == 'player'), None)
        target_nodes = [node["node_id"] for node in nodes if node["object_id"] in {4, 7, 8, 9}]  # Collecting enemy nodes.
        edge_connection = torch.tensor([[edge[0], edge[1]] for edge in processed_edges], dtype=torch.long).t()
        edge_attributes = []

        for edge in processed_edges:
            predicate_vector = torch.tensor(one_hot_dict.get(edge[3]))
            if edge[1] == player_node:
                if edge[0] in target_nodes:
                    player_edge = torch.tensor([1.0])
                    edge_attribute = torch.cat((player_edge, predicate_vector), dim=0)
                else:
                    player_edge = torch.tensor([0.0])
                    edge_attribute = torch.cat((player_edge, predicate_vector), dim=0)

            else:
                player_edge = torch.tensor([0.0])
                edge_attribute = torch.cat((player_edge, predicate_vector), dim=0)
            edge_attributes.append(edge_attribute)

        edge_attributes_stacked = torch.stack(edge_attributes, dim=0)

        # Create PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_connection, edge_attr=edge_attributes_stacked)
        return data


def graph_batch(frames):
    scene_graphs = []
    batch_graphs = []
    batch_boxes, batch_labels = generate_bounding_boxes(frames)

    # Process the batch of scene graphs    
    for i in range(len(frames)):
        scene_graph_gen = SceneGraphGeneration(batch_boxes[i], batch_labels[i])
        scene_graph = scene_graph_gen.generate_scene_graph()
        scene_graphs.append(scene_graph)

    # Process the batch of graph data
    for i in range(len(frames)):
        graph_processor = GraphProcessor(scene_graphs[i])
        processed_graphs = graph_processor.create_pytorch_geometric_data()
        batch_graphs.append(processed_graphs)
    
    batched_graph_obs = Batch.from_data_list(batch_graphs)
    return(batched_graph_obs)

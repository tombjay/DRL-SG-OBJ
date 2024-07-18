"""
This file consists the code for processing the scene graph and convert them into Pytorch geometric's data object.
"""

import torch
from torch_geometric.data import Data

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
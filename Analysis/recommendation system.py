import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from pyvis.network import Network
import gravis as gv
from sklearn.cluster import KMeans
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, GraphConv
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import negative_sampling

#import dataset we got
layer0 = pd.read_csv('/data/peng/MyProject/STAT_402_Project-main22/Data_Scraping/Data_collecting_v_2/layer/0_layer.csv')
layer1 = pd.read_csv('/data/peng/MyProject/STAT_402_Project-main22/Data_Scraping/Data_collecting_v_2/layer/1_layer.csv')

#construct a Graph 
G1 = nx.DiGraph()

for df in [layer0, layer1]:
    for index, row in df.iterrows():
        aid = row['aid']
        node = row['node']
        title_length = len(row['title'])
        ctime_length = row['ctime']
        duration_length = row['duration']
        tid = row['tid']
        G1.add_node(aid, title_length=title_length, ctime_length=ctime_length, duration_length=duration_length, tid=tid)
        G1.add_edge(node, aid)
        
# Re-index nodes
node_map = {node: i for i, node in enumerate(G1.nodes())}
G1 = nx.relabel_nodes(G1, node_map, copy=True)


node_features = [(data.get('title_length', 0), data.get('ctime_length', 0), 
                  data.get('duration_length', 0), data.get('tid', 0)) 
                 for node, data in G1.nodes(data=True)]


# Convert edge list to re-indexed node indices
edge_index = torch.tensor([(node_map[u], node_map[v]) for u, v in G1.edges() if u in node_map and v in node_map], dtype=torch.long).t().contiguous()

# Sort node features according to the new node indices
sorted_node_features = [node_features[node_map[node]] for node in G1.nodes() if node in node_map]
x = torch.tensor(sorted_node_features, dtype=torch.float)


# Add self-loops to the adjacency matrix
edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

# Create the torch_geometric.data.Data object
data = Data(x=x, edge_index=edge_index)

pagerank = nx.pagerank(G1, alpha=0.85)

#construct GNN
class GNN(nn.Module):
    def __init__(self, num_node_features, embedding_size):
        super(GNN, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = GraphConv(num_node_features, embedding_size)
        self.conv2 = GraphConv(embedding_size, embedding_size)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
for node in G1.nodes():
    if node in pagerank:
        G1.nodes[node]['pagerank'] = pagerank[node]
    else:
        G1.nodes[node]['pagerank'] = 0.0

node_features = []
for node in G1.nodes():
    features = [0.0] * 64  
    features[0] = G1.nodes[node].get('title_length', 0)  
    features[1] = G1.nodes[node].get('ctime_length', 0)  
    features[2] = G1.nodes[node].get('duration_length', 0)  
    features[3] = G1.nodes[node].get('tid', 0)  
    features[4] = G1.nodes[node].get('pagerank', 0)  

    node_features.append(features)
x = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor(list(G1.edges()), dtype=torch.long).t().contiguous()
edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
data = Data(x=x, edge_index=edge_index)

#node embedding
embedding_size = data.x.size(1) 
model = GNN(data.x.size(1), embedding_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    embeddings = model(data.x, data.edge_index)
    loss = criterion(embeddings, data.x)
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
embeddings = model(data.x, data.edge_index)
final_embeddings = embeddings.detach().numpy()


# build a function to construct our recommender
def shortest_path_between_tags(tag1, tag2):
    center1 = None
    center2 = None
    for node, attributes in G1.nodes(data=True):
        if attributes['tid'] == tag1:
            center1 = node
        elif attributes['tid'] == tag2:
            center2 = node
        if center1 is not None and center2 is not None:
            break     
            
    if center1 is None or center2 is None:
        raise ValueError("One or both of the tags are not found in the graph.")

    try:
        shortest_path = nx.shortest_path(G1, source=center1, target=center2)
    except nx.NetworkXNoPath:
#         raise ValueError("No shortest path found between the centers of the tags.")
        shortest_path=0
    if not shortest_path:
        return "No path exists between the tags."
  
    path_nodes = []
    for node in shortest_path:
        node_features = G1.nodes[node]
        path_nodes.append((node, node_features))

   
    return path_nodes
missing_tid_nodes = [node for node, data in G1.nodes(data=True) if 'tid' not in data]
print("Nodes missing 'tid' attribute:", missing_tid_nodes)
G1.remove_nodes_from(missing_tid_nodes)

tag_embeddings = {}

for node, embedding in zip(G1.nodes(), final_embeddings):
    tag = G1.nodes[node].get('tid', 0)  # default value is 0
    if tag not in tag_embeddings:
        tag_embeddings[tag] = []
    tag_embeddings[tag].append(embedding)

tag_centers = {}
for tag, embeddings in tag_embeddings.items():
    tag_centers[tag] = np.mean(embeddings, axis=0) 

# for tag, center in tag_centers.items():
#     print("Tag:", tag)
#     print("Center:", center)

def all_pairs_shortest_paths(tag_ids):
    paths = {}
    tags = list(tag_ids)  # Convert set to list
    for i in range(len(tags)):
        for j in range(i+1, len(tags)):
            path = shortest_path_between_tags(tags[i], tags[j])
            if path is not None and path != "No path exists between the tags.":
                paths[(tags[i], tags[j])] = (path, len(path) - 1)
    return paths


tag_ids = set(nx.get_node_attributes(G1, 'tid').values())
paths = all_pairs_shortest_paths(tag_ids)
for pair, (path, length) in paths.items():
    print(f"Path between {pair[0]} and {pair[1]}: {path}, length: {length}")


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def contrastive_loss(embeddings, edge_index):
    pos_a = embeddings[edge_index[0]]
    pos_b = embeddings[edge_index[1]]
    pos_sim = F.cosine_similarity(pos_a, pos_b, dim=-1)
    neg_index = negative_sampling(edge_index, num_nodes=embeddings.size(0))
    neg_a = embeddings[neg_index[0]]
    neg_b = embeddings[neg_index[1]]
    neg_sim = F.cosine_similarity(neg_a, neg_b, dim=-1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_sim, torch.ones_like(pos_sim))
    neg_loss = F.binary_cross_entropy_with_logits(neg_sim, torch.zeros_like(neg_sim))
    
    return pos_loss + neg_loss

model = GNN(data.num_features, 64)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    embeddings = model(data.x, data.edge_index)
    loss = contrastive_loss(embeddings, data.edge_index)
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))


def recommend_videos(model, data, target_tag, initial_video, num_recommendations=10):
    model.eval()
    with torch.no_grad():
        all_embeddings = model(data.x, data.edge_index)

    initial_video = node_map[initial_video]
    target_videos = [v for v, data in G1.nodes(data=True) if data.get('tid') == target_tag]
    initial_embedding = all_embeddings[initial_video].unsqueeze(0)
    target_embeddings = all_embeddings[target_videos]
    distances = torch.norm(target_embeddings - initial_embedding, dim=1)
    recommendations = torch.argsort(distances)[:num_recommendations]
    return [target_videos[i] for i in recommendations]


recommendations = recommend_videos(model, data, target_tag= 17, initial_video=311795251, num_recommendations=10)
print(recommendations)

# interface
initial_video = input("input initial video here")
target = input("input target tag here")
num = 5
node2aid = pd.read_csv('output.csv')

while True:
	if target in tag_ids:
		target_tag = target
		break
	else:
		target = input("input correct target tag here")
	
while df[df['aid']==initial_video][0]['tag'] == target_tag:
	recommendations = recommend_videos(model,data,target_tag=target_tag,initial_video=initial_video,num_recommendation=num)
    aids = [node2aid[node2aid['Node']==recommendation].loc[:,['ID']].values[0][0] for recommendation in recommendations]
	print(aids)
	next_video = input("put the index of next video here")
    initial_video = recommendations[next_video]
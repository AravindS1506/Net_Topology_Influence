import networkx as nx
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as patches

def dfs_find_back_edges(G, node, visited, rec_stack, back_edges,target):
    visited.add(node)
    rec_stack.add(node)
    neighbors = list(G.neighbors(node))
    random.shuffle(neighbors)
    for neighbor in neighbors:
        if neighbor not in visited:
            dfs_find_back_edges(G, neighbor, visited, rec_stack, back_edges,target)
        elif neighbor in rec_stack:
            # This edge (node -> neighbor) is a back edge
            if(neighbor!=target):
                back_edges.append((node, neighbor))
    rec_stack.remove(node)

def get_gs1_graph(G,target):
    mini=25
    mini_back=[] 
    G1=G.copy()  
    for i in range(0,200000): 
        visited=set()
        rec_stack=set()
        back_edges=[]
        dfs_find_back_edges(G,target,visited,rec_stack,back_edges,target)
        if(mini>len(back_edges)):   
            G1=G.copy()
            count=0
            for (u,v) in back_edges:
                G1.remove_edge(u,v)
                if nx.is_strongly_connected(G1):
                    count+=1
                else:
                    G1.add_edge(u, v)
            if(count==len(back_edges)):
                mini=count
                mini_back=back_edges
    back_edges=mini_back

    G1=G.copy()
    count=0
    for (u,v) in back_edges:
        G1.remove_edge(u,v)
        if nx.is_strongly_connected(G1):
            count+=1
        else:
            G1.add_edge(u, v)
    return G1

def get_influence_centrality(G,s1,s2):
    adj_matrix=nx.adjacency_matrix(G)
    adj_matrix_dense = adj_matrix.todense()

    adj=np.array(adj_matrix_dense)
    in_degree=adj.sum(axis=0)
    adj=adj/in_degree

    print(adj[0][6],adj[7][6])
    print(adj[3][8],adj[5][8])
    print(adj[2][13],adj[5][13])
    print(adj[4][3],adj[11][3])

    adj=adj.T
    length=adj.shape[0]
    stub=np.zeros((length,1))
    stub[s1-1,0]=0.7
    stub[s2-1,0]=0.1
    beta=np.diag(stub.flatten())
    P=np.dot(np.linalg.inv(np.eye(length)-np.dot((np.eye(length)-beta),adj)),beta)
    return np.dot(np.ones(length).T,P)/length

def get_avg_opinion(G,s1,s2):
    adj_matrix=nx.adjacency_matrix(G)
    adj_matrix_dense = adj_matrix.todense()

    adj=np.array(adj_matrix_dense)
    in_degree=adj.sum(axis=0)
    adj=adj/in_degree
    adj=adj.T
    length=adj.shape[0]
    stub=np.zeros((length,1))
    stub[s1-1,0]=0.7
    stub[s2-1,0]=0.1
    beta=np.diag(stub.flatten())
    P=np.dot(np.linalg.inv(np.eye(length)-np.dot((np.eye(length)-beta),adj)),beta)
    x0=np.random.rand(length,1)
    x0[s1-1]=0
    x0[s2-1]=1
    xf=np.dot(P,x0)
    return np.dot(np.ones(length).T,xf)/length

def get_regions(G,max_node):
    G1=G.copy()
    for u in list(G.predecessors(max_node)):
        G1.remove_edge(u, max_node)
    sorted_list=list(nx.topological_sort(G1))
    distances = {node: 0 for node in sorted_list}
    #Time complexity O(m+n)
    for node in sorted_list:
        for neighbour in G1.neighbors(node):
            distances[neighbour]=max(distances[node]+1,distances[neighbour])
    return distances

def dfs(G, node, visited):
    neighbors = list(G.neighbors(node))
    for neighbor in neighbors:
        if visited[neighbor-1]==0:
            visited[neighbor-1]=1
            dfs(G, neighbor, visited)

def get_class_sets(G,m,s1,s2):
    G1=G.copy()
    n=G1.number_of_nodes()
    for u in list(G.predecessors(m)):
        G1.remove_edge(u, m)

    nodes_in_s1=np.zeros((n,1))
    nodes_in_s2=np.zeros((n,1))
    nodes_not_endorse=np.zeros((n,1))
    dfs(G1,s1,nodes_in_s1)
    dfs(G1,s2,nodes_in_s2)
    for u in list(G.predecessors(s1)):
        G1.remove_edge(u, s1)
    for u in list(G.predecessors(s2)):
        G1.remove_edge(u, s2)
    dfs(G1,m,nodes_not_endorse)

    classification={}
    for i in range(0,n):
        if(nodes_in_s1[i] and nodes_in_s2[i]):
            classification[i+1]=3
        elif(nodes_in_s1[i]):
            classification[i+1]=1
        elif(nodes_in_s2[i]):
            classification[i+1]=2
        else:
            classification[i+1]=4
    return classification, nodes_not_endorse
def verify_gs1(G):
    cycles = list(nx.simple_cycles(G))
    print(f"Total number of cycles: {len(cycles)}")
    if cycles:
        # Convert each cycle to a set of nodes
        cycle_node_sets = [set(cycle) for cycle in cycles]
        
        # Find the intersection of all cycle node sets
        common_nodes = set.intersection(*cycle_node_sets)
        
        if common_nodes:
            print(f"Nodes present in all cycles: {common_nodes}")
        else:
            print("There is no single node present in all cycles.")
    else:
        print("No cycles found in the graph.")


# print(cycles_without_target)
# df = pd.read_csv('C:/Users/aravi/OneDrive/Documents/UCINET data/SAMPSON-1.csv',index_col=0)
# G5 = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
# G1=get_gs1_graph(G,"JOHN_BOSCO")
# with open("sampson_gs1.pkl", "wb") as f:
#     pickle.dump(G1, f)


with open("sampson_gs1.pkl", "rb") as f:
    G = pickle.load(f)

node_list = list(G.nodes())  # Get existing node names
node_mapping = {node: i+1 for i, node in enumerate(node_list)}
G_num = nx.relabel_nodes(G, node_mapping)
reverse_mapping = {v: k for k, v in node_mapping.items()}
# print(reverse_mapping[12])
# print(reverse_mapping[4])
# print(reverse_mapping[5])
# print(node_mapping['LOUIS'])
# central=nx.betweenness_centrality(G_num)
# m=max(central, key=central.get)

# reg=get_regions(G_num,m)
# print(reg)
# young={"MARK","WINFRID","ALBERT","HUGH","BONIFACE"}
# loyal={"PETER","AMBROSE","LOUIS","BERTHOLD","BONAVENTURE"}
# for y in young:
#     for l in loyal:
#         s1=node_mapping[l]
#         s2=node_mapping[y]
#         classif,not_endorse=get_class_sets(G_num,m,s1,s2)
#         z1_endorse={node for node in classif if classif[node]==1 and not_endorse[node-1]==0}
#         z1_not_endorse={node for node in classif if classif[node]==1 and not_endorse[node-1]==1}
#         z2_endorse={node for node in classif if classif[node]==2 and not_endorse[node-1]==0}
#         z2_not_endorse={node for node in classif if classif[node]==2 and not_endorse[node-1]==1}
#         z3={node for node in classif if classif[node]==3}
#         z4={node for node in classif if classif[node]==4 and node!=s1 and node!=s2}
#         if((len(z1_endorse)!=0 or len(z1_not_endorse)!=0 or z3) and (len(z2_endorse)!=0) and len(z2_not_endorse)!=0):
#             print(l,y)

s1=node_mapping["PETER"]
s2=node_mapping["HUGH"]
# classif,not_endorse=get_class_sets(G_num,m,s1,s2)
# z1_endorse={node for node in classif if classif[node]==1 and not_endorse[node-1]==0}
# z1_not_endorse={node for node in classif if classif[node]==1 and not_endorse[node-1]==1}
# z2_endorse={node for node in classif if classif[node]==2 and not_endorse[node-1]==0}
# z2_not_endorse={node for node in classif if classif[node]==2 and not_endorse[node-1]==1}
# z3={node for node in classif if classif[node]==3}
# z4={node for node in classif if classif[node]==4 and node!=s1 and node!=s2}


# print(z2_endorse)
# print(z2_not_endorse)
# print(z1_not_endorse)
# print(z1_endorse)
# print(z3)
# print(z4)

# print(list(G_num.neighbors(2)))
# print(G_num[2][15]['weight'])

# print(list(G_num.neighbors(2)))

# print(list(G_num.neighbors(14)))
# print(G_num.has_edge(16, 15))
# print(list(G_num.neighbors(14)))
# print(G_num.has_edge(6, 10))
# print(list(G_num.neighbors(3)))
# print(G_num.has_edge(8, 10))
# print(G_num.has_edge(12, 14))
# print(G_num[7][9]['weight'])


print(get_influence_centrality(G_num,s1,s2))
G_num[1][7]['weight']-=0.96
G_num.add_edge(8,7,weight=0.96)
print(get_influence_centrality(G_num,s1,s2))
G_num[4][9]['weight']-=0.96
G_num.add_edge(6,9,weight=0.96)
print(get_influence_centrality(G_num,s1,s2))
G_num[3][14]['weight']-=0.96
G_num.add_edge(6,14,weight=0.96)
print(get_influence_centrality(G_num,s1,s2))
G_num[5][4]['weight']-=0.96
G_num.add_edge(12,4,weight=0.96)
print(get_influence_centrality(G_num,s1,s2))

# # G_num[7][9]['weight']-=2.5
# # G_num[12][9]['weight']+=2.5
# # print(get_influence_centrality(G_num,s1,s2))
# # G_num[15][18]['weight']-=0.95
# # G_num.add_edge(8,18,weight=0.95)
# # print(get_influence_centrality(G_num,s1,s2))


# # G_num[3][14]['weight']-=0.95
# # G_num.add_edge(12,14,weight=0.95)
# # print(get_influence_centrality(G_num,s1,s2))
# # G_num[14][10]['weight']-=1.9
# # G_num.add_edge(12,10,weight=1.9)
# # print(get_influence_centrality(G_num,s1,s2))


# # G_num[14][10]['weight']-=1.5
# # G_num.add_edge(6,10,weight=1.5)
# # print(get_influence_centrality(G_num,s1,s2))
# # G_num[14][10]['weight']-=0.4
# # G_num[8][10]['weight']+=0.4
# # print(get_influence_centrality(G_num,s1,s2))
# # G_num[3][14]['weight']-=0.95
# # G_num.add_edge(12,14,weight=0.95)
# # print(get_influence_centrality(G_num,s1,s2))

# # print(s1,s2)
# # print(get_avg_opinion(G_num,s1,s2))
# # G_num[4][3]['weight']-=1.5
# # G_num.add_edge(s2,3,weight=1.5)
# # print(get_influence_centrality(G_num,s1,s2))
# # print(get_avg_opinion(G_num,s1,s2))
# # G_num[3][14]['weight']-=0.95
# # G_num.add_edge(s2,14,weight=0.95)
# # print(get_influence_centrality(G_num,s1,s2))
# # print(get_avg_opinion(G_num,s1,s2))
# # G_num[s1][1]['weight']-=1.5
# # G_num.add_edge(s2,1,weight=1.5)
# # print(get_influence_centrality(G_num,s1,s2))
# # print(get_avg_opinion(G_num,s1,s2))
# # G_num[s1][4]['weight']-=0.95
# # G_num.add_edge(s2,4,weight=0.95)
# # print(get_influence_centrality(G_num,s1,s2))
# # print(get_avg_opinion(G_num,s1,s2))
# # verify_gs1(G_num)
# # print(G_num[1][7]['weight'])
# # print(G_num[4][3]['weight'])
# # print(G_num[3][14]['weight'])
# # print(G_num[s1][1]['weight'])
# # print(G_num[s1][4]['weight'])
# # # if(G_num.has_edge(s2,10)):
# # #     print("Shit")
# # # G_num.add_edge(s2,10,weight=2)


# # # print(get_influence_centrality(G_num,s1,s2))

# # #Code for positioning words
# groups = {}
# for node, x in reg.items():
#     groups.setdefault(x, []).append(node)

# # # Assign positions with centered vertical spacing
# pos = {}
# for x, nodes in groups.items():
#     y_positions = np.linspace(-(len(nodes)-1)/2, (len(nodes)-1)/2, len(nodes))
#     for node, y in zip(nodes, y_positions):
#         pos[node] = (x, y)

# node_stub=[s1,s2]
# short_labels = {node: name[:3] if name[:3]!="BON" else name[:4] for node, name in reverse_mapping.items()}
# node_colours=['red' if node in node_stub else 'orange' for node in G_num.nodes()]




# # # nx.draw(G_num, pos,labels=short_labels, with_labels=True, node_color=node_colours, edge_color='gray', node_size=700, font_size=10)
# # # nx.draw_networkx_edges(G_num, pos, edgelist=[e for e in G_num.edges if e != edge_dot and e!=edge_new], edge_color="gray")

# # # # Draw highlighted edge as blue dotted
# # # nx.draw_networkx_edges(G_num, pos, edgelist=[edge_dot], edge_color="red", style="dotted", width=2)
# # # nx.draw_networkx_edges(G_num, pos, edgelist=[edge_new], edge_color="blue", width=2)
# # # for node, name in reverse_mapping.items():
# # #     text = None
# # #     if name == "PETER":
# # #         text = "s2"
# # #     elif name == "WINFRID":
# # #         text = "s1"
# # #     elif name == "JOHN_BOSCO":
# # #         text = "m"

# # #     if text:
# # #         x, y = pos[node]
# # #         y_offset = -0.2  # Shift text slightly below the node
# # #         bbox_props = dict(boxstyle="round,pad=0.2", edgecolor="black", facecolor="white")
        
# # #         plt.text(x, y + y_offset, text, fontsize=10, ha='center', color='black', bbox=bbox_props)
# # # plt.show()




# #Drawing code

# # Draw graph without node labels
# nx.draw(G_num, pos, with_labels=False, node_color=node_colours, edge_color='gray', node_size=1000, font_size=10)

# # Draw edges except highlighted ones
# nx.draw_networkx_edges(G_num, pos, edgelist=[e for e in G_num.edges], edge_color="gray",width=3.5,arrows=True,arrowsize=45)

# # Draw highlighted edges
# # nx.draw_networkx_edges(G_num, pos, edgelist=[edge_dot], edge_color="red", style="dotted", width=7,arrows=True, arrowstyle='-|>',arrowsize=45)
# # nx.draw_networkx_edges(G_num, pos, edgelist=[edge_new], edge_color="blue", width=4,arrows=True, arrowstyle='-|>',arrowsize=45)

# # Add node labels next to each node
# for node, label in short_labels.items():
#     x, y = pos[node]
#     plt.text(x, y+0.15, label, fontsize=30, ha='center', color='black',weight='bold')  # Shift labels to the right

# # Add additional labels below nodes
# for node, name in reverse_mapping.items():
#     text = None
#     if name == "PETER":
#         text = "s2"
#     elif name == "HUGH":
#         text = "s1"
#     elif name == "JOHN_BOSCO":
#         text = "m"

#     if text:
#         x, y = pos[node]
#         # bbox_props = dict(boxstyle="round,pad=0.2", edgecolor="black", facecolor="white")
#         plt.text(x, y-0.3, text, fontsize=30, ha='center', color='black')  # Shift text below node

# plt.show()

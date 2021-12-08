import networkx as nx
from copy import deepcopy
import queue
import pandas as pd
import re
from random import randrange
import numpy as np
import time
import random
import csv
import math

nodes_file = pd.read_csv('network_node_287.csv')
links_file = pd.read_csv('network_link_287.csv')

G_network=nx.DiGraph()
#tmp=0
for tmp in range(0,len(nodes_file)):
    G_network.add_node(nodes_file.iloc[tmp][0])
    #print(tmp,nodes_file.iloc[tmp][0])
    tmp = tmp+1

#Weight=[]
for tmp in range(0,len(links_file)):
    G_network.add_edge(links_file.iloc[tmp][1],links_file.iloc[tmp][2])
    #Weight=randrange(10,300)
    Weight = 1
    Density = 1
    G_network[links_file.iloc[tmp][1]][links_file.iloc[tmp][2]]['weight'] = Weight
    G_network[links_file.iloc[tmp][1]][links_file.iloc[tmp][2]]['edge'] = links_file.iloc[tmp][0]
    G_network[links_file.iloc[tmp][1]][links_file.iloc[tmp][2]]['edge_speed'] = links_file.iloc[tmp][3]
    G_network[links_file.iloc[tmp][1]][links_file.iloc[tmp][2]]['edge_length'] = links_file.iloc[tmp][4]
    G_network[links_file.iloc[tmp][1]][links_file.iloc[tmp][2]]['lane_num'] = links_file.iloc[tmp][5]
    #G_network[links_file.iloc[tmp][1]][links_file.iloc[tmp][2]]['density'] = Density #vehicle number on a lane
    #print(links_file.iloc[tmp][0],links_file.iloc[tmp][1],float(links_file.iloc[tmp][2]))
    tmp+=1

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
    
average_ffspeed = links_file['Free_Flow_Speed'].mean()
average_length = links_file['Length'].mean()
#get all the edges
Edges = []
for i in range(0,len(list(G_network.edges))):
    Edges.append(list(G_network.get_edge_data(list(G_network.edges)[i][0],list(G_network.edges)[i][1]).values())[1])

#dictionary from Edge to Node
Edge_to_Node = {}
for i in range(len(Edges)):
    for j in range(0,len(list(G_network.edges))):
        if list(G_network.get_edge_data(list(G_network.edges)[j][0],list(G_network.edges)[j][1]).values())[1] == Edges[i]:
            node_list = list(G_network.edges)[j][0],list(G_network.edges)[j][1]
    Edge_to_Node[Edges[i]] = node_list
#dictionary from Node to Edge
Node_to_Edge = {entry:key for key,entry in Edge_to_Node.items()}

#dictionary of Edge and the Weight, Key: Edge
Edge_Weight = {}
for i in range(len(Edges)):
    for j in range(0,len(list(G_network.edges))):
        if list(G_network.get_edge_data(list(G_network.edges)[j][0],list(G_network.edges)[j][1]).values())[1] == Edges[i]:
            edge_weight = list(G_network.get_edge_data(list(G_network.edges)[j][0],list(G_network.edges)[j][1]).values())[0]
    Edge_Weight[Edges[i]] = edge_weight
#Dictionay of Node and the Weight, key: Nodes ('Cxx','Cxx')
Node_Weight = {}
for i in range(len(Edges)):
    for j in range(0,len(list(G_network.edges))):
        if list(G_network.get_edge_data(list(G_network.edges)[j][0],list(G_network.edges)[j][1]).values())[1] == Edges[i]:
            edge_weight = list(G_network.get_edge_data(list(G_network.edges)[j][0],list(G_network.edges)[j][1]).values())[0]
            Node_Weight[(list(G_network.edges)[j][0],list(G_network.edges)[j][1])] = edge_weight

#ksp
from itertools import islice
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

#some nodes cannot be target node
Nodes = list(G_network.nodes)
First_column = []
Second_column = []
for i in range(0,len(list(Node_to_Edge.keys()))):
    First_column.append(list(Node_to_Edge.keys())[i][0])
    Second_column.append(list(Node_to_Edge.keys())[i][1])

unique(First_column)
unique(Second_column)
not_target = []
for i in range(0,len(Nodes)):
    if Nodes[i] in unique(First_column) and Nodes[i] not in unique(Second_column):
        not_target.append(Nodes[i])

#randomly choose target !!cannot choose the one without in-degree!!
target = []
FogB_Nodes = ['C23','C24','C25','C26',
              'C40','C41','C42','C43',
              'C56','C57','C58','C59',
              'C73','C74','C75','C76',
              'C90','C91','C92','C93',
              'C108','C109','C110','C111']
available_Nodes = list(set(FogB_Nodes)-set(not_target))#list(set(Nodes) - set(not_target))

target.append(random.choice(available_Nodes))

#dictionary to make sure that there is path between the initial edge and the target node
Pair = {}
count = 0
for j in range(0,len(list(G_network.edges))):
    if nx.has_path(G_network,list(G_network.edges)[j][1],target[0]) == True:
        count = count +1
        #print(list(G_network.edges)[j][1],target[i])
        Pair[count] = (list(G_network.edges)[j][0],list(G_network.edges)[j][1],target[0])

#get the Sumo_path
Sumo_Path = []
node_path = []
Edges = []

for i in range (0,len(target)):#target 
    for j in range(0,len(list(G_network.edges))):
        if nx.has_path(G_network,list(G_network.edges)[j][1],target[i]) == True: #and list(G_network.get_edge_data(list(G_network.edges)[j][0],list(G_network.edges)[j][1]).values())[1] == '46522160#0':
            #print('true')
            for path in k_shortest_paths(G_network, list(G_network.edges)[j][1], target[i], 3, weight = 'weight'):
                
                #print(path)
                    
                path.insert(0,list(G_network.edges)[j][0])
#                 print(path)
                node_path.append(path)
                Path = []
#                 Weights = []
                for m in range(0,len(path)-1):
                    edges = list(G_network.get_edge_data(path[m],path[m+1]).values())[1]
#                     weights = list(G_network.get_edge_data(path[m],path[m+1]).values())[0]
#                     print(weights)
                    Path.append(edges)
#                     Weights.append(weights)
                
                Sumo_Path.append(Path)


FIRST_EDGES = []
for i in range(0,len(Sumo_Path)):
    FIRST_EDGES.append(Sumo_Path[i][0])


EDGES_DISTRIBUTION = unique(FIRST_EDGES)

Probability_EDGES_DISTRIBUTION = []
for i in range(0,len(EDGES_DISTRIBUTION)):
    count = 0
    for j in range(0,len(Sumo_Path)):
        if Sumo_Path[j][0] == EDGES_DISTRIBUTION[i]:
            count = count+1
    probability = 1/count
    Probability_EDGES_DISTRIBUTION.append(probability)

EDGES = []
for i in range(0,len(EDGES_DISTRIBUTION)):
    for j in range(0,len(Sumo_Path)):
        if Sumo_Path[j][0] == EDGES_DISTRIBUTION[i]:
            #print(EDGES_DISTRIBUTION[i])
            Edges = (Sumo_Path[j],Probability_EDGES_DISTRIBUTION[i])
            EDGES.append(Edges)
Edges = []
for i in range(0,len(list(G_network.edges))):
    Edges.append(list(G_network.get_edge_data(list(G_network.edges)[i][0],list(G_network.edges)[i][1]).values())[1])


#six fog nodes
FogA_Nodes = ['C18','C19','C20','C21','C22',
              'C35','C36','C37','C38','C39',
              'C51','C52','C53','C54','C55',
              'C68','C69','C70','C71','C72',
              'C85','C86','C87','C88','C89',
              'C103','C104','C105','C106','C107']  
FogB_Nodes = ['C23','C24','C25','C26',
              'C40','C41','C42','C43',
              'C56','C57','C58','C59',
              'C73','C74','C75','C76',
              'C90','C91','C92','C93',
              'C108','C109','C110','C111']
FogC_Nodes = ['C27','C28','C29',
              'C44','C45',
              'C60','C61','C62',
              'C77','C78','C79',
              'C94','C95','C96',
              'C112','C113']
FogD_Nodes = ['C117','C118','C119','C120','C121','C122',
              'C133','C134','C135','C136','C137',
              'C148','C149','C150','C151',
              'C159','C160','C161','C162']
FogE_Nodes = ['C123','C124','C125','C126',
              'C138','C139','C140','C141',
              'C152','C153','C154','C155',
              'C163','C164','C165','C166']
FogF_Nodes = ['C114','C115','C116',
              'C127','C128','C129','C130','C131','C132',
              'C142','C143','C144','C145','C146','C147',
              'C156','C157','C158',
              'C167','C168','C169','C170','C171']

#separate edges into different fog node area
Fog_Edges = []
FogA_Edges = []
for i in range(0,len(Node_to_Edge)):
    if list(Node_to_Edge)[i][0] in FogA_Nodes and list(Node_to_Edge)[i][1] in FogA_Nodes:
        FogA_Edges.append(Node_to_Edge[(list(Node_to_Edge)[i][0],list(Node_to_Edge)[i][1])])

FogB_Edges = ['gneE31','569345502#1','gneE169','gneE197','gneE39','195743336#1']
for i in range(0,len(Node_to_Edge)):
    if list(Node_to_Edge)[i][0] in FogB_Nodes and list(Node_to_Edge)[i][1] in FogB_Nodes:
        FogB_Edges.append(Node_to_Edge[(list(Node_to_Edge)[i][0],list(Node_to_Edge)[i][1])])

FogC_Edges = ['458166897#12','458180193#1','gneE178','gneE202','420906753','458180191#1',
             'gneE45','gneE47','gneE309']
for i in range(0,len(Node_to_Edge)):
    if list(Node_to_Edge)[i][0] in FogC_Nodes and list(Node_to_Edge)[i][1] in FogC_Nodes:
        FogC_Edges.append(Node_to_Edge[(list(Node_to_Edge)[i][0],list(Node_to_Edge)[i][1])])

FogD_Edges = ['gneE62','gneE319','gneE66','gneE65','-gneE65','gneE318','618990022#11','gneE317','499172074#0',
              'gneE316','gneE68']
for i in range(0,len(Node_to_Edge)):
    if list(Node_to_Edge)[i][0] in FogD_Nodes and list(Node_to_Edge)[i][1] in FogD_Nodes:
        FogD_Edges.append(Node_to_Edge[(list(Node_to_Edge)[i][0],list(Node_to_Edge)[i][1])])

FogE_Edges = ['569345537#4','gneE71','gneE315','gneE314','gneE67','gneE72','gneE313','gneE312',
             'gneE73','464516471#8','gneE116','gneE119','420907906#0','gneE125','196116976#5',
             'gneE70']
for i in range(0,len(Node_to_Edge)):
    if list(Node_to_Edge)[i][0] in FogE_Nodes and list(Node_to_Edge)[i][1] in FogE_Nodes:
        FogE_Edges.append(Node_to_Edge[(list(Node_to_Edge)[i][0],list(Node_to_Edge)[i][1])])

FogF_Edges = []
for i in range(0,len(Node_to_Edge)):
    if list(Node_to_Edge)[i][0] in FogF_Nodes and list(Node_to_Edge)[i][1] in FogF_Nodes:
        FogF_Edges.append(Node_to_Edge[(list(Node_to_Edge)[i][0],list(Node_to_Edge)[i][1])])


#Matrix of 6 fog nodes:
Row = 6
Column = Row
A_Matrix = np.eye(Row, Column)

A_Matrix[0][1] = 1
A_Matrix[1][0] = 1
A_Matrix[0][3] = 1
A_Matrix[3][0] = 1
A_Matrix[1][2] = 1
A_Matrix[2][1] = 1
A_Matrix[1][3] = 1
A_Matrix[3][1] = 1
A_Matrix[1][4] = 1
A_Matrix[4][1] = 1
A_Matrix[2][4] = 1
A_Matrix[4][2] = 1
A_Matrix[2][5] = 1
A_Matrix[5][2] = 1
A_Matrix[3][4] = 1
A_Matrix[4][3] = 1
A_Matrix[5][4] = 1
A_Matrix[4][5] = 1






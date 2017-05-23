import numpy as np
from copy import * 

def convert_to_graphs(graph): 
	G = {}
	for i in range(24):
		G[i] = []
	edges = []
	for i in range(24):
		for j in range(i+1,24):
			edges.append((i,j))
	for i in range(len(graph)):
		direction = graph[i]
		if direction == 0:
			continue
		e = edges[i]
		if direction == 1:
			father = e[0]
			child = e[1]
			G[father].append(child)
		elif direction == 2:
			father = e[1]
			child = e[0]
			G[father].append(child)
	return G

def Check(pos, G, V):
    if pos in V:
        return True
    nV = copy(V)
    nV.append(pos)
    for node in G[pos]:
        if Check(node, G, nV):
            return True
    return False
    
def Cycle(G):
    for node in G:
        if Check(node, G, []):
            return True
    return False
def remove_cycles(graphs_pool):
	Acyclic_graphs = []
	for G in graphs_pool:
		if Cycle(G):
			continue
		Acyclic_graphs.append(G)
	return Acyclic_graphs

def generate_graphs_with_edges(G_info,k):
	assert (G_info["remaining_spots"] >= k)
	assert (G_info["next_available"] < G_info["max_slots"])

	graph = G_info["graph"]
	Next = G_info["next_available"]

	if (k == 0) or (G_info["remaining_spots"] == 0):
		return [graph]

	New_G_0 = np.copy(graph)
	New_G_1 = np.copy(graph)
	New_G_2 = np.copy(graph)
	#No edge 
	New_G_0[Next] = 0
	#direction 1
	New_G_1[Next] = 1
	#direction 2
	New_G_2[Next] = 2


	if (Next == G_info["max_slots"]-1) and (G_info["remaining_spots"] == k):

		return [New_G_1, New_G_2]

	elif (Next == G_info["max_slots"]-1):

		if k==1:
			return [New_G_1, New_G_2]

		if k >1 :
		 	return []
		return [New_G_0,New_G_1, New_G_2]

	case1 = {"next_available": Next+1,
			 "graph": New_G_0,
			 "remaining_spots": G_info["remaining_spots"],
			 "max_slots":276}

	case2 = {"next_available":Next+1,
			 "graph": New_G_1,
			 "remaining_spots": G_info["remaining_spots"]-1,
			 "max_slots":276}

	case3 = {"next_available":Next+1,
			 "graph": New_G_2,
			 "remaining_spots": G_info["remaining_spots"]-1,
			 "max_slots":276}
	if 	G_info["remaining_spots"] == k:
		return generate_graphs_with_edges(case2,k-1) + generate_graphs_with_edges(case3,k-1)

	else:
		return generate_graphs_with_edges(case1,k) + \
			   generate_graphs_with_edges(case2,k-1) + \
		       generate_graphs_with_edges(case3,k-1)


def generate_all_possible_graphs(G):

	graphs_list = []
	for num_edges in range(0,276):
		
		G_info = {"next_available": 0, "graph":G, "remaining_spots":276, "max_slots":276}
		print "number of edges",num_edges
		# t = time.time()

		new_list = generate_graphs_with_edges(G_info,num_edges)
		print "new list with cycles",len(new_list)
		new_list = map(lambda x: convert_to_graphs(x),new_list)
		new_list = remove_cycles(new_list)
		print "new list without cycles",len(new_list)
		# print "time",time.time() - t

		graphs_list = graphs_list + new_list
	return graphs_list

G = np.zeros((276,1))
print len(generate_all_possible_graphs(G))

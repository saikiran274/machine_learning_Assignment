startNode="panji"
goalNode="nellore"

route_map = {
    'panji': [{'raichur': 457}, {'mangalore': 365}, {'bellari': 409}],
    'raichur': [{'panji': 457}, {'kurnool': 100}, {'tirupati': 453}],
    'mangalore': [{'panji': 365}, {'kozhikode': 233}, {'bangalore': 352}],
    'bellari': [{'panji': 409}, {'tirupati': 379}, {'bangalore': 311}],
    'tirupati': [{'raichur': 453}, {'kurnool': 340}, {'bellari': 379}, {'nellore': 136}, {'chennai': 153}],
    'kurnool': [{'raichur': 100}, {'tirupati': 340}, {'nellore': 325}],
    'kozhikode': [{'mangalore': 233}, {'bangalore': 356}],
    'bangalore': [{'bellari': 311}, {'mangalore': 352}, {'kozhikode': 356}, {'chennai': 346}],
    'nellore': [{'kurnool': 325}, {'tirupati': 136}, {'chennai': 175}],
    'chennai': [{'tirupati': 153}, {'nellore': 175}, {'bangalore': 346}]
}

location_coordinate_map= {
    'panji': [15.4909, 73.8278],
    'raichur': [16.2076, 77.3463],
    'mangalore': [12.9141, 74.8560],
    'bellari': [15.1394, 76.9214],
    'tirupati': [13.6288, 79.4192],
    'kurnool': [15.8281, 78.0373],
    'kozhikode': [11.2588, 75.7804],
    'bangalore': [12.9716, 77.5946],
    'nellore': [14.4426, 79.9865],
    'chennai': [13.0827, 80.2707]
}



import math
def cost_heuristicfunctionvalues_eachNode_ToDestination(detinationNodeName):
    heuristicValue_node_map = {}
    for key in route_map.keys():
        lat1 = location_coordinate_map[key][0]
        lon1 = location_coordinate_map[key][1]
        lat2 = location_coordinate_map[detinationNodeName][0]
        lon2 = location_coordinate_map[detinationNodeName][1]

        dis_lat = (lat2 - lat1) * math.pi / 180.0
        dis_lon = (lon2 - lon1) * math.pi / 180.0

        lat1 = lat1 * math.pi / 180.0
        lat2 = lat2 * math.pi / 180.0

        a = (pow(math.sin(dis_lat / 2), 2) +
             pow(math.sin(dis_lon / 2), 2) *
             math.cos(lat1) * math.cos(lat2));
        rad = 6371
        c = 2 * math.asin(math.sqrt(a))
        heuristic_cost = rad * c

        heuristicValue_node_map.update({key: heuristic_cost})

    return heuristicValue_node_map



heuristicValue_city_map = cost_heuristicfunctionvalues_eachNode_ToDestination(goalNode)
print(heuristicValue_city_map)


#Code Block 1
def distanceTONextNode_g(from_city, to_city, cur_dist=0):
    path_cost=0
    for i, d in enumerate(route_map[from_city]):
        city_name = list(d.keys())[0]
        if city_name == to_city:
            path_cost = list(d.values())[0]
            return path_cost + cur_dist
        
        
#F(n) = h(n)+g(n)
# h(n) from heuristicValue_city_map
# g(n) from distanceTONextNode_g

def calculateAStarFunctionCost(fromCity, toCity, totalCost):
    h_cost = heuristicValue_city_map[toCity]
    f_cost = totalCost + h_cost

    return f_cost



traversedNodeList = []
frontierNodeDict = {}

def nextNeighbourNode(node, currentDistOfNode=0):
    for i, d in enumerate(route_map[node]):
        nodeName = list(d.keys())[0]
        nodeCost = list(d.values())[0] + currentDistOfNode
        functionCost = calculateAStarFunctionCost(node, nodeName, nodeCost)
        if nodeName not in traversedNodeList:
            frontierNodeDict[nodeName] = functionCost
        traversedNodeList.append(node)
    
    
    
#Computation call
def nextNode():
    sortedFrontier = sorted(frontierNodeDict.items(), key=lambda x: (x[1], x[0]))
    print("sortedFrontier",sortedFrontier)
    nextNode = sortedFrontier.pop(0)
    frontierNodeDict.pop(nextNode[0])
    print("nextNode",nextNode)
    return nextNode

def isGoalNode(node):
    return node == goalNode

def searchForShortestPath(startNode):
    totalPathdist = 0
    currentNode = startNode
    currentDistOfNode = 0

    nextNeighbourNode(startNode, currentDistOfNode)

    while len(frontierNodeDict) != 0:
        next_node, _ = nextNode()
        if isGoalNode(next_node):
            totalPathdist = distanceTONextNode_g(currentNode, next_node, currentDistOfNode)
            traversedNodeList.append(next_node)
            break
        else:
            try:
                currentDistOfNode = distanceTONextNode_g(currentNode, next_node, currentDistOfNode)
                currentNode = next_node
                nextNeighbourNode(next_node, currentDistOfNode)
            except:
                print("next_node :",next_node)

    print("Search Complete")
    print(totalPathdist)
    return totalPathdist


searchForShortestPath(startNode)
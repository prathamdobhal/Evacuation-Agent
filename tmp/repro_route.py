
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "backend"))

from services.graph_service import GraphService

def test_routing():
    # Use absolute paths
    base = os.path.join(os.getcwd(), "backend", "data")
    svc = GraphService(
        graph_path=os.path.join(base, "sendai_graph.json"),
        camps_path=os.path.join(base, "camps.json")
    )
    
    # User coordinates in Natori (New Southern Expansion)
    user_lat, user_lng = 38.1724, 140.8914
    
    # Earthquake at 85% severity
    calamity = "earthquake"
    severity = 0.85
    
    print(f"Testing routing for {calamity} at {severity} severity...")
    result = svc.find_route(user_lat, user_lng, calamity, severity)
    
    if "error" in result:
        print(f"FAILED: {result['error']}")
        
        # Check if adjacency list is empty or start node is isolated
        start_node = svc.nearest_node(user_lat, user_lng)
        valid_camps = svc.get_valid_camps(calamity)
        camp_nodes = svc.get_camp_nodes(valid_camps)
        
        from services.astar import build_adjacency
        excluded = svc.get_excluded_nodes(calamity)
        adj = build_adjacency(svc.graph, calamity, severity, excluded)
        
        print(f"Start node: {start_node}")
        print(f"Camp nodes: {camp_nodes}")
        print(f"Neighbors of start node: {len(adj.get(start_node, []))}")
        
    else:
        print(f"SUCCESS: Found route of distance {result['distance_km']} km")

if __name__ == "__main__":
    test_routing()

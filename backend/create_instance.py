import argparse
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import os
import copy
import osmnx as ox
import folium
import logging
import csv
from shapely.geometry import Point
import logging
import hashlib
import time
from contextlib import contextmanager
import warnings
from datetime import datetime

# Suppress pandas deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Setup logging - configure once
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove default handlers to avoid duplicates
if logger.handlers:
    logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Global variable for timing log file (will be set in main())
timing_log_file = None

# Configure OSMnx to use a shared cache directory in the project
# This ensures both direct execution and subprocess execution use the same cache
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, 'cache')
os.makedirs(cache_dir, exist_ok=True)
ox.settings.use_cache = True
ox.settings.cache_folder = cache_dir
print(f"OSMnx cache configured: {cache_dir}")
print(f"OSMnx use_cache: {ox.settings.use_cache}")

# Configure osmnx to use drivable roads and to retain all information
ox.__version__

# Timing context manager
@contextmanager
def timer(step_name):
    """Context manager to time execution of code blocks"""
    start_time = time.time()
    msg_start = f"STARTING: {step_name}"
    print(f"\n{'='*60}")
    print(f"⏱️  {msg_start}")
    print(f"{'='*60}")
    
    # Log to file
    if timing_log_file:
        timing_log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg_start}\n")
        timing_log_file.flush()
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        msg_complete = f"COMPLETED: {step_name}"
        msg_time = f"TIME: {minutes}m {seconds:.2f}s ({elapsed:.2f} seconds)"
        
        print(f"\n{'='*60}")
        print(f"✅ {msg_complete}")
        print(f"⏱️  {msg_time}")
        print(f"{'='*60}\n")
        
        # Log to file
        if timing_log_file:
            timing_log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg_complete} - {msg_time}\n")
            timing_log_file.flush()

# Set a fixed random seed for reproducibility
#random_seed = 42
#random.seed(random_seed)
#np.random.seed(random_seed)


class Instance:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_index = {}
        self.G = nx.Graph()
        self.road_speed_info = {}
        # New: store vehicle configuration here
        self.vehicle_config = {}
        self.tour_time_limit = 0
    def add_node(self, node_id, node_type, x, y, demand=0, service_time=0):
        self.nodes.append({'id': node_id, 'type': node_type, 'x': x, 'y': y, 'demand': demand, 'service_time': service_time})
        self.G.add_node(node_id, type=node_type, x=x, y=y, demand=demand, service_time=service_time)

    def add_edge(self, from_node, to_node, distance, road_type='unknown', min_speed=None, max_speed=None, road_label=None):
        distance_rounded = round(distance)
        if road_label is None:  # If road_label is not provided, get it based on road_type
            road_label = self.road_labels.get(road_type, 0)  # Use 0 or another default for unknown road types
        self.edges.append({
            'from': from_node,
            'to': to_node,
            'distance': distance_rounded,
            'road_type': road_type,
            'min_speed': min_speed,
            'max_speed': max_speed,
            'road_label': road_label
        })
        self.G.add_edge(from_node, to_node, distance=distance_rounded, road_type=road_type,
                        min_speed=min_speed, max_speed=max_speed, road_label=road_label)

    def generate_matrices(self):
        # Initialize matrices
        num_nodes = len(self.nodes)
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        self.distance_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

        # Create a mapping from node ID to index in the matrix
        node_id_to_index = {node['id']: idx for idx, node in enumerate(self.nodes)}

        # Fill the matrices
        for edge in self.edges:
            from_index = node_id_to_index[edge['from']]
            to_index = node_id_to_index[edge['to']]
            distance = edge['distance']
            road_label = edge['road_label']  # Get the road label for the edge

            # Update adjacency matrix (assuming directed graph)
            self.adj_matrix[from_index, to_index] = road_label

            # Update distance matrix (assuming directed graph)
            self.distance_matrix[from_index, to_index] = distance

    def assign_road_labels(self, G):
        # Identify all unique road types in the graph
        road_types = set()
        for _, _, data in G.edges(data=True):
            types = data.get('highway', [])
            if not isinstance(types, list):
                types = [types]
            road_types.update(types)

        # Assign an integer label to each road type
        self.road_labels = {road_type: idx+1 for idx, road_type in enumerate(sorted(road_types))}

    def populate_from_graph(self, G, depots, customers, charging_station_nodes, crossings,
                            demand_option, demand_constant, demand_range_min, demand_range_max,
                            service_time_option, service_time_constant, service_time_min, service_time_max):
        """
        Now uses user-defined logic for demands and service times.
        """

        road_speeds = {}
        road_speed_info = {}

        # Collect speed data from edges
        for _, _, data in G.edges(data=True):
            road_type = data.get('highway', 'unknown')
            if isinstance(road_type, list):
                road_type = road_type[0]  # Simplify for multiple types
            
            max_speed_str = data.get('maxspeed', None)
            if max_speed_str:
                max_speed = parse_speed(max_speed_str)
                if road_type not in road_speeds or max_speed > road_speeds[road_type]['max']:
                    road_speeds[road_type] = road_speeds.get(road_type, {})
                    road_speeds[road_type]['max'] = max_speed

        # Default speeds as before
        default_speeds = {
            'motorway': (70, 130),
            'primary': (50, 100),
            'secondary': (30, 90),
            'tertiary': (30, 70),
            'residential': (20, 50),
            'unknown': (20, 50)
        }

        node_counter = 0
        if depots:
            self.add_node(depots[0], 'd', G.nodes[depots[0]]['x'], G.nodes[depots[0]]['y'])
            self.node_index[depots[0]] = node_counter
            node_counter += 1

        # Helper: get demand and service time based on the user’s chosen options
        def get_demand():
            if demand_option == "constant":
                return demand_constant
            elif demand_option == "random":
                return random.randint(demand_range_min, demand_range_max)
            else:
                # default old logic: random between 50 and 240 in steps of 10
                return random.choice(range(50, 241, 10))

        def get_service_time():
            if service_time_option == "constant":
                return service_time_constant
            elif service_time_option == "random":
                return round(random.uniform(service_time_min, service_time_max), 3)
            else:
                # default old logic: 0.5 hours
                return 0.5

        # Add customers
        for node_id in customers:
            demand_value = get_demand()
            service_time_value = get_service_time()
            self.add_node(node_id, 'c', G.nodes[node_id]['x'], G.nodes[node_id]['y'],
                          demand=demand_value, service_time=service_time_value)
            self.node_index[node_id] = node_counter
            node_counter += 1

        # Add charging stations
        for node_id in charging_station_nodes:
            self.add_node(node_id, 'f', G.nodes[node_id]['x'], G.nodes[node_id]['y'])
            self.node_index[node_id] = node_counter
            node_counter += 1

        # Add crossings
        for node_id in crossings:
            self.add_node(node_id, 'a', G.nodes[node_id]['x'], G.nodes[node_id]['y'])
            self.node_index[node_id] = node_counter
            node_counter += 1
        
        # Step 2: Apply speeds to edges
        for u, v, data in G.edges(data=True):
            if u in self.node_index and v in self.node_index:
                road_type = data.get('highway', 'unknown')
                if isinstance(road_type, list):
                    road_type = road_type[0]

                max_speed = road_speeds.get(road_type, {}).get('max', default_speeds.get(road_type, (20, 50))[1])
                min_speed = default_speeds.get(road_type, (20, 50))[0]

                self.add_edge(u, v, data['length'], road_type=road_type,
                              min_speed=min_speed, max_speed=max_speed)
                
                if road_type not in road_speed_info:
                    road_speed_info[road_type] = {'min_speed': min_speed, 'max_speed': max_speed}
                else:
                    road_speed_info[road_type]['min_speed'] = min(road_speed_info[road_type]['min_speed'], min_speed)
                    road_speed_info[road_type]['max_speed'] = max(road_speed_info[road_type]['max_speed'], max_speed)
            
        self.road_speed_info = road_speed_info
        self.generate_matrices()

    def duplicate_nodes_as_junctions(self):
        """
        Duplicates customer and charging station nodes as road junctions.
        """
        current_max_id = max(node['id'] for node in self.nodes)
        new_nodes = []
        original_to_duplicate = {}

        for node in self.nodes:
            if node['type'] in ['d', 'c', 'f']:  # If the node is a depot, customer, or charging station
                new_id = current_max_id + 1
                new_label = f"{node['id']}c"
                new_node = {
                    'id': new_id,
                    'type': 'a',
                    'x': node['x'],
                    'y': node['y'],
                    'demand': 0,
                    'service_time': 0,
                    'node_label': new_label
                }
                new_nodes.append(new_node)
                original_to_duplicate[node['id']] = new_id
                current_max_id += 1

        # Add new nodes
        for new_node in new_nodes:
            self.add_node(new_node['id'], new_node['type'], new_node['x'], new_node['y'])

        print("ORIGINAL TO DUPLICATE:")
        print(original_to_duplicate)

        new_edges = []
        for edge in self.edges:
            from_original = edge['from']
            to_original = edge['to']
            
            from_duplicate = original_to_duplicate.get(from_original, None)
            to_duplicate = original_to_duplicate.get(to_original, None)

            if from_duplicate:
                new_edges.append((from_duplicate, to_original, edge['distance'],
                                  edge['road_type'], edge['min_speed'], edge['max_speed'], edge['road_label']))

            if to_duplicate:
                new_edges.append((from_original, to_duplicate, edge['distance'],
                                  edge['road_type'], edge['min_speed'], edge['max_speed'], edge['road_label']))

            if from_duplicate and to_duplicate:
                new_edges.append((from_duplicate, to_duplicate, edge['distance'],
                                  edge['road_type'], edge['min_speed'], edge['max_speed'], edge['road_label']))

        for new_edge in new_edges:
            self.add_edge(*new_edge)

    def save_dataset(self, file_path):
        self.generate_matrices()

        nodes_df = pd.DataFrame(self.nodes)
        nodes_df['custom_id'] = nodes_df.index
        
        nodes_df.rename(columns={'id': 'node_label'}, inplace=True)
        
        cols = ['custom_id', 'node_label'] + [col for col in nodes_df.columns if col not in ['custom_id', 'node_label']]
        nodes_df = nodes_df[cols]
        
        nodes_df.rename(columns={'custom_id': 'id'}, inplace=True)

        label_to_custom_id = nodes_df.set_index('node_label')['id'].to_dict()

        edge_df = pd.DataFrame(self.edges)

        edge_df['from'] = edge_df['from'].map(label_to_custom_id)
        edge_df['to'] = edge_df['to'].map(label_to_custom_id)

        def calculate_widths(df):
            col_widths = {col: max(len(col), df[col].astype(str).map(len).max()) for col in df.columns}
            return col_widths

        def write_fixed_width(df, file):
            col_widths = calculate_widths(df)
            format_str = ' '.join([f"{{:<{width}}}" for width in col_widths.values()]) + '\n'
            header = format_str.format(*df.columns)
            file.write(header)
            
            for _, row in df.iterrows():
                row_str = format_str.format(*row.astype(str))
                file.write(row_str)

        with open(file_path, 'w', newline='') as f:
            f.write("# Nodes\n")
            write_fixed_width(nodes_df, f)

            f.write("\n# Edges\n")
            write_fixed_width(edge_df, f)

            # -----------------------
            # NEW SECTION: Vehicle Configurations
            # -----------------------
            f.write("\n# Vehicle Configurations\n")
            # Pull each config from self.vehicle_config (defaulting to "N/A" if missing)
            battery = self.vehicle_config.get('BatteryCapacity', 'N/A')
            load_cap = self.vehicle_config.get('LoadCapacity', 'N/A')
            energy = self.vehicle_config.get('ChargingRate', 'N/A')
            time_limit = self.tour_time_limit

            f.write(f"BatteryCapacity (kWh): {battery}\n")
            f.write(f"LoadCapacity (Kg): {load_cap}\n")
            f.write(f"Charging Rate: {energy}\n")
            f.write(f"\nTime Limit: {time_limit}\n")

    def generate_summary(self):
        summary = []
        summary.append("Summary of the Created Dataset:\n")
        summary.append(f"Total number of depots: {sum(1 for node in self.nodes if node['type'] == 'd')}")
        summary.append(f"Total number of customers: {sum(1 for node in self.nodes if node['type'] == 'c')}")
        summary.append(f"Total number of charging stations: {sum(1 for node in self.nodes if node['type'] == 'f')}")
        summary.append(f"Total number of road crossings: {sum(1 for node in self.nodes if node['type'] == 'a')}")
        summary.append(f"Total number of edges (road segments): {len(self.edges)}")
        
        road_types_count = {}
        for edge in self.edges:
            road_type = edge['road_type']
            if road_type in road_types_count:
                road_types_count[road_type] += 1
            else:
                road_types_count[road_type] = 1

        summary.append("\nNumber of road segments in each road type:")
        for road_type, count in road_types_count.items():
            summary.append(f"  {road_type}: {count}")

        # Add vehicle configuration info if any
        if hasattr(self, 'vehicle_config') and self.vehicle_config:
            summary.append("\nVehicle Configuration:")
            for k, v in self.vehicle_config.items():
                summary.append(f"  {k}: {v}")

        return "\n".join(summary)

    def save_summary(self, summary_file_path):
        summary = self.generate_summary()
        with open(summary_file_path, 'w') as f:
            f.write(summary)

def parse_speed(speed_str):
    """
    Safely parse an OSM maxspeed string, converting it to an integer km/h.
    If multiple speeds (like "50|50|30"), pick one (the first).
    If parsing fails entirely, default to 50 km/h.
    """
    if isinstance(speed_str, list):
        speed_str = speed_str[0]

    if "|" in speed_str:
        speed_str = speed_str.split("|")[0].strip()

    try:
        if "mph" in speed_str.lower():
            mph_value = float(speed_str.lower().replace("mph", "").strip())
            return int(mph_value * 1.60934)
        elif "km/h" in speed_str.lower():
            kmh_value = float(speed_str.lower().replace("km/h", "").strip())
            return int(kmh_value)
        else:
            return int(speed_str)
    except ValueError:
        logging.warning(f"Could not parse speed from '{speed_str}'. Defaulting to 50 km/h.")
        return 50

def extract_network(area=None, coordinates=None, network_type='drive', road_types=None):
    if not area and not coordinates:
        raise ValueError("Either area name or coordinates must be provided.")
    
    custom_filter = build_custom_filter(road_types) if road_types else None

    if area:
        print(f"📡 Fetching network for area: {area}")
        start = time.time()
        G = ox.graph_from_place(area, network_type=network_type, custom_filter=custom_filter)
        elapsed = time.time() - start
        print(f"   ✅ Network fetched: {len(G.nodes)} nodes, {len(G.edges)} edges in {elapsed:.2f}s")
    elif coordinates:
        print(f"📡 Fetching network for coordinates...")
        start = time.time()
        if len(coordinates) == 4:
            north, south, east, west = coordinates
            G = ox.graph_from_bbox(north, south, east, west, network_type=network_type, custom_filter=custom_filter)
        elif len(coordinates) > 4 and len(coordinates) % 2 == 0:
            from shapely.geometry import Polygon
            # IMPORTANT: Swap the order to (lng, lat) for Shapely
            points = [(coordinates[i+1], coordinates[i]) for i in range(0, len(coordinates), 2)]
            polygon = Polygon(points)
            G = ox.graph_from_polygon(polygon, network_type=network_type, custom_filter=custom_filter)
        else:
            raise ValueError("Invalid coordinates provided. Provide either 4 values (bounding box) or an even number greater than 4 (polygon).")
        elapsed = time.time() - start
        print(f"   ✅ Network fetched: {len(G.nodes)} nodes, {len(G.edges)} edges in {elapsed:.2f}s")
    return G



def build_custom_filter(road_types):
    filter_str = "['highway'~'" + "|".join(road_types) + "']" if road_types else None
    return filter_str

def extract_pois(area_name, tags):
    try:
        pois = ox.geometries_from_place(area_name, tags=tags)
        if pois.empty:
            logging.warning(f"No POIs found for {tags} in {area_name}")
        return pois
    except ox._errors.InsufficientResponseError as e:
        logging.warning(f"Error fetching POIs for {tags} in {area_name}: {e}")
        return pd.DataFrame()
    except (TypeError, ValueError) as e:
        # Handle geocoding errors gracefully - these are expected for some place names
        error_msg = str(e)
        if "could not geocode" in error_msg.lower() or "geometry" in error_msg.lower() or "polygon" in error_msg.lower():
            # This is expected for some place names - silently return empty DataFrame
            # The script will use random nodes from the graph instead
            logging.info(f"Could not geocode '{area_name}' for POI extraction with tags {tags}. Will use random nodes from graph.")
            return pd.DataFrame()
        else:
            # Re-raise if it's a different TypeError/ValueError
            logging.warning(f"Unexpected error fetching POIs for {tags} in {area_name}: {e}")
            return pd.DataFrame()
    except Exception as e:
        logging.warning(f"Unexpected error fetching POIs for {tags} in {area_name}: {e}")
        return pd.DataFrame()

def extract_road_crossings(G):
    crossings = [node for node, degree in G.degree() if degree >= 2]
    return crossings

def remove_self_loops(G):
    self_loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(self_loops)
    print(f"Removed {len(self_loops)} self-loops.")

def remove_dead_end_crossings(G, depots, customers, charging_stations, crossings):
    removed_count = 0

    def is_dead_end_crossing(node):
        connected_nodes_out = {v for _, v in G.out_edges(node)}
        connected_nodes_in = {u for u, _ in G.in_edges(node)}
        all_connected_nodes = connected_nodes_out.union(connected_nodes_in)
        return len(all_connected_nodes) == 1

    while True:
        potential_dead_ends = [node for node in G.nodes if node not in depots and node not in customers and node not in charging_stations]
        dead_end_crossings = [node for node in potential_dead_ends if is_dead_end_crossing(node)]
        if not dead_end_crossings:
            break

        for node in dead_end_crossings:
            neighbors = list(set(G.neighbors(node)) - {node})
            G.remove_node(node)
            if node in crossings:
                crossings.remove(node)
            removed_count += 1
            
            for neighbor in neighbors:
                if is_dead_end_crossing(neighbor) and neighbor not in depots and neighbor not in customers and neighbor not in charging_stations:
                    G.remove_node(neighbor)
                    if neighbor in crossings:
                        crossings.remove(neighbor)
                    removed_count += 1

    print(f"Removed {removed_count} dead-end crossing nodes.")

def simplify_network(G, depots, customers, charging_stations, crossings):
    print("Starting network simplification...")
    edge_lengths = {(u, v): data['length'] for u, v, data in G.edges(data=True) if 'length' in data}
    crossing_distances = {}
    G_current = copy.deepcopy(G)
    crossings_current = crossings[:]
    critical_nodes = set(depots + customers + charging_stations)

    for node in crossings_current:
        if node not in G_current.nodes:
            continue

        neighbors = list(G_current.neighbors(node))
        total_distance = 0
        valid_edges = 0
        for nbr in neighbors:
            edge_key = (node, nbr) if (node, nbr) in edge_lengths else (nbr, node)
            if edge_key in edge_lengths:
                distance = edge_lengths[edge_key]
                total_distance += distance
                valid_edges += 1
        if valid_edges > 0:
            avg_distance = total_distance / valid_edges
            crossing_distances[node] = avg_distance

    sorted_crossings = sorted(crossing_distances.items(), key=lambda x: x[1])

    for crossing, _ in sorted_crossings:
        if crossing not in G_current:
            continue

        G_previous = copy.deepcopy(G_current)
        crossings_previous = crossings_current[:]

        G_current.remove_node(crossing)
        if crossing in crossings_current:
            crossings_current.remove(crossing)

        if is_graph_connected(G_current):
            print(f"Removed crossing {crossing} successfully.")
        else:
            G_current = G_previous
            crossings_current = crossings_previous
            print(f"Restored crossing {crossing} due to connectivity issues.")

    return G_current, crossings_current

def are_critical_nodes_interconnected(G, critical_nodes):
    H = nx.DiGraph()
    for node in critical_nodes:
        for target in critical_nodes:
            if node != target and nx.has_path(G, node, target):
                H.add_edge(node, target)
    return nx.is_strongly_connected(H)

def is_graph_connected(subgraph):
    return nx.is_strongly_connected(subgraph)

def dfs_visit(G, node, visited):
    stack = [node]
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            stack.extend(set(G.neighbors(current)) - visited)

def is_strongly_connected(G):
    if len(G) == 0:
        return False

    start_node = next(iter(G.nodes))
    visited = set()
    dfs_visit(G, start_node, visited)
    if len(visited) != len(G):
        return False

    G_rev = G.reverse(copy=True)
    visited.clear()
    dfs_visit(G_rev, start_node, visited)
    if len(visited) != len(G):
        return False
    return True

def find_unreachable_nodes(G):
    if nx.is_strongly_connected(G):
        print("The graph is strongly connected.")
        return []

    scc = list(nx.strongly_connected_components(G))
    largest_scc = max(scc, key=len)
    unreachable = [node for component in scc if component != largest_scc for node in component]
    return unreachable

def categorize_nodes(G, num_depots, num_customers, num_charging_stations, area_name, customer_tag_value, bbox=None):
    """
    Categorizes nodes into depots, customers, charging stations, and actual road crossings, with logic to handle POI counts.
    """
    # List of keys to try in priority order
    possible_keys = ['shop', 'amenity', 'leisure', 'healthcare', 'landuse', 'building']


    # Define alternative tags for each category
    #customer_tags = {'shop': 'supermarket'}

    depot_tags_list = [
        {'landuse': 'industrial'},
        {'building': 'warehouse'},
        {'amenity': 'industrial'}
    ]
    charging_station_tags_list = [
        {'amenity': 'charging_station'},
        {'amenity': 'fuel'}
    ]

    #customer_tags = None
    customer_pois = pd.DataFrame()
    message = ""  # Message for frontend feedback

    for key in possible_keys:
        customer_tags = {key: customer_tag_value}
        print("CUSTOMER TAGS: ", customer_tags)
        start = time.time()
        if area_name:
            customer_pois = extract_pois(area_name, customer_tags)
        elif bbox:
            # If bbox has exactly 4 values, use them directly; otherwise compute min/max.
            if len(bbox) == 4:
                north, south, east, west = bbox
            else:
                # bbox is a tuple of coordinates representing the polygon vertices (lat, lon, lat, lon, ...)
                lats = bbox[0::2]
                lngs = bbox[1::2]
                north = max(lats)
                south = min(lats)
                east = max(lngs)
                west = min(lngs)
            try:
                customer_pois = ox.geometries_from_bbox(north, south, east, west, tags=customer_tags)
            except ox._errors.InsufficientResponseError:
                customer_pois = pd.DataFrame()
        elapsed = time.time() - start
        print(f"   POI fetch for {key}={customer_tag_value}: {elapsed:.2f}s, found {len(customer_pois)} POIs")
        if not customer_pois.empty:
            logging.info(f"Found {len(customer_pois)} customer POIs with '{key}={customer_tag_value}'")
            break
    #customer_pois = extract_pois(area_name, customer_tags)
    #Check how many customer POIs were found
    num_found = len(customer_pois)
    if num_found < num_customers:
        message = f"Only {num_found} locations found for the chosen customer tag'{customer_tag_value}'. The rest selected randomly."
        logging.warning(message)
    else:
        message = f"Successfully found {num_customers} locations for selected customer option '{customer_tag_value}'."
    # Fetch POIs for customers

    # Fetch POIs for depots, trying multiple tags
    depot_pois = pd.DataFrame()
    start = time.time()
    for depot_tags in depot_tags_list:
        depot_pois = extract_pois(area_name, depot_tags)
        if not depot_pois.empty:
            break
    elapsed = time.time() - start
    print(f"   Depot POI fetch: {elapsed:.2f}s, found {len(depot_pois)} POIs")

    # Fetch POIs for charging stations, trying multiple tags
    charging_station_pois = pd.DataFrame()
    start = time.time()
    for charging_station_tags in charging_station_tags_list:
        charging_station_pois = extract_pois(area_name, charging_station_tags)
        if not charging_station_pois.empty:
            break
    elapsed = time.time() - start
    print(f"   Charging station POI fetch: {elapsed:.2f}s, found {len(charging_station_pois)} POIs")

    logging.info(f"Number of customer POIs: {len(customer_pois)}")
    logging.info(f"Number of depot POIs: {len(depot_pois)}")
    logging.info(f"Number of charging station POIs: {len(charging_station_pois)}")

    # Initialize sets to store node IDs
    depot_nodes = set()
    customer_nodes = set()
    charging_station_nodes = set()
    all_nodes = list(G.nodes())

    # Function to add nodes while ensuring no overlap
    def add_unique_nodes(nodes_set, pois, num_required, used_nodes, node_type_name="nodes"):
        # First, try to use POIs
        for _, row in pois.iterrows():
            if len(nodes_set) >= num_required:
                break
            geometry = row.geometry
            if isinstance(geometry, Point):
                x, y = geometry.x, geometry.y
            elif geometry.geom_type == 'Polygon':
                x, y = geometry.centroid.x, geometry.centroid.y
            else:
                logging.warning(f"Unhandled geometry type: {geometry.geom_type}")
                continue

            nearest_node = ox.distance.nearest_nodes(G, x, y)
            if nearest_node not in used_nodes:
                nodes_set.add(nearest_node)
                used_nodes.add(nearest_node)

        # Check if we have enough available nodes before trying to fill
        available_nodes_count = len(all_nodes) - len(used_nodes)
        remaining_needed = num_required - len(nodes_set)
        
        if remaining_needed > available_nodes_count:
            raise ValueError(
                f"⚠️  INSUFFICIENT NODES ERROR: Cannot assign {num_required} {node_type_name}. "
                f"Only {available_nodes_count} nodes available in the network, but still need {remaining_needed} more. "
                f"Total graph nodes: {len(all_nodes)}, already used: {len(used_nodes)}. "
                f"Please either:\n"
                f"  1. Decrease the number of requested {node_type_name}, or\n"
                f"  2. Increase the area size, or\n"
                f"  3. Include more road types in the network extraction."
            )
        
        # Add safety counter to prevent infinite loop
        max_attempts = len(all_nodes) * 10  # Reasonable upper bound
        attempts = 0
        
        # Fill remaining slots with random nodes from the graph
        while len(nodes_set) < num_required:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"⚠️  ASSIGNMENT ERROR: Failed to assign {num_required} unique {node_type_name} after {max_attempts} attempts. "
                    f"Available nodes may be insufficient. "
                    f"Assigned: {len(nodes_set)}/{num_required}, Available: {available_nodes_count}, Used: {len(used_nodes)}"
                )
            candidate = random.choice(all_nodes)
            if candidate not in used_nodes:
                nodes_set.add(candidate)
                used_nodes.add(candidate)
            attempts += 1

    # Initialize used nodes set
    used_nodes = set()

    # Add unique depot nodes
    add_unique_nodes(depot_nodes, depot_pois, num_depots, used_nodes, "depots")

    # Add unique customer nodes
    add_unique_nodes(customer_nodes, customer_pois, num_customers, used_nodes, "customers")

    # Add unique charging station nodes
    add_unique_nodes(charging_station_nodes, charging_station_pois, num_charging_stations, used_nodes, "charging stations")

    # Convert sets to lists for consistency
    depot_nodes = list(depot_nodes)
    customer_nodes = list(customer_nodes)
    charging_station_nodes = list(charging_station_nodes)

    # Identify road crossings
    crossings = extract_road_crossings(G)

    # Exclude depots, customers, and charging stations from crossings
    crossings = [node for node in crossings if node not in used_nodes]

    return depot_nodes, customer_nodes, charging_station_nodes, crossings, len(crossings), message  


def assign_edge_attributes(G):
    """
    Assigns attributes to edges, such as speed limits based on road types.
    Also, categorizes edges based on speed for visualization.
    """
    for _, _, _, data in G.edges(data=True, keys=True):
        if 'highway' in data:
            if data['highway'] == 'motorway':
                data['speed_limit'] = 100  # km/h
                data['color'] = 'red'
            else:
                data['speed_limit'] = 50  # km/h
                data['color'] = 'yellow'
        else:
            data['speed_limit'] = 30  # km/h, assume a default value for unspecified
            data['color'] = 'blue'
        data['travel_time'] = data['length'] / data['speed_limit']  # Simplified

import folium
from folium.plugins import PolyLineTextPath

#def visualize_network(G, depots, customers, charging_stations, crossings, file_path):
def visualize_network(G, depots, customers, charging_stations, crossings, file_path, show_arrows=False):

    # Create a folium map centered on the mean coordinates of the nodes
    # mean_lat = np.mean([G.nodes[node]['y'] for node in G.nodes()])
    # mean_lon = np.mean([G.nodes[node]['x'] for node in G.nodes()])
    # m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)

    # 1) Compute bounding box from your final G
    latitudes = [G.nodes[n]['y'] for n in G.nodes()]
    longitudes = [G.nodes[n]['x'] for n in G.nodes()]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    
    # 2) Create the Folium map centered on the bounding box
    center_lat, center_lon = (min_lat + max_lat)/2, (min_lon + max_lon)/2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # 3) Optionally force Folium to fit to this bounding box, so it doesn’t start too zoomed out:
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])


    # Road type to color mapping
    road_color_map = {
        'motorway': 'red',
        'motorway_link': 'darkred',
        'trunk': 'orange',
        'trunk_link': 'darkorange',
        'primary': 'yellow',
        'primary_link': 'gold',
        'secondary': 'green',
        'secondary_link': 'darkgreen',
        'tertiary': 'blue',
        'tertiary_link': 'darkblue',
        'residential': 'purple',
        'service': 'gray',
        'unclassified': 'brown',
        'road': 'pink',
        'unknown': 'black'  # default for unspecified or unknown types
    }

    # Add edges to the map
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    for index, row in edges.iterrows():
        geom = row['geometry']
        road_type = row['highway'] if 'highway' in row else 'unknown'
        road_type = road_type[0] if isinstance(road_type, list) else road_type
        color = road_color_map.get(road_type, 'black')

        line_points = [(lat, lon) for lon, lat in geom.coords]
        polyline = folium.PolyLine(line_points, color=color, weight=2.5, opacity=1)
        polyline.add_to(m)

        # Add arrows to the edges
        # folium.plugins.PolyLineTextPath(
        #     polyline,
        #     #text='       ➔',  # Unicode arrow symbol
        #     text='     ➔',  # Unicode arrow symbol
        #     #repeat=True,
        #     offset=10,
        #     attributes={'font-weight': 'bold', 'font-size': '12'}
        # ).add_to(m)

        # Only add arrows if show_arrows is True:
        if show_arrows:
            folium.plugins.PolyLineTextPath(
                polyline,
                text='     ➔',
                offset=10,
                attributes={'font-weight': 'bold', 'font-size': '12'}
            ).add_to(m)

    # Define node type based on which list they belong to
    def get_node_type(node):
        if node in depots:
            return "Depot"
        elif node in customers:
            return "Customer"
        elif node in charging_stations:
            return "Charging Station"
        elif node in crossings:
            return "Crossing"
        return "Unknown"

    # Add nodes to the map with custom markers
    for node in G.nodes():
        lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
        node_type = get_node_type(node)
        tooltip = f"Node ID: {node}, Type: {node_type}"  # Tooltip shows both ID and type
        popup = folium.Popup(tooltip, parse_html=True)
        if node_type == "Depot":
            folium.CircleMarker(location=[lat, lon], radius=10, color='red', fill=True, fill_color='red', fill_opacity=1.0, popup=popup, tooltip=tooltip).add_to(m)
        elif node_type == "Customer":
            folium.CircleMarker(location=[lat, lon], radius=8, color='green', fill=True, fill_color='green', fill_opacity=1.0, popup=popup, tooltip=tooltip).add_to(m)
        elif node_type == "Charging Station":
            folium.RegularPolygonMarker(location=[lat, lon], number_of_sides=3, radius=8, color='blue', fill=True, fill_color='blue', fill_opacity=1.0, popup=popup, tooltip=tooltip).add_to(m)
        elif node_type == "Crossing":
            folium.RegularPolygonMarker(location=[lat, lon], number_of_sides=4, radius=6, rotation=45, color='orange', fill=True, fill_color='orange', fill_opacity=1.0, popup=popup, tooltip=tooltip).add_to(m)

    # Save the map to the specified file path
    m.save(file_path)
    print(f"Visualization saved to {file_path}")



def extract_largest_scc(G):
    """
    Extracts the largest strongly connected component of the graph.
    """
    if nx.is_strongly_connected(G):
        print("The graph is already strongly connected.")
        return G
    else:
        # Get all strongly connected components, return the largest one
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        # Subgraph the original graph to include only nodes from the largest SCC
        G_scc = G.subgraph(largest_scc).copy()
        print(f"Extracted the largest SCC with {len(G_scc.nodes)} nodes and {len(G_scc.edges)} edges.")
        return G_scc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--area_name', type=str, help='Name of the area to extract the network from')
    parser.add_argument('--coordinates', type=str, help='Bounding box coordinates (north,south,east,west)')
    parser.add_argument('--instance_name', type=str, default='',
                    help='Custom name for the instance. If provided, this name will be used to name the output files.')
    parser.add_argument('--show_arrows', action='store_true', help='Display arrows on map edges')
    parser.add_argument('--num_depots', type=int, default=1, help='Number of depots')
    parser.add_argument('--num_customers', type=int, default=25, help='Number of customers')
    parser.add_argument('--num_cs', type=int, default=10, help='Number of charging stations')
    parser.add_argument('--road_types', type=str, default='motorway,trunk,primary,secondary,tertiary, service, unclassified',
                        help='Comma-separated list of road types to include in the network extraction')
    parser.add_argument('--customer_tag', type=str, default="supermarket",
                        help="Value for the customer POI tag (e.g., 'supermarket', 'pharmacy', 'cafe')")
    parser.add_argument('--simplify', action='store_true', help='Perform network simplification if specified')

    # 1) Demand & Service Time options
    parser.add_argument('--demand_option', type=str, default='default',
                        help='Options: default, constant, random')
    parser.add_argument('--demand_constant', type=int, default=100,
                        help='If demand_option=constant, use this value for all customers')
    parser.add_argument('--demand_range_min', type=int, default=50,
                        help='If demand_option=random, minimum demand')
    parser.add_argument('--demand_range_max', type=int, default=240,
                        help='If demand_option=random, maximum demand')

    parser.add_argument('--service_time_option', type=str, default='default',
                        help='Options: default, constant, random')
    parser.add_argument('--service_time_constant', type=float, default=0.5,
                        help='If service_time_option=constant, use this hours value for all customers')
    parser.add_argument('--service_time_min', type=float, default=0.1,
                        help='If service_time_option=random, minimum hours')
    parser.add_argument('--service_time_max', type=float, default=1.0,
                        help='If service_time_option=random, maximum hours')

    # 2) Vehicle Config
    parser.add_argument('--battery_capacity_option', type=str, default='default',
                        help='Options: default, user')
    parser.add_argument('--battery_capacity_value', type=float, default=100.0,
                        help='If battery_capacity_option=user, set this capacity (kWh, for example)')
    parser.add_argument('--load_capacity_option', type=str, default='default',
                        help='Options: default, user')
    parser.add_argument('--load_capacity_value', type=float, default=1000.0,
                        help='If load_capacity_option=user, set this load capacity (kg, for example)')
    parser.add_argument('--charging_rate_option', type=str, default='default',
                        help='Options: default, user')
    parser.add_argument('--charging_rate_value', type=float, default=0.2,
                        help='If charging_rate_option=user, set this charging rate (kWh/km, for example)')
    # 3) Time Limit for the Tour Completion
    parser.add_argument('--tour_time_limit', type=float, default=8.0, help='Time limit for the tour completion in hours')

    # 4) User Provided Seed for the Random Number Generator
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility. Default=42') 
    
    # 5) User Information (from web app)
    parser.add_argument('--user_ip', type=str, default='',
                        help='IP address of the user who generated the instance')
    parser.add_argument('--user_agent', type=str, default='',
                        help='User-Agent string from the browser')
    parser.add_argument('--user_referer', type=str, default='',
                        help='HTTP Referer header')

    args = parser.parse_args()

    # ------------- USE THE USER‐PROVIDED SEED -------------
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    tour_time_limit = args.tour_time_limit

    if args.coordinates:
        coordinates = tuple(map(float, args.coordinates.split(',')))
        area_name = None
        coordinates_str = args.coordinates.replace(',', '_')
        bbox = coordinates 
    else:
        coordinates = None
        area_name = args.area_name
        bbox = None


    if not area_name and not coordinates:
        raise ValueError("Either --area_name or --coordinates must be provided.")

    num_depots = args.num_depots
    num_customers = args.num_customers
    num_cs = args.num_cs
    road_types = args.road_types.split(',')
    customer_tag_value = args.customer_tag
    simplification_suffix = "simplified" if args.simplify else "not_simplified"

    datasets_base_dir = 'created_datasets'
    if not os.path.exists(datasets_base_dir):
        os.makedirs(datasets_base_dir)

    # Determine naming based on whether a custom instance name is provided.
    if args.instance_name:
        instance_name = args.instance_name.strip().replace(" ", "_")
        area_dir_name = instance_name
        base_file_name = instance_name
    else:
        if area_name:
            area_dir_name = f"{area_name.split(',')[0].replace(' ', '_')}_{num_customers}"
            base_file_name = f"{area_dir_name}_{simplification_suffix}"
        else:
            if len(coordinates_str) > 50:
                short_coords = "instance_poly"
            else:
                short_coords = coordinates_str
            area_dir_name = f"{short_coords}_{num_customers}"
            base_file_name = f"{short_coords}_{num_customers}_{simplification_suffix}"

    # Always define the base directory using area_dir_name.
    base_dir = os.path.join(datasets_base_dir, area_dir_name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Now build the file names.
    dataset_file_name = f"{base_file_name}.txt"
    dataset_summary_file_name = f"{base_file_name}_summary.txt"
    visualization_file_png = f"{base_file_name}.png"
    visualization_file = f"{base_file_name}.html"

    dataset_file_path = os.path.join(base_dir, dataset_file_name)
    dataset_summary_file_path = os.path.join(base_dir, dataset_summary_file_name)
    visualization_file_path = os.path.join(base_dir, visualization_file)

    # Create timing log file
    timing_log_path = os.path.join(base_dir, f"{base_file_name}_timing.log")
    global timing_log_file
    timing_log_file = open(timing_log_path, 'w', encoding='utf-8')
    
    # Write header
    timing_log_file.write(f"{'='*60}\n")
    timing_log_file.write(f"Instance Generation Timing Log\n")
    timing_log_file.write(f"{'='*60}\n\n")
    
    # Write user information (if provided from web app)
    if args.user_ip or args.user_agent:
        timing_log_file.write(f"--- USER INFORMATION (Web App Request) ---\n")
        if args.user_ip:
            timing_log_file.write(f"IP Address: {args.user_ip}\n")
        if args.user_agent:
            # Truncate user agent if too long
            user_agent = args.user_agent[:200] if len(args.user_agent) > 200 else args.user_agent
            timing_log_file.write(f"User-Agent: {user_agent}\n")
        if args.user_referer:
            timing_log_file.write(f"Referer: {args.user_referer}\n")
        timing_log_file.write(f"\n")
    
    # Write user configuration
    timing_log_file.write(f"--- USER CONFIGURATION ---\n")
    timing_log_file.write(f"Instance Name: {base_file_name}\n")
    timing_log_file.write(f"Area Selection Method: {'Address' if area_name else 'Coordinates'}\n")
    if area_name:
        timing_log_file.write(f"Area: {area_name}\n")
    else:
        timing_log_file.write(f"Coordinates: {coordinates_str}\n")
    
    timing_log_file.write(f"\n--- NODE CONFIGURATION ---\n")
    timing_log_file.write(f"Number of Depots: {num_depots}\n")
    timing_log_file.write(f"Number of Customers: {num_customers}\n")
    timing_log_file.write(f"Number of Charging Stations: {num_cs}\n")
    timing_log_file.write(f"Customer Choice: {customer_tag_value}\n")
    
    timing_log_file.write(f"\n--- DEMAND CONFIGURATION ---\n")
    timing_log_file.write(f"Demand Option: {args.demand_option}\n")
    if args.demand_option == 'constant':
        timing_log_file.write(f"Constant Demand: {args.demand_constant}\n")
    elif args.demand_option == 'random':
        timing_log_file.write(f"Demand Range: {args.demand_range_min} - {args.demand_range_max}\n")
    
    timing_log_file.write(f"\n--- SERVICE TIME CONFIGURATION ---\n")
    timing_log_file.write(f"Service Time Option: {args.service_time_option}\n")
    if args.service_time_option == 'constant':
        timing_log_file.write(f"Constant Service Time: {args.service_time_constant} hours\n")
    elif args.service_time_option == 'random':
        timing_log_file.write(f"Service Time Range: {args.service_time_min} - {args.service_time_max} hours\n")
    
    timing_log_file.write(f"\n--- VEHICLE CONFIGURATION ---\n")
    timing_log_file.write(f"Battery Capacity Option: {args.battery_capacity_option}\n")
    if args.battery_capacity_option == 'user':
        timing_log_file.write(f"Battery Capacity: {args.battery_capacity_value} kWh\n")
    timing_log_file.write(f"Load Capacity Option: {args.load_capacity_option}\n")
    if args.load_capacity_option == 'user':
        timing_log_file.write(f"Load Capacity: {args.load_capacity_value} kg\n")
    timing_log_file.write(f"Charging Rate Option: {args.charging_rate_option}\n")
    if args.charging_rate_option == 'user':
        timing_log_file.write(f"Charging Rate: {args.charging_rate_value}\n")
    
    timing_log_file.write(f"\n--- OTHER CONFIGURATION ---\n")
    timing_log_file.write(f"Tour Time Limit: {tour_time_limit} hours\n")
    timing_log_file.write(f"Random Seed: {args.random_seed}\n")
    timing_log_file.write(f"Network Simplification: {'Enabled' if args.simplify else 'Disabled'}\n")
    timing_log_file.write(f"Show Arrows: {'Enabled' if args.show_arrows else 'Disabled'}\n")
    timing_log_file.write(f"Road Types: {', '.join(road_types)}\n")
    
    timing_log_file.write(f"\n--- EXECUTION TIMELINE ---\n")
    timing_log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    timing_log_file.write(f"{'='*60}\n\n")
    timing_log_file.flush()

    try:
        print("AREA NAME:")
        print(area_name)

        # Add total timer at the start
        total_start = time.time()

        # TIMED: Extract network from OSM (THIS IS LIKELY THE SLOWEST PART)
        with timer("1. Extract Road Network from OSM"):
            G = extract_network(area=area_name, coordinates=coordinates, road_types=road_types)

        # TIMED: Extract largest strongly connected component
        with timer("2. Extract Largest Strongly Connected Component"):
            G = extract_largest_scc(G)

        # Early validation: Check if graph has enough nodes BEFORE categorizing
        total_nodes_needed = num_depots + num_customers + num_cs
        available_nodes = len(G.nodes())
        
        print(f"\n{'='*60}")
        print(f"📊 NETWORK VALIDATION")
        print(f"{'='*60}")
        print(f"Total nodes in extracted network: {available_nodes}")
        print(f"Total nodes requested: {total_nodes_needed}")
        print(f"  - Depots: {num_depots}")
        print(f"  - Customers: {num_customers}")
        print(f"  - Charging Stations: {num_cs}")
        print(f"{'='*60}\n")
        
        if available_nodes < total_nodes_needed:
            error_msg = (
                f"⚠️  INSUFFICIENT NETWORK NODES ERROR:\n"
                f"   Requested {total_nodes_needed} nodes (depots: {num_depots}, "
                f"customers: {num_customers}, charging stations: {num_cs}), "
                f"but extracted network only contains {available_nodes} nodes.\n\n"
                f"   SOLUTIONS:\n"
                f"   1. Decrease the number of requested nodes:\n"
                f"      - Reduce customers from {num_customers} to at most {available_nodes - num_depots - num_cs}\n"
                f"      - Or reduce other node types\n"
                f"   2. Increase the area size to extract more nodes\n"
                f"   3. Include more road types in the network extraction\n"
                f"   4. Try a different area with more road network coverage"
            )
            print(error_msg)
            if timing_log_file:
                timing_log_file.write(f"\n{'='*60}\n")
                timing_log_file.write(f"VALIDATION ERROR\n")
                timing_log_file.write(f"{error_msg}\n")
                timing_log_file.write(f"{'='*60}\n")
            raise ValueError(error_msg)
        
        print(f"✓ Network validation passed: {available_nodes} nodes available for {total_nodes_needed} requested nodes\n")

        # TIMED: Categorize nodes (find POIs, depots, customers, etc.)
        with timer("3. Categorize Nodes (Find POIs, Depots, Customers, Charging Stations)"):
            depots, customers, charging_stations, crossings, crossings_count_before, location_message = \
                categorize_nodes(G, num_depots, num_customers, num_cs, area_name, customer_tag_value, bbox)

        print(location_message)

        # Validate that we got the requested number of each type
        print(f"\n{'='*60}")
        print(f"📋 NODE ASSIGNMENT VALIDATION")
        print(f"{'='*60}")
        print(f"Depots: {len(depots)}/{num_depots} requested")
        print(f"Customers: {len(customers)}/{num_customers} requested")
        print(f"Charging Stations: {len(charging_stations)}/{num_cs} requested")
        print(f"Crossings: {len(crossings)} (auxiliary nodes)")
        print(f"{'='*60}\n")
        
        # Validate each node type separately
        validation_errors = []
        if len(depots) != num_depots:
            validation_errors.append(f"Depots: requested {num_depots} but assigned {len(depots)}")
        if len(customers) != num_customers:
            validation_errors.append(f"Customers: requested {num_customers} but assigned {len(customers)}")
        if len(charging_stations) != num_cs:
            validation_errors.append(f"Charging Stations: requested {num_cs} but assigned {len(charging_stations)}")
        
        if validation_errors:
            error_msg = (
                f"⚠️  NODE ASSIGNMENT ERROR:\n"
                f"   The following node types were not fully assigned:\n"
                f"   " + "\n   ".join(validation_errors) + "\n\n"
                f"   This should not happen if network validation passed. "
                f"Please report this issue."
            )
            print(error_msg)
            if timing_log_file:
                timing_log_file.write(f"\n{'='*60}\n")
                timing_log_file.write(f"ASSIGNMENT ERROR\n")
                timing_log_file.write(f"{error_msg}\n")
                timing_log_file.write(f"{'='*60}\n")
            raise ValueError(error_msg)
        
        print(f"✓ All node types successfully assigned!\n")

        print(f"Number of crossings before simplification: {crossings_count_before}")
        print(f"Number of nodes before simplification: {len(G.nodes)}")
        
        # TIMED: Assign edge attributes
        with timer("4. Assign Edge Attributes"):
            assign_edge_attributes(G)

        # TIMED: Simplify network (optional)
        if args.simplify:
            with timer("5. Simplify Network"):
                G, crossings = simplify_network(G, depots, customers, charging_stations, crossings)

        # TIMED: Remove dead-end crossings
        with timer("6. Remove Dead-End Crossings"):
            remove_dead_end_crossings(G, depots, customers, charging_stations, crossings)
        
        # TIMED: Re-assign edge attributes after simplification
        with timer("7. Re-assign Edge Attributes"):
            assign_edge_attributes(G)
        
        # TIMED: Visualize network (create HTML map)
        with timer("8. Generate HTML Visualization"):
            visualize_network(G, depots, customers, charging_stations, crossings, visualization_file_path, show_arrows=args.show_arrows)

        if nx.is_strongly_connected(G):
            print("The graph is already strongly connected.")
        if not G.is_directed():
            raise ValueError("Graph is not directed. Ensure that the graph extraction and processing is for a directed graph.")
        else:
            print("Graph is directed!")

        for node in G.nodes:
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            if in_degree == 0 or out_degree == 0:
                print(f"Node {node} might cause issues as it has only incoming or outgoing edges.")

        instance = Instance()

        instance.tour_time_limit = tour_time_limit

        # TIMED: Assign road labels
        with timer("9. Assign Road Labels"):
            instance.assign_road_labels(G)

        # TIMED: Populate instance from graph
        with timer("10. Populate Instance from Graph"):
            instance.populate_from_graph(
                G,
                depots,
                customers,
                charging_stations,
                crossings,
                demand_option=args.demand_option,
                demand_constant=args.demand_constant,
                demand_range_min=args.demand_range_min,
                demand_range_max=args.demand_range_max,
                service_time_option=args.service_time_option,
                service_time_constant=args.service_time_constant,
                service_time_min=args.service_time_min,
                service_time_max=args.service_time_max
            )

        # TIMED: Duplicate nodes as junctions
        with timer("11. Duplicate Nodes as Junctions"):
            instance.duplicate_nodes_as_junctions()

        # Vehicle config (fast, no timer needed)
        veh_config = {}
        # Battery capacity
        if args.battery_capacity_option == 'user':
            veh_config['BatteryCapacity'] = args.battery_capacity_value
        else:
            veh_config['BatteryCapacity'] = 100.0  # default

        # Load capacity
        if args.load_capacity_option == 'user':
            veh_config['LoadCapacity'] = args.load_capacity_value
        else:
            veh_config['LoadCapacity'] = 1000.0  # default

        # Charging Rate - FIX: Corrected typo from 'ChragingRate' to 'ChargingRate'
        if args.charging_rate_option == 'user':
            veh_config['ChargingRate'] = args.charging_rate_value
        else:
            veh_config['ChargingRate'] = 0.2  # default

        instance.vehicle_config = veh_config
        
        # TIMED: Save dataset
        with timer("12. Save Dataset to File"):
            instance.save_dataset(dataset_file_path)
        print("Instance Saved")

        # TIMED: Save summary
        with timer("13. Save Summary File"):
            instance.save_summary(dataset_summary_file_path)
        print("Summary Saved")
        
        # Print total time summary
        total_elapsed = time.time() - total_start
        total_minutes = int(total_elapsed // 60)
        total_seconds = total_elapsed % 60
        total_time_msg = f"TOTAL EXECUTION TIME: {total_minutes}m {total_seconds:.2f}s ({total_elapsed:.2f} seconds)"
        
        print(f"\n{'#'*60}")
        print(f"# ⏱️  {total_time_msg}")
        print(f"{'#'*60}\n")
        
        # Write total time to log file
        if timing_log_file:
            timing_log_file.write(f"\n{'='*60}\n")
            timing_log_file.write(f"{total_time_msg}\n")
            timing_log_file.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            timing_log_file.write(f"{'='*60}\n")
            timing_log_file.close()
            print(f"Timing log saved to: {timing_log_path}")
    
    except Exception as e:
        # Ensure log file is closed even on error
        if timing_log_file:
            timing_log_file.write(f"\n{'='*60}\n")
            timing_log_file.write(f"ERROR: {str(e)}\n")
            timing_log_file.write(f"Failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            timing_log_file.write(f"{'='*60}\n")
            timing_log_file.close()
        raise

if __name__ == "__main__":
    main()


"""
How to run the program:

### Running with an Area Name:
```bash
python create_instance.py --area_name "Sabadell, Spain" \
    --num_depots 1 --num_customers 25 --num_cs 10 --simplify \
    --road_types "motorway,trunk,primary,secondary,tertiary,service,unclassified" \
    --customer_tag "supermarket" \
    --demand_option constant --demand_constant 120 \
    --service_time_option random --service_time_min 0.1 --service_time_max 0.5 \
    --battery_capacity_option user --battery_capacity_value 200.0 \
    --load_capacity_option user --load_capacity_value 1500.0 \
    --charging_rate_option user --charging_rate_value 0.3 \
    --tour_time_limit 10.0 \
    --instance_name "Sabadell_Example" \
    --show_arrows

### Running with Coordinates:
python create_instance.py --coordinates "41.600,41.590,2.090,2.080" \
    --num_depots 2 --num_customers 30 --num_cs 5 --simplify \
    --road_types "motorway,primary,secondary,residential" \
    --customer_tag "pharmacy" \
    --demand_option random --demand_range_min 50 --demand_range_max 240 \
    --service_time_option constant --service_time_constant 0.3 \
    --battery_capacity_option default \
    --load_capacity_option user --load_capacity_value 1200.0 \
    --charging_rate_option default \
    --tour_time_limit 8.0 \
    --instance_name "Region_Example"

    
"""

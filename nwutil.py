import networkx as nx

class Request:
    def __init__(self, source: int, destination: int, holding_time: int):
        self.source = source
        self.destination = destination
        self.holding_time = holding_time

    def __repr__(self):
        return f"Request(src={self.source}, dst={self.destination}, hold={self.holding_time})"


class BaseLinkState:
    def __init__(self, u, v, capacity=20, utilization=0.0):
        if u > v: # sort by the node ID
            u, v = v, u
        self.endpoints = (u, v)
        self.capacity = capacity
        self.utilization = utilization

    def __repr__(self):
        return f"LinkState(capacity={self.capacity}, util={self.utilization})"

class LinkState(BaseLinkState):
    # Data structure to store the link state
    # Extended to track wavelength availability and lightpath information
    def __init__(self, u, v, capacity=20, utilization=0.0):
        super().__init__(u, v, capacity, utilization)
        self.wavelengths = [True] * capacity  # True = available, False = in use
        self.lightpaths = {}  # {wavelength_idx: (src, dst, expiry_time)} 


def generate_sample_graph(capacity=20):
    # Create the sample graph
    G = nx.Graph()

    G.add_nodes_from(range(9))

    # Define links: ring links + extra links
    links = [(n, (n + 1) % 9) for n in range(9)] + [(1, 7), (1, 5), (3, 6)]

    # Add edges with link state objects
    for u, v in links:
        G.add_edge(u, v, state=LinkState(u, v, capacity=capacity))
    return G


# Pre-defined paths between source-destination pairs
PATHS = {
    (0, 3): [
        [0, 1, 2, 3],       # P1
        [0, 8, 7, 6, 3]     # P2
    ],
    (0, 4): [
        [0, 1, 5, 4],       # P3
        [0, 8, 7, 6, 3, 4]  # P4
    ],
    (7, 3): [
        [7, 1, 2, 3],       # P5
        [7, 6, 3]           # P6
    ],
    (7, 4): [
        [7, 1, 5, 4],       # P7
        [7, 6, 3, 4]        # P8
    ]
}


def get_available_paths(source, destination):
    # Get available paths for a source-destination pair
    return PATHS.get((source, destination), [])


def find_available_wavelength(graph, path):
    # Find the first available wavelength on a path using first-fit allocation
    # Returns wavelength index or None if no wavelength is available
    if not path or len(path) < 2:
        return None

    # Get the link with minimum available wavelengths
    capacity = None
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u > v:
            u, v = v, u
        link_state = graph[path[i]][path[i + 1]]['state']
        if capacity is None:
            capacity = link_state.capacity

    # Check each wavelength from 0 to capacity-1
    for wavelength in range(capacity):
        available = True
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u > v:
                u, v = v, u
            link_state = graph[path[i]][path[i + 1]]['state']
            if not link_state.wavelengths[wavelength]:
                available = False
                break
        if available:
            return wavelength

    return None


def allocate_lightpath(graph, path, wavelength, request, current_time):
    # Allocate a lightpath on the given path with the specified wavelength
    expiry_time = current_time + request.holding_time

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u > v:
            u, v = v, u
        link_state = graph[path[i]][path[i + 1]]['state']
        link_state.wavelengths[wavelength] = False
        link_state.lightpaths[wavelength] = (request.source, request.destination, expiry_time)
        link_state.utilization = sum(1 for w in link_state.wavelengths if not w) / link_state.capacity


def release_expired_lightpaths(graph, current_time):
    # Release lightpaths that have expired at the current time
    for u, v, data in graph.edges(data=True):
        link_state = data['state']
        wavelengths_to_release = []

        for wavelength, (src, dst, expiry_time) in link_state.lightpaths.items():
            if expiry_time <= current_time:
                wavelengths_to_release.append(wavelength)

        for wavelength in wavelengths_to_release:
            link_state.wavelengths[wavelength] = True
            del link_state.lightpaths[wavelength]
            link_state.utilization = sum(1 for w in link_state.wavelengths if not w) / link_state.capacity


def get_network_state_vector(graph):
    # Convert the graph state into a feature vector for the DQN agent
    # Returns a flattened vector representing link utilizations
    state_vector = []

    # Sort edges for consistent ordering
    edges = sorted(graph.edges())

    for u, v in edges:
        link_state = graph[u][v]['state']
        state_vector.append(link_state.utilization)

    return state_vector
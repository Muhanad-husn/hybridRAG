
from arango import ArangoClient

# Initialize the ArangoDB client
client = ArangoClient()

# Connect to the '_system' database as the root user
sys_db = client.db('_system', username='root', password='your_password')

# Create a new database named 'graph_db' if it doesn't exist
if not sys_db.has_database('graph_db'):
    sys_db.create_database('graph_db')

# Connect to the 'graph_db' database
db = client.db('graph_db', username='root', password='your_password')

# Create or get the graph named 'my_graph'
if not db.has_graph('my_graph'):
    graph = db.create_graph('my_graph')
else:
    graph = db.graph('my_graph')

# Ensure the vertex collection 'nodes' exists
if not graph.has_vertex_collection('nodes'):
    graph.create_vertex_collection('nodes')

# Ensure the edge collection 'relationships' exists and define its relations
if not graph.has_edge_definition('relationships'):
    graph.create_edge_definition(
        edge_collection='relationships',
        from_vertex_collections=['nodes'],
        to_vertex_collections=['nodes']
    )

# Function to insert nodes into the database
def insert_node(node):
    """
    Inserts a node into the 'nodes' vertex collection if it does not already exist.
    """
    node_collection = graph.vertex_collection('nodes')
    if not node_collection.has(node['id']):
        node_collection.insert({
            '_key': node['id'],  # Unique identifier for the node
            'type': node['type'],  # Node type or label
            **(node.get('properties') or {})  # Additional metadata, if any
        })

# Function to insert relationships into the database
def insert_relationship(relationship):
    """
    Inserts a relationship (edge) into the 'relationships' edge collection
    if it does not already exist.
    """
    edge_collection = graph.edge_collection('relationships')
    source_id = relationship['source']['id']
    target_id = relationship['target']['id']
    edge_key = f"{source_id}_{relationship['type']}_{target_id}"
    if not edge_collection.has(edge_key):
        edge_collection.insert({
            '_key': edge_key,  # Unique identifier for the edge
            '_from': f"nodes/{source_id}",  # Source node ID
            '_to': f"nodes/{target_id}",  # Target node ID
            'type': relationship['type'],  # Relationship type
            **(relationship.get('properties') or {})  # Additional metadata, if any
        })

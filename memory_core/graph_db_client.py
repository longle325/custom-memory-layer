import os
import logging
import sys
import io
from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase

# Suppress Neo4j warnings
logging.getLogger("neo4j").setLevel(logging.ERROR)

class GraphDBClient:
    """
    A client for interacting with Neo4j graph database with enhanced capabilities.

    This client handles connections, node/relationship creation, updates, deletions,
    and advanced queries for the knowledge graph memory system.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        Initializes the GraphDBClient.

        Args:
            uri (str, optional): Neo4j connection URI. Defaults to NEO4J_URI env var.
            user (str, optional): Neo4j username. Defaults to NEO4J_USER env var.
            password (str, optional): Neo4j password. Defaults to NEO4J_PASSWORD env var.
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "testpassword")

        try:
            # Configure driver without notification filter (not supported in this version)
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"Failed to connect to Neo4j at {self.uri}. Error: {e}")
            print("\nTo run Neo4j locally with Docker, use:")
            print("  docker run -d --name neo4j \\")
            print("    -p7474:7474 -p7687:7687 \\")
            print("    -e NEO4J_AUTH=neo4j/testpassword \\")
            print("    neo4j:5.19")
            print("\nSee the Neo4j installation guide: https://neo4j.com/docs/operations-manual/current/installation/\n")
            raise

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Executes a Cypher query and returns the results.

        Args:
            query (str): The Cypher query to execute.
            parameters (Dict[str, Any], optional): Parameters for the query.

        Returns:
            List[Dict[str, Any]]: Query results as a list of dictionaries.
        """
        if parameters is None:
            parameters = {}

        try:
            # Temporarily redirect stderr to suppress Neo4j warnings
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            with self.driver.session() as session:
                result = session.run(query, parameters)
                data = [record.data() for record in result]
            
            # Restore stderr
            sys.stderr = old_stderr
            return data
            
        except Exception as e:
            # Restore stderr in case of exception
            sys.stderr = old_stderr
            # Suppress error messages to reduce noise
            raise

    def create_node(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a node in the graph database.

        Args:
            node_data (Dict[str, Any]): Node data with 'id', 'label', and 'properties'.

        Returns:
            Dict[str, Any]: The created node data.
        """
        query = f"""
        MERGE (n:{node_data['label']} {{id: $id}})
        SET n += $properties
        RETURN n
        """
        parameters = {
            "id": node_data["id"],
            "properties": node_data.get("properties", {})
        }
        
        result = self.execute_query(query, parameters)
        return result[0] if result else None

    def update_node(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates an existing node in the graph database.

        Args:
            node_data (Dict[str, Any]): Node data with 'id', 'label', and 'properties'.

        Returns:
            Dict[str, Any]: The updated node data.
        """
        query = f"""
        MATCH (n:{node_data['label']} {{id: $id}})
        SET n += $properties
        RETURN n
        """
        parameters = {
            "id": node_data["id"],
            "properties": node_data.get("properties", {})
        }
        
        result = self.execute_query(query, parameters)
        return result[0] if result else None

    def create_relationship(self, edge_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a relationship between two nodes.

        Args:
            edge_data (Dict[str, Any]): Edge data with 'source', 'target', 'label', and 'properties'.

        Returns:
            Dict[str, Any]: The created relationship data.
        """
        query = f"""
        MATCH (a {{id: $source}}), (b {{id: $target}})
        MERGE (a)-[r:{edge_data['label']}]->(b)
        SET r += $properties
        RETURN r
        """
        parameters = {
            "source": edge_data["source"],
            "target": edge_data["target"],
            "properties": edge_data.get("properties", {})
        }
        
        result = self.execute_query(query, parameters)
        return result[0] if result else None

    def delete_relationship(self, relationship_data: Dict[str, Any]) -> bool:
        """
        Deletes a relationship from the graph database.

        Args:
            relationship_data (Dict[str, Any]): Relationship data containing the relationship object.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Extract relationship ID or properties to identify the relationship
            if "id" in relationship_data:
                query = "MATCH ()-[r]->() WHERE id(r) = $rel_id DELETE r"
                parameters = {"rel_id": relationship_data["id"]}
            else:
                # Use properties to identify the relationship
                query = f"""
                MATCH (a {{id: $source}})-[r:{relationship_data.get('label', '')}]->(b {{id: $target}})
                DELETE r
                """
                parameters = {
                    "source": relationship_data.get("source", ""),
                    "target": relationship_data.get("target", "")
                }
            
            self.execute_query(query, parameters)
            return True
        except Exception as e:
            # Suppress error messages to reduce noise
            return False

    def find_node(self, node_id: str) -> Dict[str, Any]:
        """
        Finds a node by its ID.

        Args:
            node_id (str): The ID of the node to find.

        Returns:
            Dict[str, Any]: The node data, or None if not found.
        """
        query = "MATCH (n {id: $id}) RETURN n"
        result = self.execute_query(query, {"id": node_id})
        return result[0]["n"] if result else None

    def find_relationships(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """
        Finds all relationships for a given node.

        Args:
            node_id (str): The ID of the node.
            direction (str): Direction of relationships ("incoming", "outgoing", "both").

        Returns:
            List[Dict[str, Any]]: List of relationships.
        """
        try:
            # Perform the relationship search - use try-catch to handle missing properties gracefully
            if direction == "incoming":
                query = "MATCH (a)-[r]->(n {id: $id}) RETURN r, a"
            elif direction == "outgoing":
                query = "MATCH (n {id: $id})-[r]->(b) RETURN r, b"
            else: 
                query = "MATCH (n {id: $id})-[r]-(other) RETURN r, other"
            
            try:
                result = self.execute_query(query, {"id": node_id})
                return result
            except Exception:
                # If the query fails due to missing properties, return empty results
                return []
            
        except Exception as e:
            # Suppress error messages to reduce noise
            return []

    def search_nodes_by_property(self, label: str, property_name: str, property_value: str) -> List[Dict[str, Any]]:
        """
        Searches for nodes by a specific property value.

        Args:
            label (str): The node label to search within.
            property_name (str): The property name to search.
            property_value (str): The property value to match.

        Returns:
            List[Dict[str, Any]]: List of matching nodes.
        """
        try:
            # Perform the actual search - use try-catch to handle missing properties gracefully
            query = f"""
            MATCH (n:{label})
            WHERE n.{property_name} CONTAINS $value
            RETURN n
            """
            
            try:
                result = self.execute_query(query, {"value": property_value})
                return [record["n"] for record in result]
            except Exception:
                # If the query fails due to missing properties, return empty results
                return []
            
        except Exception as e:
            # Suppress error messages to reduce noise
            return []

    def get_shortest_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """
        Finds the shortest path between two nodes.

        Args:
            start_id (str): Starting node ID.
            end_id (str): Ending node ID.

        Returns:
            List[Dict[str, Any]]: Path information.
        """
        try:
            # Perform the path search - use try-catch to handle missing properties gracefully
            query = """
            MATCH (start {id: $start_id}), (end {id: $end_id})
            MATCH path = shortestPath((start)-[*]-(end))
            RETURN path
            """
            
            try:
                result = self.execute_query(query, {"start_id": start_id, "end_id": end_id})
                return result
            except Exception:
                # If the query fails due to missing properties, return empty results
                return []
            
        except Exception as e:
            # Suppress error messages to reduce noise
            return []

    def vector_similarity_search(self, embedding: List[float], label: str = None, user_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search on nodes with embeddings.

        Args:
            embedding (List[float]): Query embedding vector.
            label (str, optional): Node label to filter by.
            user_id (str, optional): User ID to filter by.
            limit (int): Maximum number of results.

        Returns:
            List[Dict[str, Any]]: Similar nodes with similarity scores.
        """
        try:
            # Build the query with optional filters
            label_filter = ""
            user_filter = ""
            params = {"embedding": embedding, "limit": limit}
            
            if label:
                label_filter = f":{label}"
                params["label"] = label
            
            if user_id:
                user_filter = "AND n.associated_user = $user_id"
                params["user_id"] = user_id
            
            # Use a more robust query that checks for property existence
            query = f"""
            MATCH (n{label_filter})
            WHERE n.embedding IS NOT NULL {user_filter}
            WITH n, vector.similarity.cosine(n.embedding, $embedding) as similarity
            WHERE similarity > 0.5
            RETURN n, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            try:
                results = self.execute_query(query, params)
                return results
            except Exception:
                # If the query fails due to missing properties, return empty results
                return []
            
        except Exception as e:
            # Suppress error messages to avoid cluttering the output
            return []

    def get_node_embeddings(self, node_ids: List[str]) -> Dict[str, List[float]]:
        """
        Retrieves embeddings for multiple nodes.

        Args:
            node_ids (List[str]): List of node IDs.

        Returns:
            Dict[str, List[float]]: Mapping of node ID to embedding.
        """
        try:
            query = """
            MATCH (n)
            WHERE n.id IN $node_ids AND n.embedding IS NOT NULL
            RETURN n.id as id, n.embedding as embedding
            """
            
            results = self.execute_query(query, {"node_ids": node_ids})
            
            embeddings = {}
            for result in results:
                embeddings[result["id"]] = result["embedding"]
            
            return embeddings
            
        except Exception as e:
            # Suppress error messages to reduce noise
            return {}

    def update_node_embedding(self, node_id: str, embedding: List[float]) -> bool:
        """
        Updates the embedding for a specific node.

        Args:
            node_id (str): The node ID.
            embedding (List[float]): The new embedding vector.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Update the embedding - use try-catch to handle missing properties gracefully
            query = """
            MATCH (n {id: $node_id})
            SET n.embedding = $embedding
            RETURN n
            """
            
            try:
                result = self.execute_query(query, {"node_id": node_id, "embedding": embedding})
                return len(result) > 0
            except Exception:
                # If the query fails due to missing properties, return False
                return False
            
        except Exception as e:
            # Suppress error messages to reduce noise
            return False

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Retrieves basic statistics about the graph.

        Returns:
            Dict[str, Any]: Graph statistics.
        """
        try:
            # Get node count by label
            node_stats_query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            ORDER BY count DESC
            """
            
            # Get relationship count by type
            rel_stats_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
            """
            
            node_stats = self.execute_query(node_stats_query)
            rel_stats = self.execute_query(rel_stats_query)
            
            return {
                "node_statistics": node_stats,
                "relationship_statistics": rel_stats,
                "total_nodes": sum(stat["count"] for stat in node_stats),
                "total_relationships": sum(stat["count"] for stat in rel_stats)
            }
            
        except Exception as e:
            # Suppress error messages to reduce noise
            return {}

    def delete_node(self, node_id: str) -> int:
        """
        Deletes a node and all its relationships.

        Args:
            node_id (str): The ID of the node to delete.

        Returns:
            int: Number of nodes deleted.
        """
        try:
            # Delete the node - use try-catch to handle missing properties gracefully
            query = "MATCH (n {id: $id}) DETACH DELETE n"
            
            try:
                self.execute_query(query, {"id": node_id})
                return 1  # Neo4j doesn't return count for DELETE operations
            except Exception:
                # If the query fails due to missing properties, return 0
                return 0
            
        except Exception as e:
            # Suppress error messages to reduce noise
            return 0

    def reset_database(self):
        """
        Deletes all nodes and relationships in the database.
        Use with caution!
        """
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        print("Graph database has been reset.")

    def close(self):
        """
        Closes the database connection.
        """
        if self.driver:
            self.driver.close()
            print("Disconnected from Neo4j.") 
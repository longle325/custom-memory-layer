import json
import re
from typing import Dict, Any, List, Optional
from .llm_client import LLMClient
from .graph_db_client import GraphDBClient
from .embedding_client import EmbeddingClient

class GraphMemory:
    """
    Manages knowledge graph memory operations with advanced features:
    - Entity deduplication using vector similarity
    - Conflict resolution for information updates
    - Enhanced semantic search capabilities
    - Multi-stage search with LLM analysis
    """

    def __init__(self, llm_client: LLMClient, graph_db_client: GraphDBClient, embedding_client: EmbeddingClient = None):
        """
        Initializes the GraphMemory with enhanced capabilities.

        Args:
            llm_client (LLMClient): The LLM client for text processing.
            graph_db_client (GraphDBClient): The graph database client.
            embedding_client (EmbeddingClient, optional): The embedding client for similarity search.
        """
        self.llm_client = llm_client
        self.graph_db_client = graph_db_client
        self.embedding_client = embedding_client
        
        # Similarity threshold for entity deduplication
        self.similarity_threshold = 0.8
        
        # Initialize embedding client if not provided
        if self.embedding_client is None:
            try:
                from .embedding_client import EmbeddingClient
                self.embedding_client = EmbeddingClient()
            except ImportError:
                print("Warning: EmbeddingClient not available. Entity deduplication will be disabled.")
                self.embedding_client = None

    def _sanitize_label(self, label: str) -> str:
        """
        Sanitize node labels to be valid Neo4j identifiers.
        
        Args:
            label (str): The original label.
            
        Returns:
            str: Sanitized label safe for Neo4j.
        """
        if not label:
            return "Entity"
        
        # Remove spaces and special characters, keep only alphanumeric and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label.strip())
        
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = "Entity_" + sanitized
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "Entity"
        
        return sanitized

    def add(self, text: str, user_id: str = None) -> Dict[str, Any]:
        """
        Extracts entities and relationships from text and adds them to the graph with deduplication.

        Args:
            text (str): The input text to process.
            user_id (str, optional): User identifier for context.

        Returns:
            Dict[str, Any]: Summary of nodes and edges created/updated.
        """
        try:
            # Extract graph data using LLM
            graph_data = self.llm_client.extract_graph_entities(text)
            
            nodes_created = []
            nodes_merged = []
            edges_created = []
            conflicts_resolved = []

            # Add user context if provided
            if user_id:
                user_node = {
                    "id": f"user_{user_id}",
                    "label": "User",
                    "properties": {"user_id": user_id, "name": f"User {user_id}"}
                }
                self.graph_db_client.create_node(user_node)
                nodes_created.append(user_node)

            # Process nodes with deduplication
            for node in graph_data.get("nodes", []):
                try:
                    # Sanitize the node label
                    original_label = node.get("label", "Entity")
                    node["label"] = self._sanitize_label(original_label)
                    
                    # Add context properties
                    if user_id:
                        node["properties"]["associated_user"] = user_id
                    
                    # Check for similar existing nodes
                    similar_nodes = self._find_similar_entities(
                        node["id"], 
                        node["label"], 
                        user_id
                    )
                    
                    if similar_nodes and self.embedding_client:
                        # Merge with existing node
                        existing_node = similar_nodes[0]
                        merged_node = self._merge_entities(node, existing_node)
                        self.graph_db_client.update_node(merged_node)
                        nodes_merged.append({
                            "new": node,
                            "existing": existing_node,
                            "merged": merged_node
                        })
                    else:
                        # Create new node
                        created_node = self.graph_db_client.create_node(node)
                        if created_node:
                            nodes_created.append(node)
                            
                            # Connect to user if provided
                            if user_id:
                                user_edge = {
                                    "source": f"user_{user_id}",
                                    "target": node["id"],
                                    "label": "MENTIONED",
                                    "properties": {"context": "conversation"}
                                }
                                self.graph_db_client.create_relationship(user_edge)
                                
                except Exception as e:
                    print(f"Error processing node {node.get('id', 'unknown')}: {e}")

            # Process relationships with conflict resolution
            for edge in graph_data.get("edges", []):
                try:
                    # Sanitize the relationship label
                    original_label = edge.get("label", "RELATED_TO")
                    edge["label"] = self._sanitize_label(original_label)
                    
                    # Add context properties
                    if user_id:
                        edge["properties"]["associated_user"] = user_id
                    
                    # Check for conflicting relationships
                    conflicts = self._find_conflicting_relationships(edge, user_id)
                    
                    if conflicts:
                        # Resolve conflicts using LLM
                        resolution = self._resolve_relationship_conflicts(edge, conflicts, text)
                        if resolution["action"] == "replace":
                            # Delete conflicting relationships
                            for conflict in conflicts:
                                self.graph_db_client.delete_relationship(conflict)
                            conflicts_resolved.extend(conflicts)
                        
                        if resolution["action"] in ["replace", "add"]:
                            # Add new relationship
                            created_edge = self.graph_db_client.create_relationship(edge)
                            if created_edge:
                                edges_created.append(edge)
                    else:
                        # No conflicts, add normally
                        created_edge = self.graph_db_client.create_relationship(edge)
                        if created_edge:
                            edges_created.append(edge)
                            
                except Exception as e:
                    print(f"Error processing edge {edge.get('source', '?')}->{edge.get('target', '?')}: {e}")

            return {
                "status": "success",
                "nodes_created": len(nodes_created),
                "nodes_merged": len(nodes_merged),
                "edges_created": len(edges_created),
                "conflicts_resolved": len(conflicts_resolved),
                "nodes": nodes_created,
                "merged_nodes": nodes_merged,
                "edges": edges_created,
                "resolved_conflicts": conflicts_resolved
            }

        except Exception as e:
            print(f"Error in GraphMemory.add: {e}")
            return {
                "status": "error",
                "message": str(e),
                "nodes_created": 0,
                "nodes_merged": 0,
                "edges_created": 0,
                "conflicts_resolved": 0
            }

    def search(self, query: str, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Enhanced search across graph memory with multiple search strategies.

        Args:
            query (str): Search query.
            user_id (str, optional): User identifier for filtering.

        Returns:
            List[Dict[str, Any]]: Combined search results.
        """
        try:
            print(f"🔍 Graph search: Query='{query}', User='{user_id}'")
            
            # Extract entities from query
            query_entities = self._extract_query_entities(query)
            
            # Perform multiple search strategies
            all_results = []
            
            # 1. Entity-based search
            for entity in query_entities:
                entity_results = self._graph_entity_search(entity, user_id)
                all_results.extend(entity_results)
            
            # 2. Vector-based semantic search
            if self.embedding_client:
                vector_results = self._vector_search(query, user_id)
                all_results.extend(vector_results)
            
            # 3. Relationship search
            relationship_results = self._relationship_search(query, user_id)
            all_results.extend(relationship_results)
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_search_results(unique_results, query)
            
            print(f"🔍 Graph search: Found {len(ranked_results)} unique results from {len(all_results)} total matches")
            return ranked_results

        except Exception as e:
            print(f"Error in GraphMemory.search: {e}")
            return []

    def _find_similar_entities(self, entity_name: str, entity_type: str, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Find entities similar to the given entity.

        Args:
            entity_name (str): Name of the entity to find similar entities for.
            entity_type (str): Type of the entity.
            user_id (str, optional): User identifier for filtering.

        Returns:
            List[Dict[str, Any]]: List of similar entities with similarity scores.
        """
        if not self.embedding_client:
            return []
        
        try:
            # Get embedding for the entity
            entity_embedding = self.embedding_client.get_embedding(entity_name)
            
            # Build user filter
            user_filter = ""
            params = {"entity_type": entity_type, "embedding": entity_embedding}
            if user_id:
                user_filter = "AND n.associated_user = $user_id"
                params["user_id"] = user_id
            
            # Search for similar entities - use try-catch to handle missing properties gracefully
            query = f"""
            MATCH (n)
            WHERE n.label = $entity_type {user_filter}
            WITH n, vector.similarity.cosine(n.embedding, $embedding) as similarity
            WHERE similarity > $threshold
            RETURN n, similarity
            ORDER BY similarity DESC
            LIMIT 5
            """
            params["threshold"] = self.similarity_threshold
            
            try:
                results = self.graph_db_client.execute_query(query, params)
                return results
            except Exception:
                # If the query fails due to missing properties, return empty results
                return []
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def _merge_entities(self, new_node: Dict[str, Any], existing_node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge properties of new node with existing node.

        Args:
            new_node (Dict[str, Any]): New node data.
            existing_node (Dict[str, Any]): Existing node data.

        Returns:
            Dict[str, Any]: Merged node data.
        """
        merged_properties = existing_node["properties"].copy()
        
        # Merge properties, preferring new values for conflicts
        for key, value in new_node["properties"].items():
            if key not in merged_properties or merged_properties[key] != value:
                merged_properties[key] = value
        
        # Update mention count
        merged_properties["mentions"] = merged_properties.get("mentions", 0) + 1
        
        return {
            "id": existing_node["id"],
            "label": existing_node["label"],
            "properties": merged_properties
        }

    def _find_conflicting_relationships(self, new_edge: Dict[str, Any], user_id: str = None) -> List[Dict[str, Any]]:
        """
        Find relationships that might conflict with the new relationship.

        Args:
            new_edge (Dict[str, Any]): New relationship to check.
            user_id (str, optional): User identifier for filtering.

        Returns:
            List[Dict[str, Any]]: List of conflicting relationships.
        """
        try:
            # Build user filter
            user_filter = ""
            params = {
                "source": new_edge["source"],
                "target": new_edge["target"],
                "relationship_type": new_edge["label"]
            }
            if user_id:
                user_filter = "AND r.associated_user = $user_id"
                params["user_id"] = user_id
            
            # Find existing relationships between the same nodes with the same type
            query = f"""
            MATCH (a {{id: $source}})-[r:{new_edge['label']}]->(b {{id: $target}})
            WHERE 1=1 {user_filter}
            RETURN r
            """
            
            results = self.graph_db_client.execute_query(query, params)
            return results
            
        except Exception as e:
            print(f"Error finding conflicting relationships: {e}")
            return []

    def _resolve_relationship_conflicts(self, new_edge: Dict[str, Any], conflicts: List[Dict], context: str) -> Dict[str, Any]:
        """
        Use LLM to resolve relationship conflicts.

        Args:
            new_edge (Dict[str, Any]): New relationship.
            conflicts (List[Dict]): List of conflicting relationships.
            context (str): Original text context.

        Returns:
            Dict[str, Any]: Resolution decision.
        """
        try:
            # Prepare conflict information for LLM
            conflict_info = []
            for conflict in conflicts:
                conflict_info.append({
                    "source": conflict.get("source", ""),
                    "target": conflict.get("target", ""),
                    "relationship": conflict.get("label", ""),
                    "properties": conflict.get("properties", {})
                })
            
            # Create prompt for conflict resolution
            prompt = f"""
            Analyze the following relationship conflict and determine the best action:
            
            New relationship: {new_edge['source']} --[{new_edge['label']}]--> {new_edge['target']}
            Context: {context}
            
            Existing conflicting relationships:
            {json.dumps(conflict_info, indent=2)}
            
            Determine if the new relationship should:
            1. "replace" - Replace conflicting relationships (if new info is more accurate)
            2. "add" - Add alongside existing relationships (if complementary)
            3. "skip" - Skip adding (if redundant or less accurate)
            
            Respond with JSON: {{"action": "replace|add|skip", "reason": "explanation"}}
            """
            
            response = self.llm_client.generate_response(prompt, is_json=True)
            resolution = json.loads(response)
            
            return resolution
            
        except Exception as e:
            print(f"Error resolving conflicts: {e}")
            # Default to adding if resolution fails
            return {"action": "add", "reason": "Error in conflict resolution"}

    def _extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract entities from search query using LLM.

        Args:
            query (str): Search query.

        Returns:
            List[Dict[str, Any]]: Extracted entities.
        """
        try:
            # Use LLM to extract entities from query
            prompt = f"""
            Extract all entities mentioned in the following search query:
            Query: "{query}"
            
            Return as JSON array of objects with "name" and "type" fields.
            Example: [{{"name": "John", "type": "Person"}}, {{"name": "Google", "type": "Organization"}}]
            If no entities are found, return an empty array: []
            """
            
            response = self.llm_client.generate_response(prompt, is_json=True)
            entities = json.loads(response)
            
            # Handle different response formats
            if isinstance(entities, list):
                return entities
            elif isinstance(entities, dict) and "entities" in entities:
                return entities["entities"] if isinstance(entities["entities"], list) else []
            else:
                print(f"Unexpected entity extraction response format: {type(entities)}")
                return []
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error in entity extraction: {e}")
            return []
        except Exception as e:
            print(f"Error extracting query entities: {e}")
            return []

    def _vector_search(self, query: str, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform vector-based semantic search.

        Args:
            query (str): Search query.
            user_id (str, optional): User identifier for filtering.

        Returns:
            List[Dict[str, Any]]: Vector search results.
        """
        if not self.embedding_client:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_client.get_embedding(query, task_type="RETRIEVAL_QUERY")
            
            # Build user filter
            user_filter = ""
            params = {"embedding": query_embedding, "threshold": self.similarity_threshold}
            if user_id:
                user_filter = "AND n.associated_user = $user_id"
                params["user_id"] = user_id
            
            # Search for similar nodes - use try-catch to handle missing properties gracefully
            query_cypher = f"""
            MATCH (n)
            WHERE n.embedding IS NOT NULL {user_filter}
            WITH n, vector.similarity.cosine(n.embedding, $embedding) as similarity
            WHERE similarity > $threshold
            RETURN n, similarity
            ORDER BY similarity DESC
            LIMIT 10
            """
            
            try:
                results = self.graph_db_client.execute_query(query_cypher, params)
                
                # Format results
                formatted_results = []
                for result in results:
                    node_data = result["n"]
                    formatted_results.append({
                        "type": "vector_match",
                        "data": node_data,
                        "relevance": result["similarity"],
                        "source": "vector_search"
                    })
                
                print(f"🔍 Graph vector search: Found {len(formatted_results)} results (threshold: {self.similarity_threshold})")
                return formatted_results
            except Exception:
                # If the query fails due to missing properties, return empty results
                print(f"🔍 Graph vector search: No results (missing properties)")
                return []
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

    def _graph_entity_search(self, entity: Dict[str, Any], user_id: str = None) -> List[Dict[str, Any]]:
        """
        Search for entities in the graph.

        Args:
            entity (Dict[str, Any]): Entity to search for.
            user_id (str, optional): User identifier for filtering.

        Returns:
            List[Dict[str, Any]]: Entity search results.
        """
        try:
            # Build user filter
            user_filter = ""
            params = {"entity_name": entity["name"], "entity_type": entity["type"]}
            if user_id:
                user_filter = "AND n.associated_user = $user_id"
                params["user_id"] = user_id
            
            # Search for matching entities - use try-catch to handle missing properties gracefully
            query = f"""
            MATCH (n)
            WHERE n.label = $entity_type {user_filter}
            AND (n.name CONTAINS $entity_name OR n.id CONTAINS $entity_name)
            RETURN n
            LIMIT 5
            """
            
            try:
                results = self.graph_db_client.execute_query(query, params)
                
                # Format results
                formatted_results = []
                for result in results:
                    node_data = result["n"]
                    formatted_results.append({
                        "type": "entity_match",
                        "data": node_data,
                        "relevance": 0.9,  # High relevance for direct matches
                        "source": "entity_search"
                    })
                
                print(f"🔍 Graph entity search: Found {len(formatted_results)} matches for '{entity['name']}' ({entity['type']})")
                return formatted_results
            except Exception:
                # If the query fails due to missing properties, return empty results
                print(f"🔍 Graph entity search: No matches for '{entity['name']}' ({entity['type']}) - missing properties")
                return []
            
        except Exception as e:
            print(f"Error in entity search: {e}")
            return []

    def _relationship_search(self, query: str, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Search for relationships in the graph.

        Args:
            query (str): Search query.
            user_id (str, optional): User identifier for filtering.

        Returns:
            List[Dict[str, Any]]: Relationship search results.
        """
        try:
            # Build user filter
            user_filter = ""
            params = {"query": query.lower()}
            if user_id:
                user_filter = "AND r.associated_user = $user_id"
                params["user_id"] = user_id
            
            # Search for relationships containing query terms
            query_cypher = f"""
            MATCH (a)-[r]->(b)
            WHERE 1=1 {user_filter}
            AND (toLower(type(r)) CONTAINS $query 
                 OR toLower(a.name) CONTAINS $query 
                 OR toLower(b.name) CONTAINS $query)
            RETURN a, r, b
            LIMIT 10
            """
            
            results = self.graph_db_client.execute_query(query_cypher, params)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "type": "relationship_match",
                    "data": {
                        "source": result["a"],
                        "relationship": result["r"],
                        "target": result["b"]
                    },
                    "relevance": 0.7,  # Medium relevance for relationship matches
                    "source": "relationship_search"
                })
            
            print(f"🔍 Graph relationship search: Found {len(formatted_results)} relationship matches for '{query}'")
            return formatted_results
            
        except Exception as e:
            print(f"Error in relationship search: {e}")
            return []

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on data content.

        Args:
            results (List[Dict[str, Any]]): List of search results.

        Returns:
            List[Dict[str, Any]]: Deduplicated results.
        """
        seen = set()
        unique_results = []
        
        for result in results:
            # Create a unique identifier for the result
            if result["type"] == "vector_match":
                result_id = f"node_{result['data'].get('id', '')}"
            elif result["type"] == "entity_match":
                result_id = f"node_{result['data'].get('id', '')}"
            elif result["type"] == "relationship_match":
                result_id = f"rel_{result['data']['source'].get('id', '')}_{result['data']['target'].get('id', '')}"
            else:
                result_id = str(hash(str(result["data"])))
            
            if result_id not in seen:
                seen.add(result_id)
                unique_results.append(result)
        
        return unique_results

    def _rank_search_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rank search results based on relevance and query matching.

        Args:
            results (List[Dict[str, Any]]): List of search results.
            query (str): Original search query.

        Returns:
            List[Dict[str, Any]]: Ranked results.
        """
        # Calculate additional relevance scores
        for result in results:
            base_relevance = result.get("relevance", 0.5)
            
            # Boost relevance based on query term matching
            query_terms = query.lower().split()
            data_text = str(result["data"]).lower()
            
            term_matches = sum(1 for term in query_terms if term in data_text)
            term_boost = min(term_matches * 0.1, 0.3)  # Max 0.3 boost
            
            # Source-specific boosts
            source_boost = 0.0
            if result.get("source") == "vector_search":
                source_boost = 0.1  # Vector search gets slight boost
            elif result.get("source") == "entity_search":
                source_boost = 0.2  # Direct entity matches get higher boost
            
            # Calculate final relevance
            result["relevance"] = min(base_relevance + term_boost + source_boost, 1.0)
        
        # Sort by relevance
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return results

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieves all graph data associated with a specific user.

        Args:
            user_id (str): The user identifier.

        Returns:
            Dict[str, Any]: User's knowledge graph context.
        """
        try:
            # Get user node
            user_node = self.graph_db_client.find_node(f"user_{user_id}")
            
            # Get all nodes associated with the user
            associated_nodes = self.graph_db_client.execute_query(
                """
                MATCH (user:User {user_id: $user_id})-[:MENTIONED]->(n)
                RETURN n, labels(n) as labels
                """,
                {"user_id": user_id}
            )
            
            # Get relationships between associated nodes
            relationships = self.graph_db_client.execute_query(
                """
                MATCH (user:User {user_id: $user_id})-[:MENTIONED]->(n1)
                MATCH (n1)-[r]-(n2)
                WHERE (user)-[:MENTIONED]->(n2)
                RETURN r, n1, n2
                """,
                {"user_id": user_id}
            )
            
            return {
                "user_node": user_node,
                "associated_nodes": associated_nodes,
                "relationships": relationships,
                "total_nodes": len(associated_nodes),
                "total_relationships": len(relationships)
            }

        except Exception as e:
            print(f"Error getting user context for {user_id}: {e}")
            return {}

    def find_connections(self, entity1_id: str, entity2_id: str) -> List[Dict[str, Any]]:
        """
        Finds connections between two entities in the graph.

        Args:
            entity1_id (str): First entity ID.
            entity2_id (str): Second entity ID.

        Returns:
            List[Dict[str, Any]]: Connection paths between entities.
        """
        try:
            paths = self.graph_db_client.get_shortest_path(entity1_id, entity2_id)
            return paths
        except Exception as e:
            print(f"Error finding connections between {entity1_id} and {entity2_id}: {e}")
            return []

    def delete_user_data(self, user_id: str) -> bool:
        """
        Deletes all graph data associated with a specific user.

        Args:
            user_id (str): The user identifier.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Delete user node and all associated data
            self.graph_db_client.execute_query(
                """
                MATCH (user:User {user_id: $user_id})
                OPTIONAL MATCH (user)-[:MENTIONED]->(n)
                DETACH DELETE user, n
                """,
                {"user_id": user_id}
            )
            return True
        except Exception as e:
            print(f"Error deleting user data for {user_id}: {e}")
            return False

    def _calculate_relevance(self, node_data: Dict[str, Any], keywords: List[str]) -> float:
        """
        Calculates relevance score for a node based on keywords.

        Args:
            node_data (Dict[str, Any]): Node data.
            keywords (List[str]): Search keywords.

        Returns:
            float: Relevance score (0-1).
        """
        relevance = 0.0
        text_content = " ".join(str(v) for v in node_data.values()).lower()
        
        for keyword in keywords:
            if keyword in text_content:
                relevance += 1.0 / len(keywords)
                
        return relevance 
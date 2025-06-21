import json
import time
import functools
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .llm_client import LLMClient
from .embedding_client import EmbeddingClient
from .vector_db_client import VectorDBClient
from .graph_db_client import GraphDBClient
from .graph_memory import GraphMemory

class Memory:
    """
    Main memory orchestrator that combines vector and graph memory systems with enhanced capabilities:
    - Entity deduplication using vector similarity
    - Conflict resolution for information updates
    - Multi-stage semantic search
    - Intelligent memory management
    - Performance optimizations with connection pooling and caching
    """

    def __init__(self, 
                 google_api_key: str = None,
                 milvus_uri: str = "http://milvus-standalone:19530",
                 neo4j_uri: str = None,
                 neo4j_user: str = None,
                 neo4j_password: str = None):
        """
        Initializes the Memory system with all required clients.

        Args:
            google_api_key (str, optional): Google API key for LLM and embeddings.
            milvus_uri (str): Milvus database URI.
            neo4j_uri (str, optional): Neo4j database URI.
            neo4j_user (str, optional): Neo4j username.
            neo4j_password (str, optional): Neo4j password.
        """
        # Initialize core clients with connection pooling
        self.llm_client = LLMClient(api_key=google_api_key)
        self.embedding_client = EmbeddingClient(api_key=google_api_key)
        self.vector_db_client = VectorDBClient(uri=milvus_uri)
        
        # Performance monitoring
        self.performance_metrics = {}
        self._lock = threading.Lock()
        
        # Initialize graph components if Neo4j credentials provided
        self.graph_enabled = False
        if neo4j_uri or neo4j_user or neo4j_password:
            try:
                self.graph_db_client = GraphDBClient(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password
                )
                # Pass embedding client to GraphMemory for similarity search
                self.graph_memory = GraphMemory(
                    self.llm_client, 
                    self.graph_db_client, 
                    self.embedding_client
                )
                self.graph_enabled = True
                print("Graph memory enabled with Neo4j and enhanced capabilities")
            except Exception as e:
                print(f"Graph memory disabled due to Neo4j connection error: {e}")
                self.graph_enabled = False
        else:
            print("Graph memory disabled - no Neo4j credentials provided")

    @staticmethod
    def _time_operation(operation_name: str):
        """Decorator for timing operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                start_time = time.time()
                try:
                    result = func(self, *args, **kwargs)
                    duration = time.time() - start_time
                    with self._lock:
                        self.performance_metrics[operation_name] = duration
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    with self._lock:
                        self.performance_metrics[f"{operation_name}_error"] = duration
                    raise e
            return wrapper
        return decorator

    @staticmethod
    def _retry_operation(max_retries: int = 3, delay: float = 1.0):
        """Decorator for retrying failed operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                last_exception = None
                for attempt in range(max_retries):
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            time.sleep(delay * (2 ** attempt))  # Exponential backoff
                raise last_exception
            return wrapper
        return decorator

    @_time_operation("add_memory")
    @_retry_operation(max_retries=2)
    def add(self, text: str, user_id: str = None) -> Dict[str, Any]:
        """
        Adds new information to both vector and graph memory systems with enhanced processing.

        Args:
            text (str): The input text containing new information.
            user_id (str, optional): User identifier for personalized memory.

        Returns:
            Dict[str, Any]: Summary of operations performed with detailed results.
        """
        results = {
            "vector_memory": {"status": "skipped"},
            "graph_memory": {"status": "skipped"},
            "user_id": user_id,
            "enhanced_features": {
                "entity_deduplication": False,
                "conflict_resolution": False,
                "semantic_search": False
            }
        }

        # Use ThreadPoolExecutor for parallel processing with optimized worker count
        max_workers = 2 if self.graph_enabled else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            # Submit vector memory task
            futures["vector"] = executor.submit(self._add_to_vector_memory, text, user_id)
            
            # Submit graph memory task if enabled
            if self.graph_enabled:
                futures["graph"] = executor.submit(self._add_to_graph_memory, text, user_id)

            # Collect results with timeout
            for task_name, future in futures.items():
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    if task_name == "vector":
                        results["vector_memory"] = result
                    elif task_name == "graph":
                        results["graph_memory"] = result
                        # Update enhanced features status
                        if result.get("status") == "success":
                            results["enhanced_features"]["entity_deduplication"] = result.get("nodes_merged", 0) > 0
                            results["enhanced_features"]["conflict_resolution"] = result.get("conflicts_resolved", 0) > 0
                except Exception as e:
                    error_result = {"status": "error", "message": str(e)}
                    if task_name == "vector":
                        results["vector_memory"] = error_result
                    elif task_name == "graph":
                        results["graph_memory"] = error_result

        return results

    @_time_operation("add_vector_memory")
    def _add_to_vector_memory(self, text: str, user_id: str = None) -> Dict[str, Any]:
        """
        Adds information to vector memory with intelligent updates and conflict resolution.

        Args:
            text (str): Input text.
            user_id (str, optional): User identifier.

        Returns:
            Dict[str, Any]: Vector memory operation results.
        """
        try:
            # Extract facts from text with early termination for empty results
            fact_data = self.llm_client.extract_facts(text)
            new_facts = fact_data.get("facts", [])
            
            if not new_facts:
                return {"status": "no_facts", "message": "No facts extracted from text"}

            # Batch process embeddings for better performance
            embeddings = []
            for fact in new_facts:
                embedding = self.embedding_client.get_embedding(fact)
                embeddings.append(embedding)

            # Search for similar memories with optimized limit and user filtering
            search_results = []
            for embedding in embeddings:
                results = self.vector_db_client.search(embedding, limit=10, user_id=user_id)
                search_results.extend(results)

            # Handle empty memory scenario gracefully
            if not search_results:
                # No existing memories found, batch add new facts
                actions = []
                for fact in new_facts:
                    actions.append({
                        "event": "ADD",
                        "text": fact,
                        "id": None
                    })
            else:
                # Determine memory update actions with enhanced conflict resolution
                update_data = self.llm_client.update_memory(new_facts, search_results)
                actions = update_data.get("memory", [])

            # Execute actions with detailed tracking and batch processing
            actions_performed = {
                "ADD": 0,
                "UPDATE": 0,
                "DELETE": 0,
                "NONE": 0,
                "CONFLICTS_RESOLVED": 0
            }

            # Batch process actions for better performance
            batch_texts = []
            batch_embeddings = []
            batch_user_ids = []
            batch_ids_to_delete = []

            for action in actions:
                event = action.get("event")
                text_content = action.get("text", "")
                
                if event == "ADD":
                    batch_texts.append(text_content)
                    batch_embeddings.append(self.embedding_client.get_embedding(text_content))
                    batch_user_ids.append(user_id)
                    actions_performed["ADD"] += 1
                    
                elif event == "UPDATE":
                    memory_id = action.get("id")
                    if memory_id:
                        batch_ids_to_delete.append(memory_id)
                        batch_texts.append(text_content)
                        batch_embeddings.append(self.embedding_client.get_embedding(text_content))
                        batch_user_ids.append(user_id)
                        actions_performed["UPDATE"] += 1
                        actions_performed["CONFLICTS_RESOLVED"] += 1
                        
                elif event == "DELETE":
                    memory_id = action.get("id")
                    if memory_id:
                        batch_ids_to_delete.append(memory_id)
                        actions_performed["DELETE"] += 1
                        
                elif event == "NONE":
                    actions_performed["NONE"] += 1

            # Execute batch operations
            if batch_texts:
                # Pass user_id directly to vector database instead of prepending to text
                self.vector_db_client.insert(batch_texts, batch_embeddings, batch_user_ids)

            if batch_ids_to_delete:
                for memory_id in batch_ids_to_delete:
                    self.vector_db_client.delete(memory_id)

            return {
                "status": "success",
                "actions_performed": actions_performed,
                "facts_processed": len(new_facts),
                "memories_added": actions_performed["ADD"],
                "memories_updated": actions_performed["UPDATE"],
                "memories_deleted": actions_performed["DELETE"],
                "conflicts_resolved": actions_performed["CONFLICTS_RESOLVED"]
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    @_time_operation("add_graph_memory")
    def _add_to_graph_memory(self, text: str, user_id: str = None) -> Dict[str, Any]:
        """
        Adds information to graph memory with entity extraction and relationship mapping.

        Args:
            text (str): Input text.
            user_id (str, optional): User identifier.

        Returns:
            Dict[str, Any]: Graph memory operation results.
        """
        try:
            if not self.graph_enabled:
                return {"status": "disabled", "message": "Graph memory not enabled"}

            result = self.graph_memory.add(text, user_id)
            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}

    @_time_operation("search_memory")
    @_retry_operation(max_retries=2)
    def search(self, query: str, user_id: str = None, limit: int = 5) -> Dict[str, Any]:
        """
        Performs multi-stage semantic search across vector and graph memory systems.

        Args:
            query (str): Search query.
            user_id (str, optional): User identifier for personalized search.
            limit (int): Maximum number of results to return.

        Returns:
            Dict[str, Any]: Combined search results from both memory systems.
        """
        try:
            # Get query embedding for vector search
            query_embedding = self.embedding_client.get_embedding(query)
            
            # Parallel search in both memory systems
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                
                # Vector search with user isolation
                futures["vector"] = executor.submit(
                    self.vector_db_client.search, 
                    query_embedding, 
                    limit,
                    user_id
                )
                
                # Graph search if enabled
                if self.graph_enabled:
                    futures["graph"] = executor.submit(
                        self.graph_memory.search, 
                        query, 
                        user_id
                    )

                # Collect results
                vector_results = futures["vector"].result(timeout=15)
                graph_results = futures["graph"].result(timeout=15) if self.graph_enabled else []

            # Combine and rank results
            combined_results = self._combine_search_results(vector_results, graph_results, query)
            ranked_results = self._rank_combined_results(combined_results, query)

            return {
                "vector_results": vector_results,
                "graph_results": graph_results,
                "combined_results": ranked_results[:limit],
                "total_results": len(ranked_results)
            }

        except Exception as e:
            return {
                "vector_results": [],
                "graph_results": [],
                "combined_results": [],
                "total_results": 0,
                "error": str(e)
            }

    def _combine_search_results(self, vector_results: List[Dict], graph_results: List[Dict], query: str) -> List[Dict]:
        """
        Combines results from vector and graph memory with deduplication.

        Args:
            vector_results (List[Dict]): Results from vector memory.
            graph_results (List[Dict]): Results from graph memory.
            query (str): Original search query.

        Returns:
            List[Dict]: Combined and deduplicated results.
        """
        combined = []
        seen_texts = set()

        # Add vector results
        for result in vector_results:
            text = result.get("text", "")
            if text and text not in seen_texts:
                combined.append({
                    **result,
                    "source": "vector",
                    "type": "semantic_match"
                })
                seen_texts.add(text)

        # Add graph results
        for result in graph_results:
            data = result.get("data", "")
            if data and data not in seen_texts:
                combined.append({
                    **result,
                    "source": "graph",
                    "type": "entity_relationship"
                })
                seen_texts.add(data)

        return combined

    def _rank_combined_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Ranks combined results based on relevance and source diversity.

        Args:
            results (List[Dict]): Combined search results.
            query (str): Original search query.

        Returns:
            List[Dict]: Ranked results.
        """
        # Simple ranking based on relevance score and source diversity
        for result in results:
            # Boost graph results slightly for entity relationships
            if result.get("source") == "graph":
                result["final_score"] = result.get("relevance", 0) * 1.1
            else:
                result["final_score"] = result.get("relevance", 0)

        # Sort by final score
        return sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)

    @_time_operation("get_memory_stats")
    def get_memory_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """
        Retrieves comprehensive memory statistics.

        Args:
            user_id (str, optional): User identifier for personalized stats.

        Returns:
            Dict[str, Any]: Memory statistics.
        """
        try:
            stats = {
                "vector_memory": {},
                "graph_memory": {},
                "performance_metrics": self.performance_metrics
            }

            # Get vector memory stats
            vector_stats = self.vector_db_client.get_statistics()
            stats["vector_memory"] = vector_stats

            # Get graph memory stats if enabled
            if self.graph_enabled:
                graph_stats = self.graph_db_client.get_graph_statistics()
                stats["graph_memory"] = graph_stats

            return stats

        except Exception as e:
            return {"error": str(e)}

    @_time_operation("delete_user_data")
    @_retry_operation(max_retries=2)
    def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Deletes all data associated with a specific user.

        Args:
            user_id (str): User identifier.

        Returns:
            Dict[str, Any]: Deletion operation results.
        """
        try:
            results = {
                "vector_deleted": 0,
                "graph_deleted": 0,
                "status": "success"
            }

            # Delete from vector memory using proper user_id field
            try:
                vector_result = self.vector_db_client.delete_by_user_id(user_id)
                results["vector_deleted"] = vector_result
            except Exception as e:
                results["vector_error"] = str(e)

            # Delete from graph memory if enabled
            if self.graph_enabled:
                try:
                    graph_result = self.graph_memory.delete_user_data(user_id)
                    results["graph_deleted"] = graph_result if isinstance(graph_result, int) else 0
                except Exception as e:
                    results["graph_error"] = str(e)

            return results

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns performance and operational statistics.

        Returns:
            Dict[str, Any]: Statistics dictionary.
        """
        return {
            "performance_metrics": self.performance_metrics,
            "graph_enabled": self.graph_enabled,
            "memory_system_status": "operational"
        }

    def reset_all(self):
        """
        Resets all memory systems (use with caution).
        """
        try:
            self.vector_db_client.reset()
            if self.graph_enabled:
                self.graph_db_client.reset_database()
            print("All memory systems reset successfully")
        except Exception as e:
            print(f"Error resetting memory systems: {e}")

    def close(self):
        """
        Properly closes all database connections.
        """
        try:
            self.vector_db_client.close()
            if self.graph_enabled:
                self.graph_db_client.close()
            print("All memory system connections closed")
        except Exception as e:
            print(f"Error closing memory system connections: {e}") 
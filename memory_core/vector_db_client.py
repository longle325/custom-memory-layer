import uuid
import time
import functools
from typing import List, Dict, Any, Optional
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

class VectorDBClient:
    """
    A client for interacting with a Milvus vector database with enhanced capabilities.

    This client handles the connection, collection management, and advanced
    CRUD (Create, Read, Update, Delete) operations for vector memories.

    It requires the `pymilvus` package to be installed (`pip install pymilvus`).
    """
    def __init__(
        self,
        uri: str = "http://milvus-standalone:19530",
        collection_name: str = "custom_memory_collection"
    ):
        """
        Initializes the VectorDBClient and connects to Milvus.

        Args:
            uri (str): The URI for the Milvus standalone or cluster instance.
                       Defaults to "http://milvus-standalone:19530".
            collection_name (str): The name of the collection to use for storing memories.
                                   If it doesn't exist, it will be created automatically.
        """
        self.uri = uri
        self.collection_name = collection_name
        self._connection_alive = False
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        
        try:
            self._connect()
            print(f"Successfully connected to Milvus at {self.uri}")
        except Exception as e:
            print(f"Failed to connect to Milvus at {self.uri}. Error: {e}")
            print("\nTo run Milvus locally with Docker, use:")
            print("  docker run -d --name milvus-standalone \\")
            print("    -p 19530:19530 -p 9091:9091 \\")
            print("    milvusdb/milvus:v2.3.9-standalone")
            print("\nSee the Milvus installation guide: https://milvus.io/docs/install_standalone-docker.md\n")
            raise

        self._create_collection_if_not_exists()
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def _connect(self):
        """Establish connection to Milvus with retry logic"""
        try:
            connections.connect(alias="default", uri=self.uri)
            self._connection_alive = True
            self._last_health_check = time.time()
        except Exception as e:
            self._connection_alive = False
            raise e

    def _check_connection_health(self):
        """Check if connection is healthy and reconnect if needed"""
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            try:
                # Simple health check query
                self.collection.query(expr="id != ''", limit=1)
                self._connection_alive = True
                self._last_health_check = current_time
            except Exception:
                self._connection_alive = False
                self._connect()

    @staticmethod
    def _retry_operation(max_retries: int = 3, delay: float = 1.0):
        """Decorator for retrying failed operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                last_exception = None
                for attempt in range(max_retries):
                    try:
                        self._check_connection_health()
                        return func(self, *args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            time.sleep(delay * (2 ** attempt))  # Exponential backoff
                            # Try to reconnect on failure
                            try:
                                self._connect()
                            except:
                                pass
                raise last_exception
            return wrapper
        return decorator

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
                    print(f"⏱️ {operation_name}: {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    print(f"❌ {operation_name} error: {duration:.3f}s - {e}")
                    raise e
            return wrapper
        return decorator

    def _create_collection_if_not_exists(self):
        """
        Creates the Milvus collection with a predefined schema if it doesn't already exist.
        """
        if utility.has_collection(self.collection_name):
            # Check if existing collection has user_id field
            self._check_and_migrate_collection()
            return

        print(f"Collection '{self.collection_name}' not found. Creating a new one.")
        
        # Define the schema for the collection with user_id field for proper isolation
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100)  # Add user_id field
        ]
        schema = CollectionSchema(fields, description="Collection for custom memory with user isolation")

        # Create the collection
        collection = Collection(self.collection_name, schema)

        # Optimized index parameters for better performance
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Collection '{self.collection_name}' created and index built successfully.")

    def _check_and_migrate_collection(self):
        """
        Check if existing collection has user_id field and handle migration if needed.
        """
        try:
            # Get existing collection
            existing_collection = Collection(self.collection_name)
            existing_collection.load()
            
            # Check if user_id field exists
            schema = existing_collection.schema
            field_names = [field.name for field in schema.fields]
            
            if "user_id" not in field_names:
                print("⚠️  Existing collection doesn't have user_id field. Migration needed.")
                print("   For now, user isolation will be limited. Consider recreating the collection.")
                print("   To recreate: delete the collection and restart the application.")
                
                # For backward compatibility, we'll use text-based filtering for existing collections
                self._use_text_based_filtering = True
            else:
                self._use_text_based_filtering = False
                print("✅ Collection has user_id field - proper user isolation enabled.")
                
        except Exception as e:
            print(f"Error checking collection schema: {e}")
            self._use_text_based_filtering = True

    @_time_operation("insert_memories")
    @_retry_operation(max_retries=2)
    def insert(self, texts: List[str], vectors: List[List[float]], user_ids: List[str] = None) -> List[str]:
        """
        Inserts memories (texts and their corresponding vectors) into the collection.

        Args:
            texts (List[str]): The list of text memories.
            vectors (List[List[float]]): The list of embedding vectors.
            user_ids (List[str], optional): The list of user IDs for each memory.

        Returns:
            List[str]: A list of unique IDs generated for the inserted memories.
        """
        if len(texts) != len(vectors):
            raise ValueError("The number of texts and vectors must be the same.")
        
        if user_ids and len(texts) != len(user_ids):
            raise ValueError("The number of texts and user_ids must be the same.")

        # Generate IDs and prepare data
        ids = [str(uuid.uuid4()) for _ in texts]
        
        # Handle user_ids - if not provided, use None for each
        if user_ids is None:
            user_ids = [None] * len(texts)
        
        # Check if we need to use text-based filtering for backward compatibility
        if hasattr(self, '_use_text_based_filtering') and self._use_text_based_filtering:
            # For existing collections without user_id field, prepend user_id to text
            processed_texts = []
            for i, text in enumerate(texts):
                if user_ids[i]:
                    processed_texts.append(f"[User: {user_ids[i]}] {text}")
                else:
                    processed_texts.append(text)
            entities = [ids, processed_texts, vectors]
        else:
            # For new collections with user_id field
            entities = [ids, texts, vectors, user_ids]
        
        # Batch insert with error handling
        try:
            self.collection.insert(entities)
            self.collection.flush()  # Ensure data is written to disk
            print(f"✅ Inserted {len(ids)} memories into '{self.collection_name}'.")
            return ids
        except Exception as e:
            print(f"❌ Failed to insert memories: {e}")
            raise

    @_time_operation("search_memories")
    @_retry_operation(max_retries=2)
    def search(self, query_vector: List[float], limit: int = 5, user_id: str = None) -> List[Dict[str, Any]]:
        """
        Searches for the most similar memories based on a query vector with user isolation.

        Args:
            query_vector (List[float]): The vector representation of the search query.
            limit (int): The maximum number of results to return.
            user_id (str, optional): User identifier to filter results.

        Returns:
            List[Dict[str, Any]]: A list of search results, each containing the
                                  memory's id, text, and similarity score.
        """
        # Optimized search parameters for better performance
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        try:
            # Check if we need to use text-based filtering for backward compatibility
            if hasattr(self, '_use_text_based_filtering') and self._use_text_based_filtering:
                # For existing collections without user_id field, use text-based filtering
                filter_expr = None
                output_fields = ["id", "text"]
                
                # First, get all results
                results = self.collection.search(
                    data=[query_vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=limit * 3,  # Get more results to filter
                    output_fields=output_fields
                )
                
                # Filter results by user_id in text
                processed_results = []
                for hit in results[0]:
                    text = hit.entity.get("text", "")
                    if user_id:
                        # Check if text contains user_id marker
                        if f"[User: {user_id}]" in text:
                            # Remove the user_id marker from displayed text
                            clean_text = text.replace(f"[User: {user_id}] ", "")
                            processed_results.append({
                                "id": hit.entity.get("id"),
                                "text": clean_text,
                                "user_id": user_id,
                                "score": hit.distance,
                                "relevance": 1.0 / (1.0 + hit.distance)
                            })
                    else:
                        # No user_id filter, include all results
                        processed_results.append({
                            "id": hit.entity.get("id"),
                            "text": text,
                            "user_id": None,
                            "score": hit.distance,
                            "relevance": 1.0 / (1.0 + hit.distance)
                        })
                
                # Limit results
                processed_results = processed_results[:limit]
            else:
                # For new collections with user_id field, use proper database filtering
                filter_expr = None
                if user_id:
                    filter_expr = f'user_id == "{user_id}"'
                
                results = self.collection.search(
                    data=[query_vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=limit,
                    output_fields=["id", "text", "user_id"],
                    expr=filter_expr
                )

                processed_results = []
                for hit in results[0]:
                    processed_results.append({
                        "id": hit.entity.get("id"),
                        "text": hit.entity.get("text"),
                        "user_id": hit.entity.get("user_id"),
                        "score": hit.distance,
                        "relevance": 1.0 / (1.0 + hit.distance)  # Convert distance to relevance score
                    })
            
            print(f"🔍 Vector search: Found {len(processed_results)} results (limit: {limit}, user_id: {user_id})")
            return processed_results
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return []

    @_time_operation("get_memory")
    @_retry_operation(max_retries=2)
    def get(self, memory_id: str) -> Dict[str, Any]:
        """
        Retrieves a single memory by its unique ID.

        Args:
            memory_id (str): The ID of the memory to retrieve.

        Returns:
            Dict[str, Any]: The memory object, or None if not found.
        """
        try:
            results = self.collection.query(
                expr=f"id == '{memory_id}'",
                output_fields=["id", "text"]
            )
            return results[0] if results else None
        except Exception as e:
            print(f"❌ Failed to get memory {memory_id}: {e}")
            return None
        
    @_time_operation("delete_memory")
    @_retry_operation(max_retries=2)
    def delete(self, memory_id: str) -> int:
        """
        Deletes a memory by its unique ID.

        Args:
            memory_id (str): The ID of the memory to delete.

        Returns:
            int: The number of memories deleted (usually 1 or 0).
        """
        try:
            delete_result = self.collection.delete(f"id in ['{memory_id}']")
            self.collection.flush()
            print(f"🗑️ Deleted memory with id: {memory_id}")
            return delete_result.delete_count
        except Exception as e:
            print(f"❌ Failed to delete memory {memory_id}: {e}")
            return 0

    @_time_operation("delete_by_filter")
    @_retry_operation(max_retries=2)
    def delete_by_filter(self, text_filter: str) -> int:
        """
        Deletes memories that contain the specified text filter.

        Args:
            text_filter (str): Text pattern to match for deletion.

        Returns:
            int: The number of memories deleted.
        """
        try:
            # First, find all memories that match the filter
            all_memories = self.list_all(limit=10000)  # Get all memories
            matching_ids = []
            
            for memory in all_memories:
                if text_filter in memory.get("text", ""):
                    matching_ids.append(memory["id"])
            
            if not matching_ids:
                return 0
            
            # Batch delete matching memories
            delete_result = self.collection.delete(f"id in {matching_ids}")
            self.collection.flush()
            
            print(f"🗑️ Deleted {len(matching_ids)} memories matching filter: {text_filter}")
            return len(matching_ids)
            
        except Exception as e:
            print(f"❌ Error deleting by filter: {e}")
            return 0

    @_time_operation("delete_by_user_id")
    @_retry_operation(max_retries=2)
    def delete_by_user_id(self, user_id: str) -> int:
        """
        Deletes all memories associated with a specific user ID.

        Args:
            user_id (str): User identifier to delete memories for.

        Returns:
            int: The number of memories deleted.
        """
        try:
            # Check if we need to use text-based filtering for backward compatibility
            if hasattr(self, '_use_text_based_filtering') and self._use_text_based_filtering:
                # For existing collections without user_id field, use text-based filtering
                return self.delete_by_filter(f"[User: {user_id}]")
            else:
                # For new collections with user_id field, use proper database filtering
                delete_result = self.collection.delete(f'user_id == "{user_id}"')
                self.collection.flush()
                
                print(f"🗑️ Deleted memories for user: {user_id}")
                return delete_result.delete_count
            
        except Exception as e:
            print(f"❌ Error deleting by user_id: {e}")
            return 0

    @_time_operation("list_all_memories")
    @_retry_operation(max_retries=2)
    def list_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves all memories from the collection, up to a given limit.

        Args:
            limit (int): The maximum number of memories to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of all memories.
        """
        try:
            results = self.collection.query(
                expr="id != ''",
                output_fields=["id", "text"],
                limit=limit
            )
            return results
        except Exception as e:
            print(f"❌ Failed to list memories: {e}")
            return []

    @_time_operation("get_statistics")
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves statistics about the collection.

        Returns:
            Dict[str, Any]: Collection statistics.
        """
        try:
            # Get collection statistics
            stats = {
                "collection_name": self.collection_name,
                "total_entities": self.collection.num_entities,
                "index_status": "loaded" if self.collection.has_index() else "not_loaded",
                "connection_status": "healthy" if self._connection_alive else "unhealthy"
            }
            
            # Get index information if available
            if self.collection.has_index():
                index_info = self.collection.index()
                stats["index_type"] = index_info.params.get("index_type", "unknown")
                stats["metric_type"] = index_info.params.get("metric_type", "unknown")
            
            return stats
        except Exception as e:
            return {
                "error": str(e),
                "collection_name": self.collection_name,
                "connection_status": "error"
            }

    @_time_operation("search_by_text")
    def search_by_text(self, text_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for memories by text content using exact matching.

        Args:
            text_query (str): The text to search for.
            limit (int): The maximum number of results to return.

        Returns:
            List[Dict[str, Any]]: A list of matching memories.
        """
        try:
            # Use text matching for exact searches
            results = self.collection.query(
                expr=f"text like '%{text_query}%'",
                output_fields=["id", "text"],
                limit=limit
            )
            return results
        except Exception as e:
            print(f"❌ Text search failed: {e}")
            return []

    @_time_operation("batch_delete")
    @_retry_operation(max_retries=2)
    def batch_delete(self, memory_ids: List[str]) -> int:
        """
        Deletes multiple memories by their IDs.

        Args:
            memory_ids (List[str]): List of memory IDs to delete.

        Returns:
            int: The number of memories deleted.
        """
        if not memory_ids:
            return 0
            
        try:
            # Format IDs for deletion query
            formatted_ids = [f"'{id}'" for id in memory_ids]
            ids_string = f"[{', '.join(formatted_ids)}]"
            
            delete_result = self.collection.delete(f"id in {ids_string}")
            self.collection.flush()
            
            print(f"🗑️ Batch deleted {len(memory_ids)} memories")
            return delete_result.delete_count
        except Exception as e:
            print(f"❌ Batch delete failed: {e}")
            return 0

    def reset(self):
        """
        Resets the collection by dropping and recreating it.
        """
        try:
            # Drop the collection
            utility.drop_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' dropped.")
            
            # Recreate the collection
            self._create_collection_if_not_exists()
            self.collection = Collection(self.collection_name)
            self.collection.load()
            print(f"Collection '{self.collection_name}' recreated.")
        except Exception as e:
            print(f"❌ Failed to reset collection: {e}")

    def close(self):
        """
        Closes the connection to Milvus.
        """
        try:
            connections.disconnect("default")
            self._connection_alive = False
            print("Milvus connection closed.")
        except Exception as e:
            print(f"❌ Error closing Milvus connection: {e}")

    def disconnect(self):
        """
        Alias for close() method.
        """
        self.close() 
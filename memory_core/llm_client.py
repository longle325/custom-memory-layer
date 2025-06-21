import os
import json
import re
import time
import functools
import hashlib
from typing import Dict, Any, Optional, List
import google.generativeai as genai

class LLMClient:
    """
    A client for interacting with Google's Gemini models for text generation.

    This client handles both regular text generation and structured JSON responses
    for memory management and fact extraction tasks with performance optimizations.
    """

    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        """
        Initializes the LLMClient.

        Args:
            api_key (str, optional): The Google API key. If not provided, it will
                                     be read from the GOOGLE_API_KEY environment variable.
            model (str, optional): The name of the generative model to use.
                                   Defaults to "gemini-1.5-flash" for fast responses.
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        
        # Performance optimizations
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
        self._max_retries = 3
        self._base_delay = 1.0

    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate a cache key for the request"""
        # Create a hash of the prompt and parameters
        cache_data = f"{prompt}:{sorted(kwargs.items())}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if time.time() - cached_data["timestamp"] < self._cache_ttl:
                return cached_data["response"]
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache the response with timestamp"""
        self._cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }

    @staticmethod
    def _retry_operation(max_retries: int = None, base_delay: float = None):
        """Decorator for retrying failed operations with exponential backoff"""
        max_retries = max_retries or 3
        base_delay = base_delay or 1.0
        
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
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            time.sleep(delay)
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

    def _extract_json_from_response(self, response_text: str) -> str:
        """
        Extract JSON from response text, handling various formats with improved error handling.
        
        Args:
            response_text (str): The raw response text from the LLM.
            
        Returns:
            str: Extracted JSON string.
        """
        if not response_text:
            return None
            
        # Remove any markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # First, try to parse the entire response as JSON (handles arrays and objects)
        try:
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            pass
        
        # Find JSON object (curly braces)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx + 1]
            # Validate that it's actually JSON
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # Find JSON array (square brackets)
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx + 1]
            # Validate that it's actually JSON
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found, try to construct a basic one
        print(f"Warning: Could not extract valid JSON from: {response_text[:100]}...")
        return None

    @_time_operation("generate_response")
    @_retry_operation(max_retries=3, base_delay=1.0)
    def generate_response(self, prompt: str, is_json: bool = False, temperature: float = 0.1) -> str:
        """
        Generates a response from the LLM based on the given prompt with caching and retry logic.

        Args:
            prompt (str): The input prompt for the model.
            is_json (bool): Whether to expect and validate JSON response format.
            temperature (float): Controls randomness in the response (0.0 to 1.0).

        Returns:
            str: The generated response from the model.

        Raises:
            Exception: If there is an error calling the API or parsing JSON.
        """
        # Check cache first (only for non-JSON requests to avoid caching structured data)
        if not is_json:
            cache_key = self._get_cache_key(prompt, temperature=temperature)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response

        try:
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }

            # Add JSON format instruction to prompt if needed
            if is_json:
                enhanced_prompt = f"{prompt}\n\nIMPORTANT: You must respond with ONLY valid JSON. Do not include any explanations, markdown formatting, or additional text. Start your response with {{ and end with }}."
            else:
                enhanced_prompt = prompt

            response = self.model.generate_content(
                enhanced_prompt,
                generation_config=generation_config
            )

            response_text = response.text.strip()

            # Validate JSON if requested
            if is_json:
                try:
                    # First try to parse as-is
                    json.loads(response_text)
                    return response_text
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    extracted_json = self._extract_json_from_response(response_text)
                    if extracted_json:
                        return extracted_json
                    else:
                        raise Exception(f"Could not extract valid JSON from response: {response_text[:200]}")

            # Cache non-JSON responses
            if not is_json:
                cache_key = self._get_cache_key(prompt, temperature=temperature)
                self._cache_response(cache_key, response_text)

            return response_text

        except Exception as e:
            print(f"An error occurred while generating the response: {e}")
            raise

    @_time_operation("extract_facts")
    @_retry_operation(max_retries=2)
    def extract_facts(self, text: str) -> Dict[str, Any]:
        """
        Extracts structured facts from text with improved error handling.

        Args:
            text (str): The input text to extract facts from.

        Returns:
            Dict[str, Any]: A dictionary containing extracted facts.
        """
        from .prompts import FACT_EXTRACTION_PROMPT
        
        # Check cache first
        cache_key = self._get_cache_key("extract_facts", text=text)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            try:
                return json.loads(cached_response)
            except json.JSONDecodeError:
                pass  # Continue with fresh extraction if cached response is invalid
        
        # Create a more explicit prompt
        prompt = f"""{FACT_EXTRACTION_PROMPT}

Input: {text}

Remember: Respond with ONLY valid JSON in this exact format: {{"facts": ["fact1", "fact2"]}}"""
        
        try:
            response_text = self.generate_response(prompt, is_json=True)
            result = json.loads(response_text)
            
            # Ensure the result has the expected structure
            if "facts" not in result:
                result = {"facts": []}
            elif not isinstance(result["facts"], list):
                result["facts"] = []
            
            # Cache the result
            self._cache_response(cache_key, response_text)
                
            return result
        except Exception as e:
            print(f"Fact extraction failed: {e}")
            # Return empty facts as fallback
            fallback_result = {"facts": []}
            self._cache_response(cache_key, json.dumps(fallback_result))
            return fallback_result

    @_time_operation("update_memory")
    @_retry_operation(max_retries=2)
    def update_memory(self, new_facts: list, existing_memories: list) -> Dict[str, Any]:
        """
        Determines memory update actions based on new facts and existing memories.

        Args:
            new_facts (list): List of new facts extracted from input.
            existing_memories (list): List of existing memory objects.

        Returns:
            Dict[str, Any]: A dictionary containing memory update actions.
        """
        from .prompts import MEMORY_UPDATE_PROMPT
        
        # Check cache first
        cache_key = self._get_cache_key("update_memory", new_facts=str(new_facts), existing_memories=str(existing_memories))
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            try:
                return json.loads(cached_response)
            except json.JSONDecodeError:
                pass  # Continue with fresh processing if cached response is invalid
        
        # Create a more explicit prompt with proper formatting
        formatted_prompt = MEMORY_UPDATE_PROMPT.format(
            existing_memories=existing_memories,
            new_facts=new_facts
        )
        
        prompt = f"""{formatted_prompt}

Remember: Respond with ONLY valid JSON in this exact format:
{{
  "memory": [
    {{
      "event": "ADD|UPDATE|DELETE|NONE",
      "text": "memory text",
      "id": "memory_id (for UPDATE/DELETE)"
    }}
  ]
}}"""
        
        try:
            response_text = self.generate_response(prompt, is_json=True)
            result = json.loads(response_text)
            
            # Ensure the result has the expected structure
            if "memory" not in result:
                result = {"memory": []}
            elif not isinstance(result["memory"], list):
                result["memory"] = []
            
            # Cache the result
            self._cache_response(cache_key, response_text)
            
            return result
        except Exception as e:
            print(f"Memory update processing failed: {e}")
            # Return empty memory actions as fallback
            fallback_result = {"memory": []}
            self._cache_response(cache_key, json.dumps(fallback_result))
            return fallback_result

    @_time_operation("extract_graph_entities")
    @_retry_operation(max_retries=2)
    def extract_graph_entities(self, text: str) -> Dict[str, Any]:
        """
        Extracts entities and relationships for graph memory with improved error handling.

        Args:
            text (str): The input text to extract entities from.

        Returns:
            Dict[str, Any]: A dictionary containing extracted entities and relationships.
        """
        from .prompts import GRAPH_EXTRACTION_PROMPT
        
        # Check cache first
        cache_key = self._get_cache_key("extract_graph_entities", text=text)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            try:
                return json.loads(cached_response)
            except json.JSONDecodeError:
                pass  # Continue with fresh extraction if cached response is invalid
        
        prompt = f"""{GRAPH_EXTRACTION_PROMPT}

Input: {text}

Remember: Respond with ONLY valid JSON in this exact format:
{{
  "nodes": [
    {{
      "id": "entity_id",
      "label": "entity_type",
      "properties": {{"name": "entity_name", "description": "entity_description"}}
    }}
  ],
  "edges": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "label": "relationship_type",
      "properties": {{"description": "relationship_description"}}
    }}
  ]
}}"""
        
        try:
            response_text = self.generate_response(prompt, is_json=True)
            result = json.loads(response_text)
            
            # Ensure the result has the expected structure
            if "nodes" not in result:
                result["nodes"] = []
            elif not isinstance(result["nodes"], list):
                result["nodes"] = []
                
            if "edges" not in result:
                result["edges"] = []
            elif not isinstance(result["edges"], list):
                result["edges"] = []
            
            # Cache the result
            self._cache_response(cache_key, response_text)
            
            return result
        except Exception as e:
            print(f"Graph entity extraction failed: {e}")
            # Return empty entities as fallback
            fallback_result = {"nodes": [], "edges": []}
            self._cache_response(cache_key, json.dumps(fallback_result))
            return fallback_result

    def clear_cache(self):
        """Clear the response cache"""
        self._cache.clear()
        print("LLM client cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_ttl": self._cache_ttl,
            "max_retries": self._max_retries
        } 
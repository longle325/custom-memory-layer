import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import sys
import uuid
import hashlib
import time
import functools
from typing import Dict, Any, Optional
from prompts import SYSTEM_PROMPT_TEMPLATE
sys.path.insert(0, "memory_core")

from memory_core import Memory, LLMClient
from config.settings import get_settings

dotenv_path ='.env'
load_dotenv(dotenv_path, override=True)
settings = get_settings()


# Streamlit page configuration
st.set_page_config(
    page_title="Custom Memory Chat Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        self.metrics[operation] = {"start": time.time()}
    
    def end_timer(self, operation: str):
        if operation in self.metrics:
            self.metrics[operation]["duration"] = time.time() - self.metrics[operation]["start"]
            self.metrics[operation]["end"] = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        return {k: v.get("duration", 0) for k, v in self.metrics.items()}

# Global performance monitor
performance_monitor = PerformanceMonitor()


@st.cache_resource
def get_memory():
    """Initialize and cache the custom memory system with connection pooling"""
    try:
        performance_monitor.start_timer("memory_init")
        memory = Memory(
            google_api_key=settings.google_api_key,
            milvus_uri=settings.milvus_uri,
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password
        )
        performance_monitor.end_timer("memory_init")
        return memory
    except Exception as e:
        st.error(f"Failed to initialize memory system: {e}")
        return None


@st.cache_resource
def get_llm_client():
    """Initialize and cache the LLM client with retry logic"""
    try:
        performance_monitor.start_timer("llm_init")
        client = LLMClient(api_key=settings.google_api_key)
        performance_monitor.end_timer("llm_init")
        return client
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        return None

# Get cached resources
memory = get_memory()
llm_client = get_llm_client()

def generate_user_id(user_name):
    """Generate a deterministic user ID based on the user name"""

    name_hash = hashlib.md5(user_name.lower().strip().encode()).hexdigest()
    namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, "custom-memory-app.com")
    user_uuid = uuid.uuid5(namespace_uuid, name_hash)
    
    return str(user_uuid)


def create_user_session(user_name):
    """Create a new user session"""
    user_id = generate_user_id(user_name)
    st.session_state.authenticated = True
    st.session_state.user = {
        "id": user_id,
        "name": user_name
    }
    st.rerun()

def sign_out():
    """Sign out the current user"""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.logout_requested = True

def build_memory_context(search_results, max_length=2000):
    """Build memory context from search results with truncation optimization"""
    memory_context = ""
    
    if search_results.get("vector_results"):
        vector_memories = search_results["vector_results"]
        memory_context += "Vector Memories:\n"
        for i, mem in enumerate(vector_memories[:3], 1):
            text = mem.get('text', 'N/A')
            if len(text) > 300:
                text = text[:297] + "..."
            memory_context += f"{i}. {text}\n"
    
    if search_results.get("graph_results"):
        graph_memories = search_results["graph_results"]
        memory_context += "\nGraph Memories:\n"
        for i, mem in enumerate(graph_memories[:3], 1):
            data = mem.get('data', 'N/A')
            if len(data) > 300:
                data = data[:297] + "..."
            memory_context += f"{i}. {data}\n"
    
    # Ensure total context doesn't exceed max_length
    if len(memory_context) > max_length:
        memory_context = memory_context[:max_length-3] + "..."
    
    return memory_context if memory_context else "No relevant memories found."

def chat_with_custom_memory(message, user_id):
    """Chat function using the custom memory system with performance monitoring"""
    if not memory or not llm_client:
        return "Memory system is not available. Please check your configuration."
    
    try:
        # 1. SEARCH for relevant memories from previous conversations
        performance_monitor.start_timer("memory_search")
        search_results = memory.search(query=message, user_id=user_id, limit=settings.max_memory_retrieval)
        performance_monitor.end_timer("memory_search")
        
        # 2. Prepare context from memories
        performance_monitor.start_timer("context_build")
        memory_context = build_memory_context(search_results)
        performance_monitor.end_timer("context_build")
        
        # 3. Create system prompt with memory context
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            memory_context=memory_context,
            message=message
        )

        # 4. Generate response using the LLM client with retry logic
        performance_monitor.start_timer("llm_generation")
        response = llm_client.generate_response(system_prompt, temperature=0.7)
        performance_monitor.end_timer("llm_generation")
        
        # 5. ADD the complete conversation to memory AFTER response (batch operation)
        performance_monitor.start_timer("memory_add")
        full_conversation = f"User: {message}\nAssistant: {response}"
        memory.add(full_conversation, user_id=user_id)
        performance_monitor.end_timer("memory_add")
        
        return response
        
    except Exception as e:
        st.error(f"Error in chat function: {e}")
        return f"I encountered an error while processing your request: {str(e)}"

def clear_user_memories(user_id):
    """Clear all memories for a specific user"""
    if not memory:
        return False
    
    try:
        performance_monitor.start_timer("memory_clear")
        result = memory.delete_user_data(user_id)
        performance_monitor.end_timer("memory_clear")
        return result
    except Exception as e:
        st.error(f"Error clearing memories: {e}")
        return False

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None

# Initialize session state
initialize_session_state()

# Check for logout flag and clear it after processing
if st.session_state.get("logout_requested", False):
    st.session_state.logout_requested = False
    st.rerun()

with st.sidebar:
    st.title("🧠 Custom Memory Chat")
    
    if not st.session_state.authenticated:
        st.subheader("Start Chat Session")
        user_name = st.text_input("Enter your name to start chatting", key="user_name")
        start_button = st.button("Start Session")
        
        if start_button:
            if user_name.strip():
                create_user_session(user_name.strip())
            else:
                st.warning("Please enter your name.")
    else:
        user = st.session_state.user
        if user:
            st.success(f"Chatting as: {user['name']}")
            st.button("End Session", on_click=sign_out)
            
            # Display user information
            st.subheader("Your Profile")
            st.write(f"**User ID:** `{user['id']}`")
            st.write(f"**Name:** {user['name']}")
            
            # Memory management options
            st.subheader("Memory Management")
            
            # Clear memories
            if st.button("Clear All Memories"):
                result = clear_user_memories(user['id'])
                if result:
                    st.success("All memories cleared!")
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("Failed to clear memories.")
            
            # System status
            st.subheader("System Status")
            if memory:
                if memory.graph_enabled:
                    st.success("✅ Graph Memory: Enabled")
                else:
                    st.info("ℹ️ Graph Memory: Disabled (No Neo4j config)")
                st.success("✅ Vector Memory: Active")
            else:
                st.error("❌ Memory System: Not Available")
            
            # Performance metrics (collapsible)
            with st.expander("Performance Metrics"):
                metrics = performance_monitor.get_metrics()
                if metrics:
                    for operation, duration in metrics.items():
                        st.write(f"{operation}: {duration:.3f}s")
                else:
                    st.write("No metrics available yet")


if st.session_state.authenticated and st.session_state.user:
    user_id = st.session_state.user['id']
    
    st.title("Chat with Custom Memory-Powered AI")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get AI response with custom memory
        with st.spinner("Thinking with memory..."):
            ai_response = chat_with_custom_memory(user_input, user_id)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Display AI response
        with st.chat_message("assistant"):
            st.write(ai_response)
else:
    st.title("Welcome to Custom Memory Chat Assistant")
    st.write("Enter your name to start chatting with the custom memory-powered AI assistant.")

    st.subheader("Advanced Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 Vector Memory")
        st.write("Semantic search and storage using Milvus vector database.")
    
    with col2:
        st.markdown("### 🕸️ Graph Memory")
        st.write("Entity and relationship storage using Neo4j graph database.")
    
    with col3:
        st.markdown("### 🔄 Smart Updates")
        st.write("Intelligent memory updates with conflict resolution.")

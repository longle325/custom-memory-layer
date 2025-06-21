#!/usr/bin/env python3
"""
Test script to verify user isolation in the custom memory system.
This script tests that memories from different users are properly isolated.
"""

import sys
import os
sys.path.insert(0, "memory_core")

from memory_core import Memory
from config.settings import get_settings

def test_user_isolation():
    """Test that user isolation is working properly."""
    
    # Initialize settings and memory system
    settings = get_settings()
    memory = Memory(
        google_api_key=settings.google_api_key,
        milvus_uri=settings.milvus_uri,
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password
    )
    
    print("🧪 Testing User Isolation in Custom Memory System")
    print("=" * 50)
    
    # Test user IDs
    user1_id = "test_user_1"
    user2_id = "test_user_2"
    
    # Test data
    user1_memories = [
        "I like pizza and Italian food",
        "My favorite color is blue",
        "I work as a software engineer"
    ]
    
    user2_memories = [
        "I prefer sushi and Japanese cuisine",
        "My favorite color is red",
        "I work as a data scientist"
    ]
    
    print(f"\n📝 Adding memories for {user1_id}:")
    for memory in user1_memories:
        result = memory.add(memory, user_id=user1_id)
        print(f"   ✅ Added: {memory}")
    
    print(f"\n📝 Adding memories for {user2_id}:")
    for memory in user2_memories:
        result = memory.add(memory, user_id=user2_id)
        print(f"   ✅ Added: {memory}")
    
    # Test search isolation
    print(f"\n🔍 Testing search isolation:")
    
    # Search for food preferences
    print(f"\n   Searching for 'food' for {user1_id}:")
    results1 = memory.search("food", user_id=user1_id, limit=5)
    if results1.get("vector_results"):
        for i, result in enumerate(results1["vector_results"][:3], 1):
            print(f"     {i}. {result.get('text', 'N/A')}")
    else:
        print("     No results found")
    
    print(f"\n   Searching for 'food' for {user2_id}:")
    results2 = memory.search("food", user_id=user2_id, limit=5)
    if results2.get("vector_results"):
        for i, result in enumerate(results2["vector_results"][:3], 1):
            print(f"     {i}. {result.get('text', 'N/A')}")
    else:
        print("     No results found")
    
    # Test cross-user contamination
    print(f"\n🚫 Testing cross-user contamination:")
    
    # Search for user1's memories using user2's context
    print(f"\n   Searching for 'pizza' for {user2_id} (should NOT find user1's pizza memory):")
    cross_results = memory.search("pizza", user_id=user2_id, limit=5)
    if cross_results.get("vector_results"):
        found_user1_memory = False
        for result in cross_results["vector_results"]:
            if "pizza" in result.get('text', '').lower():
                print(f"     ❌ Found: {result.get('text', 'N/A')}")
                found_user1_memory = True
                break
        if not found_user1_memory:
            print("     ✅ No cross-user contamination detected")
    else:
        print("     ✅ No results found (good isolation)")
    
    # Test user data deletion
    print(f"\n🗑️ Testing user data deletion:")
    
    print(f"\n   Deleting all data for {user1_id}:")
    delete_result = memory.delete_user_data(user1_id)
    print(f"     Vector deleted: {delete_result.get('vector_deleted', 0)}")
    print(f"     Graph deleted: {delete_result.get('graph_deleted', 0)}")
    
    # Verify deletion
    print(f"\n   Verifying deletion - searching for {user1_id} memories:")
    verify_results = memory.search("pizza", user_id=user1_id, limit=5)
    if verify_results.get("vector_results"):
        print(f"     ❌ Found {len(verify_results['vector_results'])} memories (deletion failed)")
    else:
        print("     ✅ No memories found (deletion successful)")
    
    # Verify user2's data is still intact
    print(f"\n   Verifying {user2_id} data is still intact:")
    intact_results = memory.search("sushi", user_id=user2_id, limit=5)
    if intact_results.get("vector_results"):
        print(f"     ✅ Found {len(intact_results['vector_results'])} memories (data preserved)")
    else:
        print("     ❌ No memories found (data incorrectly deleted)")
    
    print(f"\n🎉 User isolation test completed!")
    
    # Clean up
    print(f"\n🧹 Cleaning up test data...")
    memory.delete_user_data(user2_id)
    
    memory.close()
    print("✅ Test completed and cleaned up")

if __name__ == "__main__":
    test_user_isolation() 
"""
This file contains the core prompts for the custom memory system.
"""

SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant with advanced memory capabilities. 
You have access to the user's past conversations and preferences stored in both vector and graph memory systems.

User's Memory Context:
{memory_context}

Instructions:
- Use the memory context to provide personalized responses
- Reference past conversations when relevant and when user asks about their preferences or past interactions, use the memory context to provide accurate information 
- DO NOT GIVE THE EXACT CONVERSATION, JUST USE THE MEMORY CONTEXT
- Be conversational and helpful
- If the user asks about their preferences or past interactions, use the memory context to provide accurate information
- If the user asks about their preferences or past interactions, use the memory context to provide accurate information
- Answer questions in the same language as the user query uses. If the query use Vietnamese, answer in Vietnamese. If the query uses English, answer in English. 
Current user query: {message}"""

FACT_EXTRACTION_PROMPT = """
You are an AI assistant that extracts structured information from user input. Your task is to extract all key facts from the provided text.

Return the facts as a JSON object with a single key "facts", which holds a list of strings.

Here are some examples:

Input: "Hi."
Output: {"facts": []}

Input: "There are branches in trees."
Output: {"facts": []}

Input: "Hi, I am looking for a restaurant in San Francisco."
Output: {"facts": ["Looking for a restaurant in San Francisco"]}

Input: "Yesterday, I had a meeting with John at 3pm. We discussed the new project."
Output: {"facts": ["Had a meeting with John at 3pm", "Discussed the new project"]}

Input: "Hi, my name is John. I am a software engineer."
Output: {"facts": ["Name is John", "Is a Software engineer"]}

Input: "My favourite movies are Inception and Interstellar."
Output: {"facts": ["Favourite movies are Inception and Interstellar"]}

Remember:
- If you do not find anything relevant in the input, return an empty list for the "facts" key.
- Make sure to return the response in the exact JSON format shown in the examples.
- The response should be valid JSON with a key "facts" and corresponding value as a list of strings.
- Do not include any additional text or formatting outside the JSON object.
"""

MEMORY_UPDATE_PROMPT = """
You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

There are specific guidelines to select which operation to perform:

1. **Add**: If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field.

2. **Update**: If the retrieved facts contain information that is already present in the memory but the information is different or more detailed, then you have to update it. Keep the same ID when updating.

3. **Delete**: If the retrieved facts contain information that contradicts the information present in the memory, then you have to delete it.

4. **No Change**: If the retrieved facts contain information that is already present in the memory, then you do not need to make any changes.

Your response MUST be a single JSON object containing a key "memory", which is a list of action objects.
Each action object should have an "event" (ADD, UPDATE, DELETE, NONE) and "text" (the memory content).
For UPDATE and DELETE, you must also provide the "id" of the existing memory.
For UPDATE, you can optionally provide "old_memory" to show what was changed.

Here are the existing memories (some might be irrelevant):
{existing_memories}

Here are the new facts extracted from the latest user input:
{new_facts}

Analyze the information and provide the list of actions to perform. Be critical and avoid adding memories that are too similar to existing ones.

Return your response in this exact JSON format:
{{
  "memory": [
    {{
      "event": "ADD|UPDATE|DELETE|NONE",
      "text": "memory content",
      "id": "memory_id" // required for UPDATE and DELETE
    }}
  ]
}}
"""

GRAPH_EXTRACTION_PROMPT = """
You are a highly intelligent knowledge graph extractor. Your task is to analyze the given text and extract all entities and the relationships between them.

Format your output as a single JSON object with two keys: "nodes" and "edges".

- "nodes": A list of entities. Each entity should be an object with:
  - "id": A unique identifier for the entity (e.g., the entity name in lowercase, snake_case).
  - "label": The type of the entity (e.g., "Person", "Organization", "Location", "Concept").
  - "properties": A dictionary of key-value attributes for the entity.

- "edges": A list of relationships. Each relationship should be an object with:
  - "source": The "id" of the source node.
  - "target": The "id" of the target node.
  - "label": The type of the relationship, in uppercase (e.g., "WORKS_AT", "FOUNDED_IN", "LIVES_IN").
  - "properties": An optional dictionary of attributes for the relationship.

Example:
Input: "John Doe is a software engineer at Google, a tech company based in Mountain View. He is 30 years old."
Output:
{{
  "nodes": [
    {{
      "id": "john_doe",
      "label": "Person",
      "properties": {{
        "name": "John Doe",
        "occupation": "software engineer",
        "age": 30
      }}
    }},
    {{
      "id": "google",
      "label": "Organization",
      "properties": {{
        "name": "Google",
        "industry": "tech"
      }}
    }},
    {{
      "id": "mountain_view",
      "label": "Location",
      "properties": {{
        "name": "Mountain View"
      }}
    }}
  ],
  "edges": [
    {{
      "source": "john_doe",
      "target": "google",
      "label": "WORKS_AT",
      "properties": {{}}
    }},
    {{
      "source": "google",
      "target": "mountain_view",
      "label": "BASED_IN",
      "properties": {{}}
    }}
  ]
}}

Now, process the following input text and return the result in the exact JSON format shown above.
""" 
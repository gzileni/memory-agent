PARSER_PROMPT: str = """
You are a precise graph relationship extractor.
Extract all relationships from the text and format
them as a JSON object with this exact structure:
{{
    "graph": [
        {{
            "node": "Person/Entity",
            "target_node": "Related Entity",
            "relationship": "Type of Relationship"
        }},
        ...more relationships...
    ]
}}
Include ALL relationships mentioned in the text, including
implicit ones. Be thorough and precise.
"""

AGENT_PROMPT: str = (
    "You are an intelligent assistant with access to the "
    "following knowledge graph:\n\n"
    "Nodes: \"{nodes_str}\"\n\n"
    "Edges: \"{edges_str}\"\n\n"
    "Using this graph, Answer the following question:\n\n"
    "User Query: \"{user_query}\""
)

import streamlit as st
import boto3
from botocore.exceptions import ClientError
import json
from bedrock_utils import query_knowledge_base, generate_response, valid_prompt


# Streamlit UI
st.title("Bedrock Chat Application")

# Sidebar for configurations
st.sidebar.header("Configuration")
model_id = st.sidebar.selectbox("Select LLM Model", ["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-5-sonnet-20240620-v1:0"])
kb_id = st.sidebar.text_input("Knowledge Base ID", "AV2ZI5BXN0")
temperature = st.sidebar.select_slider("Temperature", [i/10 for i in range(0,11)],1)
top_p = st.sidebar.select_slider("Top_P", [i/1000 for i in range(0,1001)], 1)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if valid_prompt(prompt, model_id):
        # Query Knowledge Base
        kb_results = query_knowledge_base(prompt, kb_id)
        
        # Prepare context from Knowledge Base results
        if kb_results:
            context_parts = []
            for i, result in enumerate(kb_results, 1):
                context_parts.append(f"Source {i} (Confidence: {result['score']:.2f}):\n{result['content']}")
            context = "\n\n".join(context_parts)
            
            # Generate response using LLM with context
            full_prompt = f"Context from heavy machinery documentation:\n\n{context}\n\nUser Question: {prompt}"
            response = generate_response(full_prompt, model_id, temperature, top_p)
        else:
            # No relevant context found
            response = "I couldn't find specific information about that in our heavy machinery documentation. Could you try rephrasing your question or ask about a different aspect of heavy machinery?"
    else:
        response = "I'm unable to answer this, please try again"
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
import streamlit as st
from database import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    """Generates a new UUID for a chat thread, ensuring it's a string."""
    return str(uuid.uuid4())

def load_conversation(thread_id):
    """Retrieves messages from the LangGraph backend for a given thread."""
    try:
        # State contains 'messages' and 'context'
        state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
        
        # We only need the 'messages' list for display
        if state and 'messages' in state.values:
            return state.values['messages']
        else:
            return []
    except Exception as e:
        st.error(f"Error loading conversation for thread {thread_id}: {e}")
        return []

def select_thread(thread_id_to_load):
    """
    CALLBACK FUNCTION: Loads the selected thread's history into session state.
    """
    st.session_state['thread_id'] = str(thread_id_to_load)
    messages = load_conversation(thread_id_to_load)

    temp_messages = []
    for msg in messages:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        content = getattr(msg, 'content', str(msg))
        temp_messages.append({'role': role, 'content': content})

    st.session_state['message_history'] = temp_messages


def add_thread(thread_id):
    """Adds a new thread_id to the list of chat threads if it doesn't exist."""
    str_thread_id = str(thread_id)
    if str_thread_id not in [str(t) for t in st.session_state['chat_threads']]:
        st.session_state['chat_threads'].append(str_thread_id)

def reset_chat():
    """Resets the current chat session and starts a new thread."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.rerun() 

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads() or []

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI (FIXED BUTTONS) *****************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('➕ New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

# Loop through threads to create buttons with callbacks
for thread_id in st.session_state['chat_threads'][::-1]:
    
    is_current_thread = str(thread_id) == str(st.session_state['thread_id'])
    button_label = f"➡️ {str(thread_id)}" if is_current_thread else str(thread_id)

    # CRITICAL FIX: Use unique 'key' and 'on_click' callback
    st.sidebar.button(
        button_label,
        key=f"thread_btn_{thread_id}", 
        on_click=select_thread,
        args=(thread_id,),
        help="Click to load this conversation."
    )


# **************************************** Main UI ************************************

st.title(f"Current Thread: {str(st.session_state['thread_id'])[:8]}...")

# Display the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # 1. Add user message to history and display
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {'configurable': {'thread_id': str(st.session_state['thread_id'])}}
    
    full_ai_response = ""
    
    with st.chat_message('assistant'):
        
        message_placeholder = st.empty()
        
        # 2. Iterate through the stream chunks (RAG Compatible Streaming)
        # Input state MUST include the 'context' key
        for chunk in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)], 'context': ""}, 
            config=CONFIG,
        ):
            # Check for the output from the final 'chat' node
            if "chat" in chunk:
                # The node returns an AIMessage object
                new_content = chunk["chat"]['messages'][0].content
                full_ai_response += new_content
                message_placeholder.markdown(full_ai_response + "▌") 

        # 3. Display the final, complete response
        message_placeholder.markdown(full_ai_response)
    
    # 4. Save the final response to session history
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_ai_response})
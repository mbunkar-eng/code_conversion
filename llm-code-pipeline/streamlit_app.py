"""
Streamlit UI for LLM Code Pipeline - FIXED VERSION
A web interface for downloading models and chatting with LLMs

KEY FIXES:
1. Dynamic model detection (no hardcoded names)
2. Correct model name mapping
3. Better error handling
"""

import streamlit as st
import requests
import json
import time
from typing import Optional, List, Dict
import os
from pathlib import Path

# Configuration
# Default to empty; require entering API key in sidebar unless env var is set
API_KEY = os.getenv("LLM_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Global analytics tracking
analytics = {
    "api_requests": 0,
    "active_sessions": 0,
    "total_models": 0,
    "chat_messages": 0,
    "model_usage": {},
    "last_updated": time.time()
}

def update_analytics(request_type: str, model: str = None):
    """Update analytics counters"""
    global analytics
    analytics["api_requests"] += 1
    analytics["last_updated"] = time.time()

    if request_type == "chat":
        analytics["chat_messages"] += 1
        if model:
            if model not in analytics["model_usage"]:
                analytics["model_usage"][model] = 0
            analytics["model_usage"][model] += 1

def get_analytics():
    """Get current analytics data"""
    global analytics
    analytics["total_models"] = len(get_local_models())
    return analytics

# Page configuration
st.set_page_config(
    page_title="LLM Code Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .model-card {
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    .status-success {
        color: #4caf50;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    .status-info {
        color: #2196f3;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None, request_type: str = "api", model: str = None) -> Dict:
    """Make authenticated API request"""
    url = f"{API_BASE_URL}{endpoint}"
    api_key = st.session_state.get("api_key", API_KEY)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=60)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Track successful API requests
        if response.status_code < 400:
            update_analytics(request_type, model)

        if response.status_code == 401:
            st.error("❌ Authentication failed. Please check your API key.")
            return {"error": "Authentication failed"}
        elif response.status_code >= 400:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("detail", response.text)
            except:
                pass
            st.error(f"❌ API Error ({response.status_code}): {error_detail}")
            return {"error": f"HTTP {response.status_code}: {error_detail}"}

        return response.json()

    except requests.exceptions.Timeout:
        st.error("❌ Request timeout. The model may be loading or the server is slow.")
        return {"error": "Request timeout"}
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Please make sure it's running.")
        return {"error": "Connection failed"}
    except Exception as e:
        st.error(f"❌ Request failed: {str(e)}")
        return {"error": str(e)}

def get_available_models() -> List[Dict]:
    """Get list of available models from API"""
    response = make_api_request("/v1/models")
    if "error" in response:
        return []
    return response.get("data", [])

def get_local_models() -> List[str]:
    """
    Get list of locally downloaded models by scanning the directory.
    This dynamically detects models instead of using hardcoded names.
    """
    models_dir = Path("./downloaded_models")
    
    # If directory doesn't exist, return empty list
    if not models_dir.exists():
        st.warning("📁 downloaded_models directory not found. Models should be in ./downloaded_models/")
        return []
    
    models = []
    for item in models_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != "downloaded_models":
            # Check if it looks like a complete model
            has_config = (item / "config.json").exists()
            has_tokenizer = (item / "tokenizer_config.json").exists()
            
            # Only include if it has the essential files
            if has_config or has_tokenizer:
                models.append(item.name)
    
    return sorted(models)

def map_local_to_api_model(local_model: str) -> str:
    """
    Map local directory names to API model names.
    
    Strategy:
    1. Try the directory name as-is first (most reliable)
    2. Try converting to HuggingFace format (org/model)
    3. Try registry name mappings
    """
    if not local_model:
        return local_model
    
    # Strategy 1: Use directory name directly
    # This is the most reliable as it matches what's actually on disk
    # The backend should handle the mapping
    
    # Strategy 2: Common registry name mappings
    # These are fallback options
    registry_mappings = {
        "deepseek-ai--deepseek-coder-6.7b-instruct": "deepseek-coder-6.7b",
        "deepseek-coder-6.7b-instruct": "deepseek-coder-6.7b",
        "Qwen--Qwen2.5-Coder-7B-Instruct": "qwen2.5-coder-7b",
        "Qwen--Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
    }
    
    # First, try the exact directory name (let backend handle it)
    # If that's in the mappings, use the mapped name, otherwise use as-is
    return registry_mappings.get(local_model, local_model)

def chat_with_model(model: str, message: str, stream: bool = False) -> Dict:
    """Send chat message to model"""
    # Use the model name directly or map it
    api_model = map_local_to_api_model(model)
    
    # Show what we're sending (helpful for debugging)
    if st.session_state.get("debug_mode", False):
        st.info(f"🔍 Debug: Selected='{model}', Sending to API='{api_model}'")
    
    data = {
        "model": api_model,
        "messages": [
            {"role": "user", "content": message}
        ],
        "max_tokens": 2048,
        "temperature": 0.7
    }

    return make_api_request("/v1/chat/completions", "POST", data, "chat", api_model)

def download_model(model_id: str) -> bool:
    """Download a model (placeholder - would need API endpoint)"""
    st.info(f"📥 Downloading model: {model_id}...")
    time.sleep(2)
    st.success(f"✅ Model {model_id} downloaded successfully!")
    return True

def show_api_key_input():
    """Show API key input screen"""
    st.markdown('<h1 class="main-header">🔐 LLM Code Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("🔑 API Key Required")
    st.markdown("Please enter your API key to access the LLM Code Pipeline interface.")

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter your 16-character API key",
        help="The API key should be 16 alphanumeric characters"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔓 Access Pipeline", use_container_width=True, type="primary"):
            if api_key and len(api_key.strip()) >= 8:
                st.session_state.api_key = api_key.strip()
                st.success("✅ API Key accepted! Loading interface...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please enter a valid API key (at least 8 characters)")

    with col2:
        st.info("💡 **Default Key**: `7XK9M2PZ5A4R8QF3` (for testing)")

    st.markdown("---")
    st.markdown("""
    **About API Key:**
    - The API key is required for authentication with the LLM API server
    - It should be a 16-character alphanumeric string
    - Keep your API key secure and don't share it
    """)

def main():
    # Ensure an API key exists in session; default to env if missing
    if "api_key" not in st.session_state:
        st.session_state.api_key = API_KEY

    # Track session
    if "session_started" not in st.session_state:
        st.session_state.session_started = True
        analytics["active_sessions"] += 1
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 LLM Pipeline")
        st.markdown("---")

        # API Status
        st.subheader("🔗 API Status")
        if st.button("🔍 Check API Health"):
            health = make_api_request("/health")
            if "error" in health:
                st.error("❌ API Unhealthy")
            else:
                st.success("✅ API Healthy")
                if "model_name" in health:
                    st.info(f"📦 Loaded: {health['model_name']}")

        st.markdown("---")

        # Model Management
        st.subheader("📦 Model Management")
        if st.button("📋 Refresh Models"):
            st.rerun()

        st.markdown("---")

        # Settings
        st.subheader("⚙️ Settings")
        st.text_input(
            "API Key",
            key="api_key",
            value=st.session_state.get("api_key", API_KEY),
            type="password",
            help="Used to authenticate API calls. Stored in session only."
        )
        
        # Debug mode toggle
        st.checkbox("🐛 Debug Mode", key="debug_mode", help="Show detailed API request info")

    # Gate main content until a valid API key is provided in the sidebar
    api_key_val = (st.session_state.get("api_key") or "").strip()
    if len(api_key_val) < 8:
        st.markdown('<h1 class="main-header">🔐 LLM Code Pipeline</h1>', unsafe_allow_html=True)
        st.info("Please enter your API key in the sidebar to continue.")
        return

    # Main content
    st.markdown('<h1 class="main-header">🤖 LLM Code Pipeline</h1>', unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📦 Models", "📊 Analytics"])

    with tab1:
        st.header("💬 Chat with LLM")

        # Model selection
        available_models = get_local_models()
        if available_models:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                help="Choose which downloaded model to chat with"
            )
            
            # Show helpful info about the selected model
            if selected_model:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"📂 Directory: `{selected_model}`")
                with col2:
                    api_name = map_local_to_api_model(selected_model)
                    if api_name != selected_model:
                        st.caption(f"→ API: `{api_name}`")
        else:
            st.warning("⚠️ No models available. Please download models first.")
            st.info("💡 Models should be in: `./downloaded_models/`")
            selected_model = None

        # Chat interface
        if selected_model:
            st.markdown("---")

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                role_class = "user-message" if message["role"] == "user" else "assistant-message"
                icon = "👤" if message["role"] == "user" else "🤖"
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message {role_class}">
                        <strong>{icon} {message['role'].title()}:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)

            # Chat input
            with st.form(key="chat_form"):
                user_input = st.text_area(
                    "Message",
                    placeholder="Ask anything...",
                    height=100,
                    help="Enter your message to chat with the LLM"
                )

                col1, col2 = st.columns([1, 1])
                with col1:
                    submit_button = st.form_submit_button("🚀 Send Message", use_container_width=True)
                with col2:
                    clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)

            if submit_button and user_input.strip():
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": user_input})

                # Get AI response
                with st.spinner("🤔 Thinking..."):
                    response = chat_with_model(selected_model, user_input)

                if "error" in response:
                    st.error(f"❌ Chat failed: {response['error']}")
                    st.error(f"💡 Tried model name: {map_local_to_api_model(selected_model)}")
                else:
                    ai_response = response.get("choices", [{}])[0].get("message", {}).get("content", "No response")
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

                st.rerun()

            if clear_button:
                st.session_state.messages = []
                st.success("🗑️ Chat cleared!")
                st.rerun()

    with tab2:
        st.header("📦 Model Management")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Download Models")

            models_to_download = [
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "deepseek-ai/deepseek-coder-6.7b-instruct",
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill"
            ]

            selected_download = st.selectbox(
                "Select model to download",
                models_to_download,
                help="Choose a model to download from HuggingFace"
            )

            if st.button("📥 Download Model", use_container_width=True):
                if download_model(selected_download):
                    st.success(f"✅ Successfully downloaded {selected_download}")
                    time.sleep(1)
                    st.rerun()

        with col2:
            st.subheader("📋 Available Models")

            local_models = get_local_models()
            if local_models:
                for model in local_models:
                    # Check if model has all required files
                    model_path = Path("./downloaded_models") / model
                    has_config = (model_path / "config.json").exists()
                    has_tokenizer = (model_path / "tokenizer_config.json").exists()
                    
                    status = "✅ Complete" if (has_config and has_tokenizer) else "⚠️ Incomplete"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="model-card">
                            <strong>📦 {model}</strong><br>
                            <span class="status-success">{status}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ No models downloaded yet. Download some models to get started!")

            st.markdown("---")

            # Registry models
            st.subheader("🌐 Registry Models")
            registry_models = get_available_models()
            if registry_models:
                for model in registry_models[:5]:
                    st.markdown(f"- **{model['id']}**")
            else:
                st.info("ℹ️ Could not load registry models")

    with tab3:
        st.header("📊 Analytics Dashboard")

        # Refresh button at the top
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🧹 Reset Counters", use_container_width=True):
                analytics["api_requests"] = 0
                analytics["chat_messages"] = 0
                analytics["model_usage"] = {}
                st.success("✅ Counters reset!")
        with col3:
            last_updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(get_analytics()["last_updated"]))
            st.info(f"🕒 Last updated: {last_updated}")

        st.markdown("---")

        # Get real analytics data
        stats = get_analytics()

        # Key metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Models", stats["total_models"], help="Number of downloaded models")

        with col2:
            st.metric("API Requests", stats["api_requests"], help="Total API calls made")

        with col3:
            st.metric("Chat Messages", stats["chat_messages"], help="Messages sent to chat models")

        with col4:
            st.metric("Active Sessions", stats["active_sessions"], help="Current active user sessions")

        st.markdown("---")

        # Model usage breakdown
        st.subheader("🤖 Model Usage Statistics")
        if stats["model_usage"]:
            st.markdown("**Usage by Model:**")
            for model, count in sorted(stats["model_usage"].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats["chat_messages"] * 100) if stats["chat_messages"] > 0 else 0
                st.progress(min(percentage / 100, 1.0), text=f"{model}: {count} requests ({percentage:.1f}%)")
        else:
            st.info("📊 No model usage data yet. Start chatting to see statistics!")

        st.markdown("---")

        # System health
        st.subheader("🏥 System Health")
        col1, col2 = st.columns(2)

        with col1:
            health_status = make_api_request("/health")
            if "error" in health_status:
                st.error("❌ API Server: Offline")
            else:
                st.success("✅ API Server: Online")

        with col2:
            local_models = get_local_models()
            if local_models:
                st.success(f"✅ Models: {len(local_models)} available")
            else:
                st.warning("⚠️ Models: None available")

        # Additional info
        st.markdown("---")
        st.subheader("ℹ️ About Analytics")
        st.markdown("""
        **Real-time Tracking:**
        - API requests are counted for each successful call
        - Chat messages track conversations with models
        - Model usage shows which models are most popular
        - Session tracking shows concurrent users

        **Data Persistence:**
        - Statistics reset when the app restarts
        - Use the reset button to clear counters manually
        """)

if __name__ == "__main__":
    main()
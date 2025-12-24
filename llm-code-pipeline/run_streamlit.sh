#!/bin/bash
# Run Streamlit UI for LLM Code Pipeline

echo "🚀 Starting Streamlit UI for LLM Code Pipeline..."
echo "📱 UI will be available at: http://localhost:8501"
echo "🔗 Make sure the API server is running at http://localhost:8000"
echo ""

# Set environment variables
export LLM_API_KEY="7XK9M2PZ5A4R8QF3"

# Run Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import altair as alt
import time
from io import StringIO

# Page config
st.set_page_config(
    page_title="Acme Review Analyzer",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛒 Acme Product Review Analyzer</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Analytics Dashboard")
st.sidebar.info("Upload your product reviews CSV file and get instant AI-powered insights!")

# Check backend connection
def check_backend_connection():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Main app
def main():
    # Check if backend is running
    if not check_backend_connection():
        st.error("⚠️ Cannot connect to the backend API. Please make sure:")
        st.markdown("""
        1. The backend server is running: `uvicorn backend.main:app --reload`
        2. Ollama is running: `ollama serve`
        3. Mistral model is available: `ollama pull mistral`
        """)
        return

    st.success("✅ Backend API is connected and ready!")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with product reviews",
        type=["csv"],
        help="CSV should contain columns: product_id, product_name, review_text"
    )

    # Load sample data option
    if st.button("📋 Use Sample Data"):
        try:
            df = pd.read_csv("data/sample_reviews.csv")
            st.session_state.sample_data = df
            st.success("Sample data loaded successfully!")
        except FileNotFoundError:
            st.error("Sample data file not found. Please ensure data/sample_reviews.csv exists.")
            return

    # Process uploaded file or sample data
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif hasattr(st.session_state, 'sample_data'):
        df = st.session_state.sample_data

    if df is not None:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Validate required columns
        required_cols = ["product_id", "product_name", "review_text"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return

        # Analysis section
        if st.button("🚀 Analyze Reviews", type="primary"):
            analyze_reviews(df)

def analyze_reviews(df):
    """Analyze reviews and display results"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_reviews = len(df)
    
    # Process each review
    for idx, row in df.iterrows():
        try:
            status_text.text(f"Analyzing review {idx + 1} of {total_reviews}...")
            
            review_text = row["review_text"]
            
            # Call backend API
            response = requests.post(
                "http://localhost:8000/analyze/",
                data={"text": review_text},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                    "review_text": review_text,
                    "sentiment": data["sentiment"],
                    "topic": data["topic"],
                    "summary": data["summary"]
                })
            else:
                st.error(f"Error analyzing review {idx + 1}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error(f"Timeout analyzing review {idx + 1}")
        except Exception as e:
            st.error(f"Error analyzing review {idx + 1}: {str(e)}")
        
        # Update progress
        progress_bar.progress((idx + 1) / total_reviews)
        time.sleep(0.1)  # Small delay to prevent overwhelming the API
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("No reviews were successfully analyzed.")
        return
    
    # Create results DataFrame
    result_df = pd.DataFrame(results)
    
    # Display results
    st.success(f"✅ Analysis complete! Processed {len(results)} reviews.")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(results))
    with col2:
        positive_count = len(result_df[result_df["sentiment"].str.contains("Positive", case=False, na=False)])
        st.metric("Positive Reviews", positive_count)
    with col3:
        negative_count = len(result_df[result_df["sentiment"].str.contains("Negative", case=False, na=False)])
        st.metric("Negative Reviews", negative_count)
    with col4:
        unique_topics = result_df["topic"].nunique()
        st.metric("Unique Topics", unique_topics)
    
    # Results table
    st.subheader("📊 Analysis Results")
    st.dataframe(result_df, use_container_width=True)
    
    # Download button
    csv_data = result_df.to_csv(index=False)
    st.download_button(
        label="💾 Download Results as CSV",
        data=csv_data,
        file_name="review_analysis_results.csv",
        mime="text/csv"
    )
    
    # Visualizations
    display_visualizations(result_df)

def display_visualizations(df):
    """Display charts and visualizations"""
    st.subheader("📈 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        st.subheader("😊 Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        
        fig, ax = plt.subplots()
        colors = ['green', 'orange', 'red']
        sentiment_counts.plot(kind='bar', ax=ax, color=colors[:len(sentiment_counts)])
        ax.set_title("Review Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        # Topic distribution
        st.subheader("📌 Top Topics")
        topic_counts = df["topic"].value_counts().head(10)
        
        fig, ax = plt.subplots()
        topic_counts.plot(kind='barh', ax=ax)
        ax.set_title("Most Common Topics")
        ax.set_xlabel("Count")
        st.pyplot(fig)
    
    # Product-wise analysis
    if "product_name" in df.columns:
        st.subheader("🏷️ Product Analysis")
        product_sentiment = df.groupby(["product_name", "sentiment"]).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        product_sentiment.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title("Sentiment by Product")
        ax.set_xlabel("Product")
        ax.set_ylabel("Number of Reviews")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
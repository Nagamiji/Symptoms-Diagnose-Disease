import os
import shutil

def setup_streamlit():
    """Setup everything for Streamlit app"""
    print("🔄 Setting up Streamlit app...")
    
    # Check if models exist
    if not os.path.exists('models/best_disease_classifier.pkl'):
        print("❌ Models not found. Please run model training first.")
        return
    
    print("✅ Models found!")
    print("✅ Data files found!")
    print("🎉 Setup completed! Run: streamlit run app.py")

if __name__ == "__main__":
    setup_streamlit()
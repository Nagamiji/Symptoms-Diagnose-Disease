import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import sys

# Add src to path for custom modules
sys.path.append('src')

# Set Streamlit page configuration
st.set_page_config(page_title='Advanced Disease Prediction App', page_icon='🩺', layout='wide')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f7f9fc 0%, #e2e8f0 100%);
        color: #333;
        padding: 20px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .stButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stTextInput, .stMultiSelect {
        border-radius: 8px;
        border: 1px solid #ced4da;
        background: #fff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b;
        font-family: 'Arial', sans-serif;
    }
    .section {
        padding: 20px;
        margin: 10px 0;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
    }
    .section:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .symptom-chip {
        display: inline-block;
        background: #e0f2fe;
        color: #1e40af;
        padding: 5px 10px;
        border-radius: 12px;
        margin: 5px;
        font-size: 14px;
    }
    .confidence-high { color: #10b981; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
    .info-card-blue { background: #e0f2fe; color: #1e40af; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .info-card-yellow { background: #fef3c7; color: #92400e; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .info-card-green { background: #d1fae5; color: #065f46; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .info-card-red { background: #fee2e2; color: #991b1b; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .info-card-purple { background: #ede9fe; color: #5b21b6; padding: 15px; border-radius: 10px; margin: 10px 0; }
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_models():
    """Load the best model and encoders"""
    try:
        model = joblib.load('models/best_disease_classifier.pkl')
        st.success("✅ Best model loaded successfully!")
        disease_encoder = joblib.load('models/disease_encoder.pkl')
        symptom_severity_map = joblib.load('models/symptom_severity_map.pkl')
        st.success("✅ Encoders loaded successfully!")
        st.info(f"🎯 Using: {type(model).__name__} model")
        return model, disease_encoder, symptom_severity_map
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_data_files():
    """Load all data files"""
    try:
        description_df = pd.read_csv('data/raw/description.csv')
        precautions_df = pd.read_csv('data/raw/precautions_df.csv')
        medications_df = pd.read_csv('data/raw/medications.csv')
        diets_df = pd.read_csv('data/raw/diets.csv')
        workout_df = pd.read_csv('data/raw/workout_df.csv')
        st.success("✅ All data files loaded successfully!")
        try:
            doctor_df = pd.read_csv('data/raw/doctor.csv')
            st.success("✅ Doctor data loaded!")
        except:
            doctor_df = pd.DataFrame()
            st.warning("⚠️ Doctor data not found, skipping doctor information")
        return description_df, precautions_df, medications_df, diets_df, workout_df, doctor_df
    except Exception as e:
        st.error(f"❌ Error loading data files: {e}")
        return None, None, None, None, None, None

def create_input_vector(selected_symptoms, all_symptoms, symptom_severity_map):
    """Create input vector with symptom severity weights"""
    input_vector = np.zeros(len(all_symptoms))
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            weight = symptom_severity_map.get(symptom, 1)
            input_vector[index] = weight
    return input_vector.reshape(1, -1)

def predict_top_diseases(model, input_vector, disease_encoder, top_n=10):
    """Predict top N diseases with probabilities"""
    try:
        probabilities = model.predict_proba(input_vector)[0]
        top_n_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_n_probabilities = probabilities[top_n_indices]
        disease_names = disease_encoder.inverse_transform(top_n_indices)
        return list(zip(disease_names, top_n_probabilities))
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return []

def get_confidence_class(probability):
    """Get CSS class and emoji for confidence level"""
    if probability > 0.7:
        return "confidence-high", "🟢"
    elif probability > 0.3:
        return "confidence-medium", "🟡"
    else:
        return "confidence-low", "🔴"

# Main app
def main():
    # Hero section
    st.markdown(
        """
        <div class='hero-section'>
            <h1>🩺 Advanced Disease Prediction System</h1>
            <p>AI-powered disease prediction using advanced machine learning models. Select your symptoms to get started.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load models and data
    model, disease_encoder, symptom_severity_map = load_models()
    description_df, precautions_df, medications_df, diets_df, workout_df, doctor_df = load_data_files()

    if model is None or disease_encoder is None:
        st.error("🚫 Failed to load required models. Please check if models are trained.")
        st.stop()

    all_symptoms = list(symptom_severity_map.keys())

    # Sidebar with information
    with st.sidebar:
        st.markdown(
            """
            <div style='padding: 15px;'>
                <h2>ℹ️ About</h2>
                <p><strong>How it works:</strong></p>
                <ul>
                    <li>Select symptoms from the list</li>
                    <li>Get top disease predictions with confidence</li>
                    <li>View detailed information and recommendations</li>
                </ul>
                <p><strong>Model Info:</strong></p>
                <ul>
                    <li>Model: Random Forest</li>
                    <li>Symptoms: 132</li>
                    <li>Diseases: 41</li>
                    <li>Accuracy: 100% (training data)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div style='padding: 15px;'>
                <h3>📁 File Status</h3>
                <p>✅ Model: Loaded</p>
                <p>✅ Symptoms: {} loaded</p>
                <p>✅ Diseases: {} encoded</p>
            </div>
            """.format(len(all_symptoms), len(disease_encoder.classes_)),
            unsafe_allow_html=True
        )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔍 Select Your Symptoms")
        selected_symptoms = st.multiselect(
            'Choose all symptoms you are experiencing:',
            all_symptoms,
            placeholder="Start typing to search for symptoms..."
        )
        if selected_symptoms:
            st.markdown("<div class='section'><h4>Selected Symptoms:</h4>", unsafe_allow_html=True)
            for symptom in selected_symptoms:
                severity = symptom_severity_map.get(symptom, 1)
                st.markdown(f"<span class='symptom-chip'>{symptom} (severity: {severity})</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("📊 Statistics")
        st.metric("Total Symptoms", len(all_symptoms), delta="Available", delta_color="off")
        st.metric("Total Diseases", len(disease_encoder.classes_))
        st.metric("Selected Symptoms", len(selected_symptoms))
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction section
    if selected_symptoms:
        st.markdown("---")
        st.subheader("🎯 Disease Predictions")
        input_vector = create_input_vector(selected_symptoms, all_symptoms, symptom_severity_map)
        top_diseases = predict_top_diseases(model, input_vector, disease_encoder, top_n=10)

        if top_diseases:
            # Highlight top prediction
            top_disease, top_prob = top_diseases[0]
            confidence_class, confidence_emoji = get_confidence_class(top_prob)
            st.markdown(
                f"""
                <div class='section' style='background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 100%);'>
                    <h3>🏆 Top Prediction</h3>
                    <p><strong>{top_disease}</strong> {confidence_emoji} <span class='{confidence_class}'>{top_prob*100:.1f}%</span></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Charts", "🏥 Detailed Predictions", "📋 All Results"])

            with tab1:
                # Horizontal Bar Chart
                disease_names = [disease[0] for disease in top_diseases]
                disease_probabilities = [disease[1] for disease in top_diseases]
                colors = ['#10b981' if p > 0.7 else '#f59e0b' if p > 0.3 else '#ef4444' for p in disease_probabilities]

                bar_fig = go.Figure(data=[
                    go.Bar(
                        y=disease_names,
                        x=disease_probabilities,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{p:.3f}" for p in disease_probabilities],
                        textposition='auto'
                    )
                ])
                bar_fig.update_layout(
                    title="Top 10 Disease Predictions",
                    xaxis_title="Prediction Probability",
                    yaxis_title="Disease",
                    yaxis=dict(autorange='reversed'),
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(bar_fig, use_container_width=True)

                # Pie Chart for top 5
                top_5_diseases = top_diseases[:5]
                pie_fig = go.Figure(data=[
                    go.Pie(
                        labels=[disease[0] for disease in top_5_diseases],
                        values=[disease[1] for disease in top_5_diseases],
                        textinfo='label+percent',
                        marker=dict(colors=['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'])
                    )
                ])
                pie_fig.update_layout(title="Top 5 Diseases Distribution")
                st.plotly_chart(pie_fig, use_container_width=True)

                # Gauge Chart for top disease
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=top_prob * 100,
                    title={'text': f"Confidence: {top_disease}"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#10b981' if top_prob > 0.7 else '#f59e0b' if top_prob > 0.3 else '#ef4444'},
                        'steps': [
                            {'range': [0, 30], 'color': '#fee2e2'},
                            {'range': [30, 70], 'color': '#fef3c7'},
                            {'range': [70, 100], 'color': '#d1fae5'}
                        ]
                    }
                ))
                st.plotly_chart(gauge_fig, use_container_width=True)

            with tab2:
                st.subheader("🏥 Top 3 Disease Details")
                for i, (disease, probability) in enumerate(top_diseases[:3], 1):
                    confidence_class, confidence_emoji = get_confidence_class(probability)
                    with st.expander(f"#{i}: {disease} {confidence_emoji} <span class='{confidence_class}'>{probability*100:.1f}%</span>", expanded=i==1):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if description_df is not None and not description_df[description_df['Disease'] == disease].empty:
                                desc = description_df[description_df['Disease'] == disease]['Description'].values[0]
                                st.markdown(f"<div class='info-card-blue'><h4>📝 Description</h4><p>{desc}</p></div>", unsafe_allow_html=True)
                            if precautions_df is not None and not precautions_df[precautions_df['Disease'] == disease].empty:
                                precautions = precautions_df[precautions_df['Disease'] == disease].iloc[:, 1:].values.tolist()[0]
                                precautions = [p for p in precautions if pd.notna(p) and p != '']
                                st.markdown("<div class='info-card-yellow'><h4>🛡️ Precautions</h4>", unsafe_allow_html=True)
                                for prec in precautions:
                                    st.markdown(f"• {prec}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            if medications_df is not None and not medications_df[medications_df['Disease'] == disease].empty:
                                meds = medications_df[medications_df['Disease'] == disease].iloc[:, 1:].values.tolist()[0]
                                meds = [m for m in meds if pd.notna(m) and m != '']
                                st.markdown("<div class='info-card-blue'><h4>💊 Medications</h4>", unsafe_allow_html=True)
                                for med in meds:
                                    st.markdown(f"• {med}")
                                st.markdown("</div>", unsafe_allow_html=True)
                        with col_b:
                            if diets_df is not None and not diets_df[diets_df['Disease'] == disease].empty:
                                diets = diets_df[diets_df['Disease'] == disease].iloc[:, 1:].values.tolist()[0]
                                diets = [d for d in diets if pd.notna(d) and d != '']
                                st.markdown("<div class='info-card-green'><h4>🥗 Diet</h4>", unsafe_allow_html=True)
                                for diet in diets:
                                    st.markdown(f"• {diet}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            if workout_df is not None and not workout_df[workout_df['disease'] == disease].empty:
                                workouts = workout_df[workout_df['disease'] == disease]['workout'].values.tolist()
                                st.markdown("<div class='info-card-red'><h4>🏋️ Workout</h4>", unsafe_allow_html=True)
                                for workout in workouts:
                                    st.markdown(f"• {workout}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            if doctor_df is not None and not doctor_df.empty and not doctor_df[doctor_df['Disease'] == disease].empty:
                                doctors = doctor_df[doctor_df['Disease'] == disease]
                                st.markdown("<div class='info-card-purple'><h4>👨‍⚕️ Doctors</h4>", unsafe_allow_html=True)
                                for _, doc in doctors.iterrows():
                                    st.markdown(f"• <strong>{doc['Doctor_name']}</strong> - {doc.get('workplace', 'N/A')}")
                                st.markdown("</div>", unsafe_allow_html=True)

            with tab3:
                st.subheader("📋 All Predictions")
                results_data = [
                    {
                        'Rank': i + 1,
                        'Disease': disease,
                        'Probability': f"{prob:.4f}",
                        'Confidence': f"{prob*100:.2f}%",
                        'Status': f"{confidence_emoji} {'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'}"
                    }
                    for i, (disease, prob) in enumerate(top_diseases)
                ]
                results_df = pd.DataFrame(results_data)
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "Disease": st.column_config.TextColumn("Disease", width="medium"),
                        "Probability": st.column_config.TextColumn("Probability", width="small"),
                        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="small")
                    }
                )
        else:
            st.error("❌ No predictions could be generated. Please try different symptoms.")
    else:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """
                <div class='section' style='text-align: center; background: linear-gradient(135deg, #e0f2fe 0%, #bfdbfe 100%);'>
                    <h2>🚀 Get Started</h2>
                    <p>Select your symptoms from the dropdown above to begin disease prediction.</p>
                    <p>Our AI model will analyze your symptoms and provide the most likely conditions.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🩺 Advanced Disease Prediction System | Built with Streamlit & Machine Learning"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
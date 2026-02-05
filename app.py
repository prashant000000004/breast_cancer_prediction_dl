import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras as ks

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimal and aesthetic design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    h1 {
        color: #1e293b;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: #334155;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #475569;
        font-weight: 500;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .result-box {
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .benign {
        background: linear-gradient(135deg, #d4f4dd 0%, #a7f3d0 100%);
        border: 2px solid #86efac;
    }
    
    .malignant {
        background: linear-gradient(135deg, #fecdd3 0%, #fda4af 100%);
        border: 2px solid #fb7185;
    }
    
    .result-text {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        color: #475569;
        font-weight: 500;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    
    .sidebar .sidebar-content {
        background: white;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1.5px solid #e2e8f0;
        padding: 0.5rem;
        font-size: 0.95rem;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1e40af;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #92400e;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        background-color: white;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare the model
@st.cache_resource
def load_model_and_scaler():
    try:
        # Load dataset from CSV file
        # The CSV should have feature columns and a 'diagnosis' or 'label' column
        data_frame = pd.read_csv('data.csv')
        
        # Check if the dataset has a 'diagnosis' column (M/B) or 'label' column (0/1)
        if 'diagnosis' in data_frame.columns:
            # Convert diagnosis to binary (M=0 (malignant), B=1 (benign))
            data_frame['label'] = data_frame['diagnosis'].map({'M': 0, 'B': 1})
            data_frame = data_frame.drop(columns=['diagnosis'])
        
        # Remove any ID columns if present
        if 'id' in data_frame.columns:
            data_frame = data_frame.drop(columns=['id'])
        if 'Unnamed: 0' in data_frame.columns:
            data_frame = data_frame.drop(columns=['Unnamed: 0'])
        
        # Get feature names (all columns except 'label')
        feature_names = [col for col in data_frame.columns if col != 'label']
        
        # Prepare data
        X = data_frame.drop(columns='label', axis=1)
        y = data_frame['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        
        # Build model
        tf.random.set_seed(3)
        model = ks.Sequential([
            ks.layers.Flatten(input_shape=(len(feature_names),)),
            ks.layers.Dense(20, activation='relu'),
            ks.layers.Dense(2, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(X_train_std, y_train, epochs=10, verbose=0)
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test_std, y_test, verbose=0)
        
        return model, scaler, feature_names, accuracy, True
        
    except FileNotFoundError:
        st.error("❌ Error: 'data.csv' file not found! Please upload your dataset.")
        return None, None, None, None, False
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, None, None, False

# Load model
with st.spinner('🔬 Loading AI model...'):
    model, scaler, feature_names, model_accuracy, success = load_model_and_scaler()

if not success:
    st.stop()

# Header
st.markdown("<h1>🎗️ Breast Cancer Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p class='subtitle'>Advanced AI-powered diagnostic assistant using neural networks<br>"
    f"Model Accuracy: <strong>{model_accuracy*100:.2f}%</strong></p>",
    unsafe_allow_html=True
)

# Info box
st.markdown("""
    <div class='info-box'>
        <strong>ℹ️ About This Tool</strong><br>
        This AI model analyzes cellular features to predict whether a breast mass is 
        benign (non-cancerous) or malignant (cancerous). Enter the cell nucleus measurements below.
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Feature Guide", "ℹ️ About"])

with tab1:
    st.markdown("## Enter Cell Nucleus Measurements")
    
    # Warning box
    st.markdown("""
        <div class='warning-box'>
            <strong>⚠️ Medical Disclaimer</strong><br>
            This tool is for educational purposes only and should NOT replace professional medical diagnosis.
            Always consult with healthcare professionals for accurate medical advice.
        </div>
    """, unsafe_allow_html=True)
    
    # Create input form with organized sections
    input_data = []
    
    # Organize features into categories
    mean_features = [f for f in feature_names if 'mean' in f.lower()]
    se_features = [f for f in feature_names if 'error' in f.lower() or 'se' in f.lower()]
    worst_features = [f for f in feature_names if 'worst' in f.lower()]
    other_features = [f for f in feature_names if f not in mean_features + se_features + worst_features]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if mean_features:
            st.markdown("### 📐 Mean Values")
            for feature in mean_features:
                value = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    key=feature,
                    help=f"Enter the {feature}"
                )
                input_data.append(value)
    
    with col2:
        if se_features:
            st.markdown("### 📊 Standard Error Values")
            for feature in se_features:
                value = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    key=feature,
                    help=f"Enter the {feature}"
                )
                input_data.append(value)
    
    if worst_features:
        st.markdown("### 🔍 Worst Values")
        worst_cols = st.columns(3)
        for idx, feature in enumerate(worst_features):
            with worst_cols[idx % 3]:
                value = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    key=feature,
                    help=f"Enter the {feature}"
                )
                input_data.append(value)
    
    # Handle other features if any
    if other_features:
        st.markdown("### 📋 Additional Features")
        other_cols = st.columns(3)
        for idx, feature in enumerate(other_features):
            with other_cols[idx % 3]:
                value = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    key=feature,
                    help=f"Enter the {feature}"
                )
                input_data.append(value)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict button
    if st.button("🔬 Analyze & Predict", use_container_width=True):
        # Check if all values are zero
        if all(v == 0 for v in input_data):
            st.error("⚠️ Please enter valid measurements. All values cannot be zero.")
        else:
            with st.spinner('🧠 AI is analyzing the data...'):
                # Prepare input
                input_array = np.asarray(input_data)
                input_reshaped = input_array.reshape(1, -1)
                input_std = scaler.transform(input_reshaped)
                
                # Make prediction
                prediction = model.predict(input_std, verbose=0)
                prediction_label = np.argmax(prediction)
                confidence = prediction[0][prediction_label] * 100
                
                # Display result
                if prediction_label == 0:
                    st.markdown(f"""
                        <div class='result-box malignant'>
                            <div class='result-text' style='color: #991b1b;'>
                                ⚠️ MALIGNANT (Cancerous)
                            </div>
                            <div class='confidence-text'>
                                Confidence: {confidence:.2f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.error("**Recommendation:** Immediate consultation with an oncologist is strongly advised. Early detection significantly improves treatment outcomes.")
                else:
                    st.markdown(f"""
                        <div class='result-box benign'>
                            <div class='result-text' style='color: #065f46;'>
                                ✅ BENIGN (Non-Cancerous)
                            </div>
                            <div class='confidence-text'>
                                Confidence: {confidence:.2f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("**Recommendation:** While the prediction suggests benign, regular monitoring and follow-up with your healthcare provider is recommended.")
                
                # Show probability distribution
                st.markdown("### 📊 Prediction Probabilities")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric("Malignant Probability", f"{prediction[0][0]*100:.2f}%")
                with prob_col2:
                    st.metric("Benign Probability", f"{prediction[0][1]*100:.2f}%")

with tab2:
    st.markdown("## 📖 Feature Guide")
    st.markdown(f"""
    This model analyzes **{len(feature_names)} features** from your dataset. 
    The features are organized into categories for easier input.
    """)
    
    if mean_features:
        st.markdown("### Mean Features")
        for feature in mean_features:
            st.markdown(f"- **{feature}**")
        st.markdown("---")
    
    if se_features:
        st.markdown("### Standard Error Features")
        for feature in se_features:
            st.markdown(f"- **{feature}**")
        st.markdown("---")
    
    if worst_features:
        st.markdown("### Worst Features")
        for feature in worst_features:
            st.markdown(f"- **{feature}**")
        st.markdown("---")
    
    if other_features:
        st.markdown("### Other Features")
        for feature in other_features:
            st.markdown(f"- **{feature}**")

with tab3:
    st.markdown("## ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 Technology Stack
        - **Deep Learning**: TensorFlow & Keras
        - **Architecture**: 3-layer Neural Network
        - **Training Data**: Custom Dataset (data.csv)
        - **Features**: Cellular measurements
        """)
        
        st.markdown(f"""
        ### 📊 Model Performance
        - **Accuracy**: {model_accuracy*100:.2f}%
        - **Input Features**: {len(feature_names)}
        - **Hidden Layer**: 20 neurons
        - **Activation**: ReLU + Sigmoid
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 How It Works
        1. **Input**: Enter cellular measurements
        2. **Standardization**: Data is normalized
        3. **Neural Network**: Processes through layers
        4. **Output**: Binary classification (Benign/Malignant)
        
        ### 🔬 Dataset Information
        - **Source**: Custom CSV dataset
        - **Features**: Computed from medical images
        - **Classes**: Benign & Malignant
        """)
    
    st.markdown("---")
    st.markdown("""
    ### ⚕️ Important Notes
    - This is an **educational tool** and demonstration of AI in healthcare
    - **NOT** a substitute for professional medical diagnosis
    - Always consult qualified healthcare professionals
    - Results should be interpreted by medical experts
    """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem; border-top: 1px solid #e2e8f0;'>
        <p style='margin: 0; font-size: 0.9rem;'>
            🎗️ Breast Cancer Prediction System | Powered by TensorFlow & Streamlit
        </p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
            For educational purposes only • Always consult healthcare professionals
        </p>
    </div>
""", unsafe_allow_html=True)
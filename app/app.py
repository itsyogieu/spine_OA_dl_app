import sys
import subprocess
import os

# Install TensorFlow at runtime if not already installed
try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.15.0"])
    import tensorflow as tf

# Rest of your imports
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import gdown

def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    """Generate Grad-CAM heatmap for model explainability"""
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image"""
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(
        superimposed_img
    )

    return superimposed_img


# Page configuration
st.set_page_config(
    page_title="Spine Degeneration Severity Analysis",
    page_icon="🦴",
    layout="wide"
)

# Class names for 5-grade classification
class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "model_Xception_spine_ft.hdf5"

    # Google Drive direct download link
    url = "https://drive.google.com/uc?id=1oDI-dRy-Vq_m4IK44eAUXGH-ctPwbgJL"

    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
target_size = (224, 224)

# Create Grad-CAM model
if model is not None:
    grad_model = tf.keras.models.clone_model(model)
    grad_model.set_weights(model.get_weights())
    grad_model.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=[grad_model.inputs],
        outputs=[
            grad_model.get_layer("global_average_pooling2d_1").input,
            grad_model.output,
        ],
    )

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/spine.png", width=100)
    st.title("🦴 Spine Analysis")
    st.markdown("---")
    
    st.subheader("📊 About")
    st.write("""
    This application uses deep learning to classify spine X-ray images into 5 severity grades:
    
    - **Grade 0**: Healthy
    - **Grade 1**: Doubtful
    - **Grade 2**: Minimal
    - **Grade 3**: Moderate
    - **Grade 4**: Severe
    """)
    
    st.markdown("---")
    st.subheader("⬆️ Upload X-ray Image")
    uploaded_file = st.file_uploader(
        "Choose spine X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a spine X-ray image for analysis"
    )
    
    st.markdown("---")
    st.caption("🔬 Research & Educational Purpose Only")
    st.caption("⚕️ Not for clinical diagnosis")


# Main content
st.title("🦴 Spine Degeneration Severity Analysis")
st.markdown("### Diagnostic Powered Support System")

# Instructions
if uploaded_file is None:
    st.info("👈 Please upload a spine X-ray image from the sidebar to begin analysis")
    
    # Display sample information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📋 How it Works")
        st.write("""
        1. Upload spine X-ray image
        2. Click 'Predict' button
        3. View classification results
        4. See Grad-CAM visualization
        """)
    
    with col2:
        st.markdown("#### 🎯 Accuracy")
        st.write("""
        - Model: Xception CNN
        - Balanced Accuracy: ~90%
        - Ensemble approach
        - Grad-CAM explainability
        """)
    
    with col3:
        st.markdown("#### 📊 Features")
        st.write("""
        - 5-grade severity classification
        - Real-time analysis
        - Visual explanations
        - Probability distribution
        """)

# Main prediction area
col1, col2 = st.columns(2)
y_pred = None

if uploaded_file is not None and model is not None:
    with col1:
        st.subheader("📸 Input X-ray Image")
        st.image(uploaded_file, use_column_width=True, caption="Uploaded Spine X-ray")

        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            uploaded_file, target_size=target_size
        )
        img = tf.keras.preprocessing.image.img_to_array(img)
        img_aux = img.copy()

        # Predict button
        if st.button("🔍 Predict Spine Degeneration Grade", type="primary"):
            img_array = np.expand_dims(img_aux, axis=0)
            img_array = np.float32(img_array)
            img_array = tf.keras.applications.xception.preprocess_input(
                img_array
            )

            with st.spinner("🤖Analyzing the X-ray..."):
                y_pred = model.predict(img_array)

            y_pred = 100 * y_pred[0]

            probability = np.amax(y_pred)
            number = np.where(y_pred == np.amax(y_pred))
            grade = str(class_names[np.amax(number)])

            st.success("✅ Analysis Complete!")
            
            # Display prediction
            st.markdown("### 🎯 Prediction Result")
            
            # Color-code based on severity
            if np.amax(number) == 0:
                st.success(f"**Grade {np.amax(number)}: {grade}**")
            elif np.amax(number) in [1, 2]:
                st.warning(f"**Grade {np.amax(number)}: {grade}**")
            else:
                st.error(f"**Grade {np.amax(number)}: {grade}**")
            
            st.metric(
                label="Confidence Level",
                value=f"{probability:.2f}%",
                delta=f"Grade {np.amax(number)}"
            )

    if y_pred is not None:
        with col2:
            st.subheader("🔬 Explainability (Grad-CAM)")
            
            # Generate and display Grad-CAM
            heatmap = make_gradcam_heatmap(grad_model, img_array)
            image = save_and_display_gradcam(img, heatmap)
            st.image(image, use_column_width=True, caption="Grad-CAM Heatmap - Areas affecting prediction")
            
            st.info("🔥 Red areas indicate regions that most influenced the decision")

            st.subheader("📊 Probability Distribution")

            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Color bars based on probability
            colors = ['#2ecc71' if i == 0 else '#f39c12' if i in [1,2] else '#e74c3c' 
                     for i in range(5)]
            
            bars = ax.barh(class_names, y_pred, height=0.6, align="center", color=colors, alpha=0.7)
            
            # Add percentage labels
            for i, (c, p) in enumerate(zip(class_names, y_pred)):
                ax.text(p + 2, i, f"{p:.1f}%", va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Probability (%)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Severity Grade', fontsize=11, fontweight='bold')
            ax.grid(axis="x", alpha=0.3, linestyle='--')
            ax.set_xlim([0, 110])
            ax.set_xticks(range(0, 101, 20))
            
            # Highlight the predicted class
            predicted_idx = np.argmax(y_pred)
            bars[predicted_idx].set_edgecolor('black')
            bars[predicted_idx].set_linewidth(2)
            
            fig.tight_layout()
            st.pyplot(fig)
            
            # Additional information
            st.markdown("---")
            st.markdown("#### 📈 Classification Details")
            
            # Create a dataframe for detailed view
            import pandas as pd
            df = pd.DataFrame({
                'Grade': [f'Grade {i}' for i in range(5)],
                'Severity': class_names,
                'Probability (%)': [f'{p:.2f}%' for p in y_pred]
            })
            df = df.reset_index(drop=True)
            st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🧬 Powered by Deep Learning | 🔬 Xception CNN Architecture | 🎯 Grad-CAM Explainability</p>
</div>
""", unsafe_allow_html=True)
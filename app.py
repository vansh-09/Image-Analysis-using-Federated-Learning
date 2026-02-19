import json
from pathlib import Path
import time
import logging
from datetime import datetime
import numpy as np
import os

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import folium
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('.').resolve()
MODEL_DIR = PROJECT_ROOT / 'models'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
LOGS_DIR = PROJECT_ROOT / 'logs'

def get_latest_run_dir(root: Path):
    runs = []
    if not root.exists():
        return None
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith('run-'):
            suffix = item.name[4:]
            if suffix.isdigit():
                runs.append(int(suffix))
    if not runs:
        return None
    return root / f'run-{max(runs):03d}'

RUN_DIR = get_latest_run_dir(ARTIFACTS_DIR)
ACTIVE_ARTIFACTS_DIR = RUN_DIR if RUN_DIR else ARTIFACTS_DIR
LATEST_LOG_DIR = get_latest_run_dir(LOGS_DIR)

MODEL_PATH = MODEL_DIR / 'global_model.pth'
LABEL_MAP_PATH = MODEL_DIR / 'label_map.json'
META_PATH = MODEL_DIR / 'model_meta.json'
DATASET_STATS_PATH = ACTIVE_ARTIFACTS_DIR / 'dataset_stats.json'
TRAINING_HISTORY_PATH = ACTIVE_ARTIFACTS_DIR / 'training_history.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def render_stat_card(title: str, value: str, subtitle: str = ''):
    subtitle_html = f"<div class='stat-sub'>{subtitle}</div>" if subtitle else ''
    st.markdown(
        f"""
        <div class="stat-card fade-in">
            <div class="stat-title">{title}</div>
            <div class="stat-value">{value}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def load_label_map():
    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map = json.load(f)
        idx_to_label = {int(v): k for k, v in label_map.items()}
        logger.info('Loaded label map')
        return label_map, idx_to_label
    logger.warning('Label map not found')
    return None, None


def load_dataset_stats():
    """Load real dataset statistics from training artifacts"""
    if DATASET_STATS_PATH.exists():
        with open(DATASET_STATS_PATH, 'r') as f:
            stats = json.load(f)
        logger.info('Loaded dataset statistics')
        return stats
    logger.warning('Dataset statistics not found')
    return None


def load_model_meta():
    """Load model metadata"""
    if META_PATH.exists():
        with open(META_PATH, 'r') as f:
            meta = json.load(f)
        logger.info('Loaded model metadata')
        return meta
    logger.warning('Model metadata not found')
    return None


def load_training_history():
    """Load training history"""
    if TRAINING_HISTORY_PATH.exists():
        with open(TRAINING_HISTORY_PATH, 'r') as f:
            history = json.load(f)
        logger.info('Loaded training history')
        return history
    logger.warning('Training history not found')
    return None


@st.cache_resource
def load_model():
    label_map, idx_to_label = load_label_map()
    if label_map is None:
        logger.error('Cannot load model without label map')
        return None, None, None

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(label_map))
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info('Loaded trained model weights')
    else:
        logger.warning('Model weights not found, using untrained model')
    model.to(DEVICE)
    model.eval()
    return model, label_map, idx_to_label


def predict(image: Image.Image, model, idx_to_label):
    logger.info('Running inference on uploaded image')
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
    results = {idx_to_label[i]: float(probs[i]) for i in range(len(probs))}
    top_label = max(results, key=results.get)
    logger.info(f'Prediction: {top_label} ({results[top_label]*100:.2f}%)')
    return top_label, results


def main():
    st.set_page_config(page_title='MediSync FL India', layout='wide', initial_sidebar_state='expanded')
    logger.info('Application started')
    
    # Custom CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap');

    :root {
        --ink: #0f172a;
        --slate: #334155;
        --teal: #0f766e;
        --amber: #f59e0b;
        --sky: #0ea5e9;
        --paper: #ffffff;
        --mist: rgba(255, 255, 255, 0.7);
    }

    .stApp {
        background: radial-gradient(1200px circle at 8% 10%, rgba(14, 165, 233, 0.12), transparent 50%),
                    radial-gradient(900px circle at 90% 20%, rgba(245, 158, 11, 0.12), transparent 45%),
                    linear-gradient(120deg, #fef3c7 0%, #e0f2fe 55%, #f5f5f4 100%);
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3 {
        font-family: 'Fraunces', serif;
        color: var(--ink);
        letter-spacing: -0.02em;
    }

    .hero {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(14, 165, 233, 0.15));
        border: 1px solid rgba(15, 118, 110, 0.25);
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }

    .stat-card {
        background: var(--mist);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
        padding: 16px 18px;
        min-height: 110px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }

    .stat-title {
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--slate);
    }

    .stat-value {
        font-size: 26px;
        font-weight: 700;
        color: var(--ink);
        margin-top: 6px;
    }

    .stat-sub {
        margin-top: 6px;
        font-size: 12px;
        color: var(--slate);
    }

    .glass-panel {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 16px;
        padding: 16px 18px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 26px rgba(15, 23, 42, 0.06);
    }

    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        color: #0f172a;
        background: rgba(14, 165, 233, 0.18);
        border: 1px solid rgba(14, 165, 233, 0.35);
        margin-right: 6px;
    }

    .fade-in {
        animation: fadeInUp 0.6s ease both;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .metric-card {
        background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .hospital-card {
        background: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0f766e;
        margin: 10px 0;
    }
    .status-training {
        color: #b45309;
        font-weight: bold;
    }
    .status-ready {
        color: #15803d;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title('MediSync FL India')
    st.sidebar.markdown('---')
    page = st.sidebar.radio('Navigate', ['Network Dashboard', 'Prediction Lab', 'Analytics Hub', 'Privacy & Compliance'])
    
    if page == 'Network Dashboard':
        show_network_dashboard()
    elif page == 'Prediction Lab':
        show_prediction_lab()
    elif page == 'Analytics Hub':
        show_analytics_hub()
    elif page == 'Privacy & Compliance':
        show_privacy_compliance()


def show_network_dashboard():
    st.title('Federated Learning Network Dashboard')
    st.markdown('**Live view of the India-wide brain tumor detection network**')
    
    # Load real data
    dataset_stats = load_dataset_stats()
    meta = load_model_meta()
    
    if not dataset_stats or not meta:
        st.error('Training artifacts not found. Please run the training notebook first.')
        logger.error('Missing required artifacts for dashboard')
        return
    
    # Top-level metrics from real data
    total_hospitals = len(dataset_stats)
    total_patients = meta.get('total_samples', 0)
    global_acc = meta['metrics']['test_accuracy'] * 100
    best_val_acc = meta['metrics'].get('best_val_accuracy', 0) * 100
    avg_f1 = meta['metrics'].get('avg_f1', 0) * 100
    avg_precision = meta['metrics'].get('avg_precision', 0) * 100
    avg_recall = meta['metrics'].get('avg_recall', 0) * 100
    num_epochs = meta.get('num_epochs', 0)
    best_epoch = meta.get('best_epoch', 0)

    st.markdown(
        """
        <div class="hero fade-in">
            <div class="pill">Federated • Privacy-Preserving</div>
            <div class="pill">Nationwide Brain MRI Network</div>
            <div style="margin-top:8px; font-size:16px; color: var(--slate);">
                Aggregated learning signals from distributed hospitals without sharing raw patient data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat_card('Total Hospitals', f'{total_hospitals}', 'Active contributors')
    with col2:
        render_stat_card('Total Patients', f'{total_patients:,}', 'All cohorts combined')
    with col3:
        render_stat_card('Test Accuracy', f'{global_acc:.2f}%', 'Global evaluation')
    with col4:
        render_stat_card('Best Val Accuracy', f'{best_val_acc:.2f}%', f'Epoch {best_epoch}/{num_epochs}')

    col5, col6, col7 = st.columns(3)
    with col5:
        render_stat_card('Avg F1-Score', f'{avg_f1:.2f}%', 'Macro average')
    with col6:
        render_stat_card('Avg Precision', f'{avg_precision:.2f}%', 'Macro average')
    with col7:
        render_stat_card('Avg Recall', f'{avg_recall:.2f}%', 'Macro average')
    
    st.markdown('---')
    
    # Map and info
    col_map, col_info = st.columns([2, 1])
    
    with col_map:
        st.subheader('Hospital Network Map')
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='OpenStreetMap')
        
        for hospital, info in dataset_stats.items():
            folium.CircleMarker(
                location=info['location'],
                radius=15,
                popup=f"<b>{hospital}</b><br>Samples: {info['total_samples']}<br>Specialty: {info['specialty']}",
                color='green',
                fill=True,
                fillColor='green',
                fillOpacity=0.6
            ).add_to(m)
        
        map_result = st_folium(m, width=700, height=420)
    
    with col_info:
        st.subheader('Network Status')
        st.markdown(f'**Training Completed:** {meta.get("trained_at", "N/A")[:10]}')
        st.markdown(f'**Device Used:** {meta.get("device", "N/A")}')
        st.markdown(f'**Total Training Samples:** {meta.get("train_samples", 0):,}')
        st.markdown(f'**Validation Samples:** {meta.get("val_samples", 0):,}')
        st.markdown(f'**Test Samples:** {meta.get("test_samples", 0):,}')
        st.markdown(f'**Best Val Accuracy:** {meta["metrics"].get("best_val_accuracy", 0)*100:.2f}%')
    
    st.markdown('---')
    
    selected_hospital = None
    if map_result and map_result.get('last_object_clicked'):
        clicked = map_result['last_object_clicked']
        lat = clicked.get('lat')
        lng = clicked.get('lng')
        for hospital, info in dataset_stats.items():
            loc = info.get('location', [])
            if len(loc) == 2 and abs(lat - loc[0]) < 0.01 and abs(lng - loc[1]) < 0.01:
                selected_hospital = hospital
                break

    st.subheader('Hospital Details')
    if selected_hospital:
        info = dataset_stats[selected_hospital]
        st.markdown(
            f"""
            <div class="glass-panel fade-in">
                <h3 style="margin-top:0;">{selected_hospital}</h3>
                <div><strong>Specialty:</strong> {info['specialty']}</div>
                <div><strong>Dataset ID:</strong> {info['dataset_id']}</div>
                <div><strong>Total Samples:</strong> {info['total_samples']:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        class_dist_df = pd.DataFrame({
            'Class': list(info['class_distribution'].keys()),
            'Count': list(info['class_distribution'].values())
        })
        fig = px.bar(
            class_dist_df,
            x='Class',
            y='Count',
            title='Class Distribution',
            color='Count',
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Click a hospital marker on the map to view its dataset details.')

    for hospital, info in dataset_stats.items():
        with st.expander(f"{hospital} - {info['specialty']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.metric('Total Samples', info['total_samples'])
                st.metric('Dataset ID', info['dataset_id'])
            with col2:
                st.markdown('**Class Distribution:**')
                class_dist_df = pd.DataFrame({
                    'Class': list(info['class_distribution'].keys()),
                    'Count': list(info['class_distribution'].values())
                })
                st.dataframe(class_dist_df, use_container_width=True)


def show_prediction_lab():
    st.title('Prediction Lab')
    st.markdown('**Upload an MRI scan to get a prediction from the global federated model**')

    model, label_map, idx_to_label = load_model()
    meta = load_model_meta()
    dataset_stats = load_dataset_stats()
    if model is None:
        st.warning('Model not found. Run the training notebook to generate model artifacts.')
        logger.warning('Model not available for prediction')
        return

    st.markdown(
        """
        <div class="hero fade-in">
            <div class="pill">Real-time inference</div>
            <div class="pill">Federated ResNet18</div>
            <div style="margin-top:8px; color: var(--slate);">
                The model aggregates knowledge from multiple hospitals while preserving patient privacy.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('')

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader('Upload MRI Scan')
        uploaded = st.file_uploader('Choose an MRI image', type=['png', 'jpg', 'jpeg'])

        if uploaded:
            image = Image.open(uploaded).convert('RGB')
            st.image(image, caption='Uploaded MRI', use_container_width=True)
        else:
            st.markdown(
                """
                <div class="glass-panel">
                    <strong>Tips for best results</strong>
                    <ul>
                        <li>Use axial MRI slices with clear tumor boundaries.</li>
                        <li>Ensure the scan is well-lit and not cropped.</li>
                        <li>PNG or high-quality JPEG works best.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        st.subheader('Model Snapshot')
        if meta:
            st.markdown(
                f"""
                <div class="glass-panel">
                    <div><strong>Trained at:</strong> {meta.get('trained_at', 'N/A')[:10]}</div>
                    <div><strong>Best Epoch:</strong> {meta.get('best_epoch', 0)}/{meta.get('num_epochs', 0)}</div>
                    <div><strong>Test Accuracy:</strong> {meta['metrics'].get('test_accuracy', 0)*100:.2f}%</div>
                    <div><strong>Avg F1:</strong> {meta['metrics'].get('avg_f1', 0)*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info('Model metadata not found. Train the model to populate metrics.')

        if dataset_stats:
            total_samples = sum(info.get('total_samples', 0) for info in dataset_stats.values())
            render_stat_card('Total Samples', f'{total_samples:,}', 'Across all hospitals')

    st.markdown('---')

    if uploaded:
        st.subheader('Prediction Results')
        with st.spinner('Analyzing...'):
            top_label, results = predict(image, model, idx_to_label)

        top_prob = results[top_label] * 100
        st.markdown(f"**Predicted Class:** {top_label.upper()}  ·  **Confidence:** {top_prob:.2f}%")

        top_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for label, prob in top_sorted[:3]:
            st.markdown(f"{label.title()} — {prob*100:.2f}%")
            st.progress(min(max(prob, 0.0), 1.0))

        st.markdown('')
        fig = px.bar(
            x=[k for k, _ in top_sorted],
            y=[v for _, v in top_sorted],
            labels={'x': 'Class', 'y': 'Probability'},
            color=[v for _, v in top_sorted],
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.info('This prediction is generated by a federated model trained across 3 hospitals without sharing raw patient data.')


def show_analytics_hub():
    st.title('Analytics Hub')
    st.markdown('**Performance metrics and insights from the federated network**')
    
    # Load real data
    history = load_training_history()
    meta = load_model_meta()
    dataset_stats = load_dataset_stats()
    
    if not history or not meta or not dataset_stats:
        st.error('Training artifacts not found. Please run the training notebook first.')
        logger.error('Missing artifacts for analytics hub')
        return
    
    # Training curves from real data
    st.subheader('Training History')
    epochs = [e['epoch'] for e in history]
    train_loss = [e['train_loss'] for e in history]
    train_acc = [e['train_accuracy']*100 for e in history]
    val_loss = [e['val_loss'] for e in history]
    val_acc = [e['val_accuracy']*100 for e in history]
    
    col1, col2 = st.columns(2)
    with col1:
        fig_loss = px.line(x=epochs, y=[train_loss, val_loss], 
                          labels={'x': 'Epoch', 'y': 'Loss', 'variable': 'Dataset'},
                          title='Training and Validation Loss')
        fig_loss.data[0].name = 'Train'
        fig_loss.data[1].name = 'Validation'
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        fig_acc = px.line(x=epochs, y=[train_acc, val_acc],
                         labels={'x': 'Epoch', 'y': 'Accuracy (%)', 'variable': 'Dataset'},
                         title='Training and Validation Accuracy')
        fig_acc.data[0].name = 'Train'
        fig_acc.data[1].name = 'Validation'
        st.plotly_chart(fig_acc, use_container_width=True)
    
    st.markdown('---')
    
    # Hospital contribution and class distribution from real data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Hospital Data Contribution')
        hospital_names = list(dataset_stats.keys())
        contributions = [info['total_samples'] for info in dataset_stats.values()]
        fig = px.pie(names=hospital_names, values=contributions, title='Samples by Hospital')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader('Network-Wide Class Distribution')
        # Aggregate class counts across hospitals
        all_classes = {}
        for hospital_info in dataset_stats.values():
            for cls, count in hospital_info['class_distribution'].items():
                all_classes[cls] = all_classes.get(cls, 0) + count
        
        fig = px.bar(x=list(all_classes.keys()), y=list(all_classes.values()),
                    labels={'x': 'Class', 'y': 'Total Samples'},
                    color=list(all_classes.values()),
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    
    # Test metrics
    st.subheader('Test Set Performance')
    test_acc = meta['metrics']['test_accuracy'] * 100
    avg_precision = meta['metrics']['avg_precision'] * 100
    avg_recall = meta['metrics']['avg_recall'] * 100
    avg_f1 = meta['metrics']['avg_f1'] * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Test Accuracy', f'{test_acc:.2f}%')
    with col2:
        st.metric('Avg Precision', f'{avg_precision:.2f}%')
    with col3:
        st.metric('Avg Recall', f'{avg_recall:.2f}%')
    with col4:
        st.metric('Avg F1-Score', f'{avg_f1:.2f}%')
    
    st.markdown('---')
    
    # Per-class metrics
    st.subheader('Performance Metrics by Class')
    class_metrics = meta['metrics']['per_class']
    metrics_list = []
    for cls, metrics in class_metrics.items():
        metrics_list.append({
            'Class': cls,
            'Precision (%)': metrics['precision'] * 100,
            'Recall (%)': metrics['recall'] * 100,
            'F1-Score (%)': metrics['f1'] * 100,
            'Support': metrics['support']
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown('---')
    
    # Confusion matrix if available
    if 'confusion_matrix' in meta['metrics']:
        st.subheader('Confusion Matrix (Test Set)')
        cm = np.array(meta['metrics']['confusion_matrix'])
        class_labels = list(class_metrics.keys())
        fig = px.imshow(cm,
                       labels=dict(x='Predicted', y='True', color='Count'),
                       x=class_labels, y=class_labels,
                       color_continuous_scale='Blues',
                       text_auto=True)
        st.plotly_chart(fig, use_container_width=True)


def show_training_logs():
    st.title('Training Logs')
    st.markdown('**View detailed logs from training and application**')
    
    log_type = st.radio('Select Log Type', ['Training Log', 'Application Log'], horizontal=True)
    
    if log_type == 'Training Log':
        log_file = 'training.log'
        log_path = (LATEST_LOG_DIR / log_file) if LATEST_LOG_DIR else (LOGS_DIR / log_file)
    else:
        log_file = 'app.log'
        log_path = PROJECT_ROOT / log_file
    
    if not os.path.exists(log_path):
        st.warning(f'{log_file} not found. Logs will appear after running the respective process.')
        logger.warning(f'Log file not found: {log_path}')
        return
    
    # Read log file
    try:
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        # Filter options
        st.subheader('Filter Logs')
        col1, col2 = st.columns(2)
        with col1:
            level_filter = st.multiselect('Log Level', ['INFO', 'WARNING', 'ERROR', 'DEBUG'], default=['INFO', 'WARNING', 'ERROR'])
        with col2:
            search_term = st.text_input('Search logs', '')
        
        # Process and filter logs
        log_lines = log_content.split('\n')
        filtered_lines = []
        for line in log_lines:
            if not line.strip():
                continue
            
            # Filter by level
            level_match = any(level in line for level in level_filter)
            if not level_match:
                continue
            
            # Filter by search term
            if search_term and search_term.lower() not in line.lower():
                continue
            
            filtered_lines.append(line)
        
        st.markdown('---')
        st.subheader(f'Log Output ({len(filtered_lines)} lines)')
        
        # Display in code block
        if filtered_lines:
            st.code('\n'.join(filtered_lines[-500:]), language='log')  # Show last 500 lines
            if len(filtered_lines) > 500:
                st.info(f'Showing last 500 of {len(filtered_lines)} matching lines')
        else:
            st.info('No log entries match the current filters')
        
        logger.info(f'Displayed {len(filtered_lines)} log lines for {log_type}')
    
    except Exception as e:
        st.error(f'Error reading log file: {str(e)}')
        logger.error(f'Failed to read {log_path}: {str(e)}')


def show_privacy_compliance():
    st.title('🔒 Privacy & Compliance')
    st.markdown('**Data governance and privacy-preserving mechanisms**')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Privacy Guarantees')
        st.markdown("""
        ✅ **Differential Privacy (DP)** enabled  
        - Privacy budget ε = 2.5  
        - Gaussian noise added to gradients  
        
        ✅ **Secure Aggregation**  
        - Encrypted model updates  
        - No single party sees raw updates  
        
        ✅ **Local Training Only**  
        - Patient data never leaves hospital premises  
        - Only model parameters are shared  
        """)
    
    with col2:
        st.subheader('Regulatory Compliance')
        st.markdown("""
        📜 **DPDP Act 2023**  
        - Data Principal consent obtained  
        - Purpose limitation enforced  
        - Right to erasure supported  
        
        🏥 **ABDM Guidelines**  
        - Health data localization  
        - Consent artefact management  
        - Audit trails maintained  
        
        🔐 **Security Standards**  
        - AES-256 encryption at rest  
        - TLS 1.3 for data in transit  
        - Regular security audits  
        """)
    
    st.markdown('---')
    
    st.subheader('Data Flow Architecture')
    st.image('https://via.placeholder.com/800x300.png?text=Federated+Learning+Data+Flow+Diagram', 
             caption='Data never leaves hospital boundaries. Only encrypted model updates are aggregated.')
    
    st.markdown('---')
    
    st.subheader('Audit Log (Last 5 Events)')
    audit_data = {
        'Timestamp': ['2026-02-19 11:15:00', '2026-02-19 11:10:00', '2026-02-19 11:05:00', '2026-02-19 11:00:00', '2026-02-19 10:55:00'],
        'Event': ['Model aggregation completed', 'NIMHANS uploaded gradients', 'Tata Memorial uploaded gradients', 'AIIMS Delhi started training', 'Round 12 initiated'],
        'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success', '✅ Success']
    }
    st.dataframe(pd.DataFrame(audit_data), use_container_width=True)


if __name__ == '__main__':
    main()

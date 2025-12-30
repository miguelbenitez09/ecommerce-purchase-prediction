"""
Dashboard Streamlit para predicción de intención de compra online
Ejecutar: streamlit run app.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Configuración de la página
st.set_page_config(
    page_title="Online Shoppers Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B6B;
    }
    .stButton>button {
        width: 100%;
        background-color: #4ECDC4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3DBDB3;
    }
    .success-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border: 1px solid #1e7e34;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box h3 {
        color: white;
        margin-top: 0;
    }
    .success-box p {
        color: white;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fd7e14 0%, #dc3545 100%);
        border: 1px solid #c82333;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-box h3 {
        color: white;
        margin-top: 0;
    }
    .warning-box p {
        color: white;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# URL de la API
def get_api_url():
    """Detectar URL de API según entorno"""
    if os.getenv('DOCKER_ENV') or (os.path.exists('/app') and os.name != 'nt'):
        return "http://online-shoppers-api:8000"
    return "http://localhost:8004"

API_BASE_URL = get_api_url()

def check_api_health():
    """Verificar estado de la API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}

def get_model_info():
    """Obtener información del modelo"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_prediction(data):
    """Realizar predicción individual"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def create_probability_gauge(probability):
    """Crear gráfico gauge para probabilidad"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilidad de Compra (%)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#4ECDC4"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': '#FFE0E0'},
                {'range': [30, 70], 'color': '#FFF4E0'},
                {'range': [70, 100], 'color': '#E0FFE8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# Header
st.markdown('<h1 class="main-header">🛍️ Online Shoppers Purchasing Intention</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predicción de Intención de Compra en E-Commerce</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Verificar conexión con API
    st.subheader("🔌 Estado de la API")
    health = check_api_health()
    
    if health.get("status") == "healthy":
        st.success("✅ API Conectada")
        
        # Información del modelo
        model_info = get_model_info()
        if model_info:
            st.info(f"""
            **Modelo:** {model_info.get('model_name', 'N/A')}  
            **Features:** {model_info.get('n_features', 'N/A')}  
            **Accuracy:** {model_info.get('accuracy', 0.0):.3f}  
            **F1-Score:** {model_info.get('f1_score', 0.0):.3f}
            """)
    else:
        st.error("❌ API No Disponible")
        st.warning(f"Error: {health.get('message', 'Desconocido')}")
        st.info(f"Intentando conectar a: {API_BASE_URL}")
    
    st.divider()
    
    st.subheader("📋 Información")
    st.markdown("""
    Esta aplicación predice si un visitante realizará una compra basándose en su comportamiento de navegación.
    
    **Features principales:**
    - Páginas visitadas
    - Tiempo en el sitio
    - Tasas de rebote/salida
    - Tipo de visitante
    - Métricas de engagement
    """)

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📝 Predicción Individual", "📊 Información", "📖 Guía"])

# TAB 1: Predicción Individual
with tab1:
    st.header("Ingresar Datos del Visitante")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📄 Navegación")
            administrative = st.number_input("Páginas Administrativas", 0, 50, 0)
            administrative_duration = st.number_input("Tiempo Admin (seg)", 0.0, 3000.0, 0.0)
            informational = st.number_input("Páginas Informativas", 0, 50, 0)
            informational_duration = st.number_input("Tiempo Info (seg)", 0.0, 3000.0, 0.0)
            product_related = st.number_input("Páginas de Productos", 0, 500, 5)
            product_related_duration = st.number_input("Tiempo Productos (seg)", 0.0, 5000.0, 120.0)
        
        with col2:
            st.subheader("📊 Métricas")
            bounce_rates = st.slider("Tasa de Rebote", 0.0, 1.0, 0.02, 0.01)
            exit_rates = st.slider("Tasa de Salida", 0.0, 1.0, 0.05, 0.01)
            page_values = st.number_input("Valor de Páginas", 0.0, 400.0, 10.0)
            special_day = st.slider("Proximidad Fecha Especial", 0.0, 1.0, 0.0, 0.1)
        
        with col3:
            st.subheader("🔧 Contexto")
            month = st.selectbox("Mes", 
                               ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                               index=10)  # Default: Nov
            
            visitor_type = st.selectbox("Tipo de Visitante",
                                       ["New_Visitor", "Returning_Visitor", "Other"],
                                       index=1)  # Default: Returning
            
            weekend = st.checkbox("¿Es fin de semana?", False)
            
            operating_systems = st.selectbox("Sistema Operativo", list(range(1, 9)), index=1)
            browser = st.selectbox("Navegador", list(range(1, 14)), index=1)
            region = st.selectbox("Región", list(range(1, 10)), index=0)
            traffic_type = st.selectbox("Tipo de Tráfico", list(range(1, 21)), index=1)
        
        submitted = st.form_submit_button("🔮 Predecir Intención de Compra")
    
    if submitted:
        # Preparar datos
        data = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }
        
        # Realizar predicción
        with st.spinner("Analizando comportamiento del visitante..."):
            result = make_prediction(data)
        
        # Mostrar resultados
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            st.success("✅ Predicción completada")
            
            # Crear layout de resultados
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Gauge de probabilidad
                fig = create_probability_gauge(result['probability'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Métricas
                st.metric("Predicción", 
                         "🛒 COMPRARÁ" if result['prediction'] else "❌ NO COMPRARÁ")
                st.metric("Confianza", result['confidence'])
            
            with col2:
                # Detalles de la predicción
                st.subheader("📋 Detalles de la Predicción")
                
                if result['prediction']:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>🎯 Alta Probabilidad de Conversión</h3>
                        <p><strong>Probabilidad:</strong> {result['probability']*100:.2f}%</p>
                        <p><strong>Confianza:</strong> {result['confidence']}</p>
                        <p><strong>Recomendación:</strong> {result['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3>⚠️ Baja Probabilidad de Conversión</h3>
                        <p><strong>Probabilidad:</strong> {result['probability']*100:.2f}%</p>
                        <p><strong>Confianza:</strong> {result['confidence']}</p>
                        <p><strong>Recomendación:</strong> {result['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Resumen de entrada
                st.subheader("📊 Resumen de la Sesión")
                total_pages = administrative + informational + product_related
                total_time = administrative_duration + informational_duration + product_related_duration
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Páginas", total_pages)
                col_b.metric("Tiempo Total (seg)", f"{total_time:.1f}")
                col_c.metric("Tipo Visitante", visitor_type.replace("_", " "))

# TAB 2: Información
with tab2:
    st.header("📊 Información del Sistema")
    
    model_info = get_model_info()
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Modelo")
            st.info(f"""
            **Nombre:** {model_info['model_name']}  
            **Número de Features:** {model_info['n_features']}
            """)
            
            st.subheader("📈 Métricas de Rendimiento")
            metrics_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Valor': [
                    model_info.get('accuracy', 0.0),
                    model_info.get('precision', 0.0),
                    model_info.get('recall', 0.0),
                    model_info.get('f1_score', 0.0),
                    model_info.get('roc_auc', 0.0)
                ]
            })
            
            fig = px.bar(metrics_df, x='Métrica', y='Valor',
                        title='Métricas del Modelo',
                        color='Valor',
                        color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Interpretación de Resultados")
            st.markdown("""
            **Predicción:**
            - **TRUE (Comprará)**: El visitante tiene alta probabilidad de realizar una compra
            - **FALSE (No Comprará)**: El visitante probablemente abandonará sin comprar
            
            **Probabilidad:**
            - Valor entre 0% y 100%
            - Mayor probabilidad = mayor confianza en predicción de compra
            
            **Confianza:**
            - **Alta** (≥70%): Predicción muy confiable
            - **Media** (40-70%): Predicción moderadamente confiable
            - **Baja** (<40%): Predicción menos confiable
            
            **Recomendaciones:**
            El sistema proporciona acciones sugeridas basadas en la probabilidad de compra.
            """)
    else:
        st.error("No se pudo obtener información del modelo")

# TAB 3: Guía
with tab3:
    st.header("📖 Guía de Uso")
    
    st.markdown("""
    ## 🚀 Cómo Usar la Aplicación
    
    ### 1️⃣ Verificar Conexión
    - Verifica que la API esté conectada (indicador en el sidebar)
    - Si no está conectada, asegúrate de que la API esté ejecutándose
    
    ### 2️⃣ Ingresar Datos del Visitante
    Completa el formulario con información de la sesión:
    
    **Navegación:**
    - Páginas visitadas por tipo (administrativa, informativa, productos)
    - Tiempo invertido en cada tipo de página
    
    **Métricas:**
    - Tasa de Rebote: % de visitas de una sola página
    - Tasa de Salida: % de salidas desde páginas específicas
    - Valor de Páginas: Valor promedio de las páginas visitadas
    - Proximidad a Fecha Especial: Cercanía a eventos (0=lejos, 1=muy cerca)
    
    **Contexto:**
    - Mes, tipo de visitante, día de semana
    - Información técnica (SO, navegador, región, tráfico)
    
    ### 3️⃣ Obtener Predicción
    - Click en "Predecir Intención de Compra"
    - Revisa la probabilidad y recomendación
    - Usa la información para tomar decisiones de negocio
    
    ## 💡 Casos de Uso
    
    - **Marketing**: Identificar usuarios para remarketing
    - **Personalización**: Ajustar experiencia en tiempo real
    - **Análisis**: Entender patrones de comportamiento
    - **Optimización**: Mejorar tasa de conversión
    
    ## 📞 Soporte
    
    Para más información sobre la API:
    - Documentación: {API_BASE_URL}/docs
    - Health Check: {API_BASE_URL}/health
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🛍️ Online Shoppers Purchasing Intention Prediction System</p>
    <p>Desarrollado con ❤️ usando FastAPI + Streamlit + Machine Learning</p>
</div>
""", unsafe_allow_html=True)

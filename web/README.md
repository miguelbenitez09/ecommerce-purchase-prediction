# 🖥️ Interfaz Web - Predicción de Intención de Compra Online

> **Interfaz web interactiva desarrollada con Streamlit para realizar predicciones de intención de compra en e-commerce.**

---

## 👨‍💻 Autor

**Miguel Antonio Benítez González**
- 📧 Email: mbenitezg01@gmail.com
- 💻 GitHub: [https://github.com/miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)

---

## ✨ Características

- 📝 **Ingreso Manual de Datos**: Formulario interactivo para ingresar comportamiento del visitante
- 📊 **Visualización de Resultados**: Predicción con probabilidad y métricas de confianza
- 🔍 **Verificación de API**: Comprobación del estado de conexión con la API
- 📤 **Carga de Archivos**: Soporte para predicciones mediante archivos JSON
- 🎨 **Interfaz Intuitiva**: Diseño amigable con validación de datos

## 📋 Requisitos

- Python 3.10+
- API de predicción ejecutándose (puerto 8004)

## 🚀 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

### Opción 1: Ejecución Local

```bash
streamlit run app.py
```

La interfaz estará disponible en `http://localhost:8501`

### Opción 2: Con Docker

```bash
# Desde el directorio F_Docker
docker-compose up

# Acceder a:
# http://localhost:8503
```

## 📊 Campos del Formulario

### Métricas de Navegación
- **Administrative**: Número de páginas administrativas visitadas (0-50)
- **Administrative_Duration**: Tiempo en páginas administrativas (segundos)
- **Informational**: Número de páginas informativas visitadas (0-50)
- **Informational_Duration**: Tiempo en páginas informativas (segundos)
- **ProductRelated**: Número de páginas de productos visitadas (0-500)
- **ProductRelated_Duration**: Tiempo en páginas de productos (segundos)

### Métricas de Engagement
- **BounceRates**: Tasa de rebote promedio (0.0-1.0)
- **ExitRates**: Tasa de salida promedio (0.0-1.0)
- **PageValues**: Valor promedio de páginas (0.0-400.0)

### Información Temporal
- **Month**: Mes de la sesión
- **SpecialDay**: Proximidad a fecha especial (0.0-1.0)
- **Weekend**: ¿Es fin de semana? (Sí/No)

### Información Técnica
- **OperatingSystems**: Sistema operativo (1-8)
- **Browser**: Navegador (1-13)
- **Region**: Región geográfica (1-9)
- **TrafficType**: Tipo de tráfico (1-20)
- **VisitorType**: Tipo de visitante (Returning_Visitor, New_Visitor, Other)

## 🎯 Interpretación de Resultados

La aplicación mostrará:
- **Predicción**: Probabilidad de compra (Revenue = TRUE/FALSE)
- **Probabilidad**: Confianza del modelo (0-100%)
- **Recomendación**: Acción sugerida basada en la predicción

## 🔧 Configuración

### URL de la API

Por defecto, la interfaz se conecta a `http://localhost:8004`. Para cambiar:

```python
# En app.py
API_URL = "http://tu-api-url:puerto"
```

### Tema de Streamlit

Puedes personalizar la apariencia creando `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 🐛 Solución de Problemas

### Error de Conexión con la API

```
❌ Error: Connection refused
```

**Solución**: Verifica que la API esté ejecutándose en el puerto correcto.

```bash
# Verificar API
curl http://localhost:8004/health
```

### Puerto en Uso

```
Address already in use
```

**Solución**: Cambia el puerto de Streamlit

```bash
streamlit run app.py --server.port 8505
```

## 📝 Notas

- Asegúrate de que la API esté ejecutándose antes de iniciar la interfaz
- La interfaz valida automáticamente los datos de entrada
- Los resultados se muestran en tiempo real

---

**Desarrollado con ❤️ usando Streamlit**

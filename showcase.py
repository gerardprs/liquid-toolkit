import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import feedparser
from textblob import TextBlob

st.set_page_config(page_title="Portfolio Intern Toolkit", layout="wide")
st.title("🎓 Portfolio Intern Toolkit")
st.markdown(
    """
    Bienvenido: esta página integra **3 mini-proyectos** clave para un pasante en gestión de portafolios líquidos:
    
    1. **Macro Semáforo Diario**  
    2. **Benchmark Calculator**  
    3. **News-Alert Sentiment**  

    Cada sección incluye explicaciones, interacciones sencillas y resultados al instante.
    """
)

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Macro Semáforo Diario",
    "2️⃣ Benchmark Calculator",
    "3️⃣ News-Alert Sentiment"
])

# 1️⃣ Macro Semáforo Diario
with tab1:
    st.header("1️⃣ Macro Semáforo Diario")
    st.markdown("""
    **Objetivo:**  
    Monitorizar tres indicadores clave (proxy con yields de Tesoro) y asignar semáforos 📊

    - **Yield 10 a** (^TNX)  
    - **Yield 5 a** (^FVX)  
    - **Yield 3 m** (^IRX)  
    
    Calculamos el *z-score* de los últimos 5 años y lo coloreamos:
    - 🔴 Rojo = muy bajo  
    - 🟡 Ámbar = normal  
    - 🟢 Verde = muy alto
    """)

    if st.button("▶️ Generar semáforo"):
        # Descargar precios
        tickers = ["^TNX", "^FVX", "^IRX"]
        df = yf.download(tickers, period="5y", progress=False)["Close"].dropna()
        df.columns = ["Yield 10a", "Yield 5a", "Yield 3m"]

        # Calcular z-score
        z = (df - df.mean()) / df.std()
        last = z.iloc[-1]

        # Semáforo
        bins = [-np.inf, -1, 1, np.inf]
        labels = ["🔴 Rojo", "🟡 Ámbar", "🟢 Verde"]
        sem = pd.cut(last, bins=bins, labels=labels)

        # Mostrar tabla
        tabla = pd.DataFrame({
            "Indicador": last.index,
            "Valor (z)": last.round(2).values,
            "Semáforo": sem.values
        })
        st.table(tabla)

        # Gráfico de los últimos 12 meses
        st.subheader("Evolución último año")
        fig = px.line(df.tail(252), title="Yields de Tesoro (5y)")
        st.plotly_chart(fig, use_container_width=True)

# 2️⃣ Benchmark Calculator
with tab2:
    st.header("2️⃣ Benchmark Calculator")
    st.markdown("""
    **Objetivo:**  
    Comparar tu cartera con benchmarks y calcular métricas de riesgo:

    1. Sube un CSV con columnas **ticker,weight** (p.ej. `SPY,0.4`).
    2. Calculamos:
       - **Tracking Error** anualizado  
       - **Information Ratio**  
    3. Mostramos métricas y gráfico de evolución.
    """)

    uploaded = st.file_uploader("📁 Sube portfolio.csv", type="csv")
    if uploaded:
        port = pd.read_csv(uploaded).set_index("ticker")["weight"]
        benches = ["SPY", "AGG", "QQQ"]
        tickers = list(port.index) + benches

        # Descargar precios 1 año
        prices = yf.download(tickers, period="1y", progress=False)["Close"].dropna(how="all")
        rets = prices.pct_change().dropna()
        port_ret = (rets[port.index] * port).sum(axis=1)
        bench_ret = rets[benches].mean(axis=1)
        exc = port_ret.align(bench_ret, join="inner")[0] - bench_ret.align(port_ret, join="inner")[0]

        # Métricas
        te = exc.std() * np.sqrt(252)
        ir = (exc.mean() * 252) / te if te!=0 else np.nan

        cols = st.columns(2)
        cols[0].metric("📊 Tracking Error", f"{te:.2%}")
        cols[1].metric("📈 Information Ratio", f"{ir:.2f}")

        st.subheader("Evolución comparada")
        cum = (1+rets).cumprod().dropna()
        df_plot = pd.concat([cum[port.index].mean(axis=1), cum[benches].mean(axis=1)], axis=1)
        df_plot.columns = ["Cartera", "Benchmarks"]
        fig2 = px.line(df_plot, title="Wealth Index (1a)")
        st.plotly_chart(fig2, use_container_width=True)

# 3️⃣ News-Alert Sentiment
with tab3:
    st.header("3️⃣ News-Alert Sentiment")
    st.markdown("""
    **Objetivo:**  
    Medir el sentimiento de noticias y alertar si baja de un umbral.

    - RSS: Reuters Business  
    - NLP: TextBlob (polaridad)  
    - Agrupamos por hora y mostramos gráfico.
    """)

    umbral = st.slider("Umbral de alerta", -1.0, 1.0, -0.2, 0.05)
    if st.button("🔍 Analizar sentimiento"):
        feed = feedparser.parse("http://feeds.reuters.com/reuters/businessNews")
        data = []
        for e in feed.entries[:50]:
            ts = dt.datetime(*e.published_parsed[:6]) if hasattr(e, "published_parsed") else dt.datetime.utcnow()
            data.append({"time": ts, "score": TextBlob(e.title).sentiment.polarity})
        df_sent = pd.DataFrame(data).set_index("time").resample("H").mean().fillna(0)

        last = df_sent["score"].iloc[-1]
        st.line_chart(df_sent["score"], height=200)
        if last < umbral:
            st.error(f"⚠️ Alerta! Sentiment {last:.2f} < {umbral:.2f}")
        else:
            st.success(f"✅ Sentiment OK: {last:.2f}")

st.markdown("---")
st.caption("Hecho con ❤️ por [Tu Nombre]")

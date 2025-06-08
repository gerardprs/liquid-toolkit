import datetime as dt
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from sklearn.decomposition import PCA
import feedparser
from textblob import TextBlob
from fredapi import Fred
from io import BytesIO

st.set_page_config(page_title="Portafolio Python", layout="wide")
st.title("📚 Mi Portafolio de Mini‐Proyectos Integrados")

st.markdown(
    """
    Este dashboard ejecuta **3 mini‐proyectos** sin salir de Streamlit:
    1️⃣ **Macro Semáforo Diario**  
    2️⃣ **Benchmark Calculator**  
    3️⃣ **News‐Alert Sentiment**  
    """
)

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Macro Semáforo Diario",
    "2️⃣ Benchmark Calculator",
    "3️⃣ News‐Alert Sentiment"
])

# ==== TAB 1: Macro Semáforo Diario ====
with tab1:
    st.header("1️⃣ Macro Semáforo Diario")
    st.write(
        "- Descarga 5 series FRED\n"
        "- Calcula z-score histórico y asigna semáforos 🔴🟡🟢\n"
        "- Genera un Excel descargable"
    )

    if st.button("▶️ Ejecutar Macro Semáforo"):
        # --- 1) Descarga series FRED ---
        fred = Fred(api_key="62deb3b46aa3632a30ee4f2885c1f32a")
        SERIES = {
            "NAPMNOI":"PMI manufacturero (ISM)",
            "PCEPI":  "PCE YoY",
            "UNRATE": "Tasa desempleo",
            "T5YIFR":"Breakeven 5y5y",
            "MANEMP":"ISM New Orders"
        }
        dfs = []
        for code,label in SERIES.items():
            s = fred.get_series(code, observation_start="2000-01-01")
            s.name = label
            dfs.append(s)
        df = pd.concat(dfs, axis=1).dropna()

        # --- 2) Calcular z-score y semáforos ---
        z = (df - df.mean())/df.std()
        last = z.iloc[-1]
        bins = [-float("inf"), -1, 1, float("inf")]
        flags = pd.cut(last, bins=bins, labels=["🔴 Rojo","🟡 Ámbar","🟢 Verde"])
        out = pd.DataFrame({
            "Indicador": last.index,
            "Z-score":   last.round(2).values,
            "Semaphore": flags.values
        })

        # --- 3) Ofrecer descarga de Excel ---
        towrite = BytesIO()
        out.to_excel(towrite, index=False, engine="xlsxwriter")
        towrite.seek(0)
        st.success("✅ Informe listo")
        st.download_button(
            "⬇️ Descargar Excel",
            data=towrite,
            file_name=f"macro_flag_{dt.datetime.utcnow():%Y%m%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==== TAB 2: Benchmark Calculator ====
with tab2:
    st.header("2️⃣ Benchmark Calculator")
    st.write(
        "- Sube un CSV con columnas `ticker,weight` (peso en decimales)\n"
        "- Calcula Tracking Error & Information Ratio\n"
        "- Descarga el reporte Excel"
    )
    uploaded = st.file_uploader("📁 portfolio.csv", type="csv")
    if uploaded:
        df_port = pd.read_csv(uploaded).set_index("ticker")["weight"]
        BENCH = ["SPY","AGG","QQQ"]
        prices = yf.download(list(df_port.index)+BENCH, period="1y", progress=False)["Close"].dropna(how="all")
        rets = prices.pct_change().dropna()
        port_ret  = (rets[df_port.index] * df_port).sum(axis=1)
        bench_ret = rets[BENCH].mean(axis=1)
        # TE / IR
        excess = port_ret.align(bench_ret, join="inner")[0] - bench_ret.align(port_ret, join="inner")[0]
        te = excess.std()* (252**0.5)
        ir = (excess.mean()*252) / te if te else float("nan")

        # Preparar reporte
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            pd.DataFrame({"Tracking Error": [te], "Info Ratio": [ir]}).to_excel(writer, sheet_name="Metrics", index=False)
            prices.to_excel(writer, sheet_name="Prices")
        buf.seek(0)

        st.metric("Tracking Error (ann.)", f"{te:.2%}")
        st.metric("Information Ratio", f"{ir:.2f}")
        st.success("✅ Reporte listo")
        st.download_button(
            "⬇️ Descargar Benchmark Report",
            data=buf,
            file_name=f"benchmark_{dt.datetime.utcnow():%Y%m%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==== TAB 3: News-Alert Sentiment ====
with tab3:
    st.header("3️⃣ News-Alert Sentiment")
    st.write(
        "- Lee titulares Reuters Business (RSS)\n"
        "- Calcula polaridad media por hora\n"
        "- Si cae bajo umbral → alerta visible"
    )
    threshold = st.slider("Umbral de alerta", -1.0, 1.0, -0.2, 0.05)
    if st.button("🔍 Ejecutar Sentiment Alert"):
        feed = feedparser.parse("http://feeds.reuters.com/reuters/businessNews")
        rows = []
        for e in feed.entries[:50]:
            ts = dt.datetime(*e.published_parsed[:6]) if hasattr(e, "published_parsed") else dt.datetime.utcnow()
            score = TextBlob(e.title).sentiment.polarity
            rows.append({"time":ts, "score":score})
        df_sent = pd.DataFrame(rows).set_index("time").resample("H").mean().fillna(0)
        last = df_sent["score"].iloc[-1]
        st.line_chart(df_sent["score"])
        if last < threshold:
            st.error(f"⚠️ Sentiment {last:.2f} por debajo de umbral {threshold:.2f}")
        else:
            st.success(f"✅ Sentiment OK: {last:.2f}")

st.markdown("---")
st.caption("🔧 Proyectos integrados — Solo Python y Streamlit")

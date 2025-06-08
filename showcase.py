import datetime as dt
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import numpy as np
from io import BytesIO
import feedparser
from textblob import TextBlob

st.set_page_config(page_title="Portafolio Mini-Proyectos", layout="wide")
st.title("📚 Portafolio de Mini-Proyectos (sin FRED)")

st.markdown("""
Este dashboard integra **3 mini-proyectos** totalmente funcionales:

1️⃣ **Macro Semáforo Diario** (con yields de Tesoro)  
2️⃣ **Benchmark Calculator**  
3️⃣ **News-Alert Sentiment**  
""")

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Macro Semáforo Diario",
    "2️⃣ Benchmark Calculator",
    "3️⃣ News-Alert Sentiment"
])

# ---------------------------------------------------
# 1️⃣ Macro Semáforo Diario (con yields proxy)
# ---------------------------------------------------
with tab1:
    st.header("1️⃣ Macro Semáforo Diario")
    st.write(
        "- Usamos tres tickers de yield de Tesoro: ^TNX (10a), ^FVX (5a), ^IRX (3m)\n"
        "- Calculamos z-score de las últimas cotas\n"
        "- Semáforo: 🔴 Rojo / 🟡 Ámbar / 🟢 Verde\n"
        "- Excel descargable"
    )
    if st.button("▶️ Ejecutar Macro Semáforo"):
        # 1) Descargar precios
        tickers = ["^TNX","^FVX","^IRX"]
        df = yf.download(tickers, period="5y", progress=False)["Close"].dropna(how="all")
        df.columns = ["Yield 10a","Yield 5a","Yield 3m"]

        # 2) z-score
        z = (df - df.mean())/df.std()
        last = z.iloc[-1]
        bins = [-np.inf, -1, 1, np.inf]
        flags = pd.cut(last, bins=bins, labels=["🔴 Rojo","🟡 Ámbar","🟢 Verde"])

        # 3) tabla
        out = pd.DataFrame({
            "Serie": last.index,
            "Z-score": last.round(2).values,
            "Semáforo": flags.values
        })

        # 4) descarga
        buf = BytesIO()
        out.to_excel(buf, index=False, engine="xlsxwriter")
        buf.seek(0)
        st.success("✅ Informe listo")
        st.download_button(
            "⬇️ Descargar Excel",
            data=buf,
            file_name=f"macro_semaforo_{dt.datetime.utcnow():%Y%m%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------------------------------------------------
# 2️⃣ Benchmark Calculator
# ---------------------------------------------------
with tab2:
    st.header("2️⃣ Benchmark Calculator")
    st.write(
        "- Sube un CSV con columnas `ticker,weight` (peso en decimales)\n"
        "- Descarga 1 año de precios (yfinance) y calcula:\n"
        "   - Tracking Error anualizado\n"
        "   - Information Ratio\n"
        "- Excel descargable con métricas y precios brutos"
    )
    uploaded = st.file_uploader("📁 Sube tu portfolio.csv", type="csv")
    if uploaded:
        df_port = pd.read_csv(uploaded).set_index("ticker")["weight"]
        bench = ["SPY","AGG","QQQ"]
        tickers = list(df_port.index) + bench
        prices = yf.download(tickers, period="1y", progress=False)["Close"].dropna(how="all")
        rets = prices.pct_change().dropna()
        port_ret  = (rets[df_port.index] * df_port).sum(axis=1)
        bench_ret = rets[bench].mean(axis=1)
        exc = port_ret.align(bench_ret, join="inner")[0] - bench_ret.align(port_ret, join="inner")[0]
        te = exc.std()*np.sqrt(252)
        ir = (exc.mean()*252)/te if te else float("nan")

        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            pd.DataFrame({"Tracking Error":[te],"Info Ratio":[ir]}).to_excel(writer, sheet_name="Metrics", index=False)
            prices.to_excel(writer, sheet_name="Prices")
        buf.seek(0)

        st.metric("Tracking Error", f"{te:.2%}")
        st.metric("Information Ratio", f"{ir:.2f}")
        st.success("✅ Reporte listo")
        st.download_button(
            "⬇️ Descargar Reporte",
            data=buf,
            file_name=f"benchmark_{dt.datetime.utcnow():%Y%m%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------------------------------------------------
# 3️⃣ News-Alert Sentiment
# ---------------------------------------------------
with tab3:
    st.header("3️⃣ News-Alert Sentiment")
    st.write(
        "- RSS de Reuters Business\n"
        "- Polaridad hourly con TextBlob\n"
        "- Gráfico + Alerta si cae bajo umbral"
    )
    threshold = st.slider("Umbral de alerta", -1.0, 1.0, -0.2, 0.05)
    if st.button("🔍 Ejecutar Sentiment"):
        feed = feedparser.parse("http://feeds.reuters.com/reuters/businessNews")
        rows = []
        for e in feed.entries[:50]:
            ts = dt.datetime(*e.published_parsed[:6]) if hasattr(e, "published_parsed") else dt.datetime.utcnow()
            score = TextBlob(e.title).sentiment.polarity
            rows.append({"time":ts, "score":score})
        df_sent = pd.DataFrame(rows).set_index("time").resample("H").mean().fillna(0)
        last = df_sent["score"].iloc[-1]

        st.line_chart(df_sent["score"], height=200)
        if last < threshold:
            st.error(f"⚠️ Sentiment {last:.2f} < {threshold:.2f}")
        else:
            st.success(f"✅ Sentiment OK: {last:.2f}")

st.markdown("---")
st.caption("✅ Todo corre en un solo entorno: Streamlit + yfinance + pandas + feedparser + textblob")


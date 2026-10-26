
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Paribu Paper Bot",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Paribu — Sanal Al-Sat Laboratuvarı")
st.caption(
    "Sürüm 0.2 • Yalnızca simülasyon • "
    "Gerçek emir göndermez • Kâr garantisi yoktur"
)

with st.sidebar:
    st.header("⚙️ Simülasyon ayarları")
    initial_cash = st.number_input(
        "Başlangıç bakiyesi (TL)",
        min_value=1000.0,
        value=10000.0,
        step=1000.0
    )
    allocation = st.slider(
        "İşlem başına bakiye (%)", 5, 100, 25
    )
    short_window = st.slider(
        "Kısa hareketli ortalama", 3, 30, 7
    )
    long_window = st.slider(
        "Uzun hareketli ortalama", 10, 100, 25
    )
    fee_pct = st.number_input(
        "Tek yön komisyon (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.2,
        step=0.05
    )
    slippage_pct = st.number_input(
        "Kayma (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.05
    )
    stop_loss = st.slider(
        "Zarar durdur (%)", 1, 30, 5
    )
    take_profit = st.slider(
        "Kâr al (%)", 1, 100, 10
    )
    use_sample = st.checkbox(
        "Örnek fiyat verisi kullan",
        value=True
    )
    uploaded_file = st.file_uploader(
        "Veya fiyat CSV dosyası yükle",
        type=["csv"]
    )

st.info(
    "Bu uygulama eğitim amaçlıdır. Örnek veriler gerçek "
    "piyasa fiyatları değildir. CSV kullanırsan sonuçlar "
    "yüklediğin veriye bağlıdır."
)

def make_sample_data():
    rng = np.random.default_rng(42)
    periods = 300
    changes = rng.normal(0.0005, 0.018, periods)
    prices = 100 + np.cumsum(changes * 100)
    prices = np.maximum(prices, 1)
    return pd.DataFrame({
        "Date": pd.date_range(
            end=pd.Timestamp.today().normalize(),
            periods=periods,
            freq="h"
        ),
        "Close": prices
    })

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = [
            str(c).strip().lower() for c in df.columns
        ]

        if "close" not in df.columns:
            st.error(
                "CSV dosyasında Close adlı kapanış fiyatı "
                "sütunu bulunmalı."
            )
            st.stop()

        if "date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["date"], errors="coerce"
            )
        elif "timestamp" in df.columns:
            df["Date"] = pd.to_datetime(
                df["timestamp"], errors="coerce"
            )
        else:
            df["Date"] = np.arange(len(df))

        df["Close"] = pd.to_numeric(
            df["close"], errors="coerce"
        )
        df = df[["Date", "Close"]].dropna()
        df = df[df["Close"] > 0].reset_index(drop=True)

    elif use_sample:
        df = make_sample_data()
    else:
        st.warning(
            "Devam etmek için CSV yükle veya örnek veriyi aç."
        )
        st.stop()

    if len(df) < 30:
        st.error("Simülasyon için en az 30 geçerli fiyat satırı gerekli.")
        st.stop()

    if short_window >= long_window:
        st.error(
            "Kısa hareketli ortalama, uzun ortalamadan küçük olmalı."
        )
        st.stop()

    df["SMA_short"] = df["Close"].rolling(
        short_window
    ).mean()
    df["SMA_long"] = df["Close"].rolling(
        long_window
    ).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    df["Signal"] = (
        (df["SMA_short"] > df["SMA_long"]) &
        (df["RSI"] < 70)
    ).astype(int)

    cash = float(initial_cash)
    quantity = 0.0
    entry_price = 0.0
    entry_cost = 0.0
    trades = []
    equity_history = []

    fee = fee_pct / 100
    slippage = slippage_pct / 100
    stop = stop_loss / 100
    target = take_profit / 100

    for i in range(len(df)):
        price = float(df.loc[i, "Close"])
        signal = int(df.loc[i, "Signal"])
        date = df.loc[i, "Date"]

        if quantity > 0:
            exit_reason = None

            if price <= entry_price * (1 - stop):
                exit_reason = "Zarar durdur"
            elif price >= entry_price * (1 + target):
                exit_reason = "Kâr al"
            elif signal == 0:
                exit_reason = "Gösterge çıkışı"

            if exit_reason:
                sell_price = price * (1 - slippage)
                proceeds = quantity * sell_price
                sell_fee = proceeds * fee
                net_proceeds = proceeds - sell_fee
                pnl = net_proceeds - entry_cost

                cash += net_proceeds
                trades.append({
                    "Tarih": date,
                    "İşlem": "SAT",
                    "Fiyat": sell_price,
                    "Miktar": quantity,
                    "Net sonuç (TL)": pnl,
                    "Neden": exit_reason
                })

                quantity = 0.0
                entry_price = 0.0
                entry_cost = 0.0

        elif signal == 1 and cash > 0:
            budget = cash * allocation / 100
            buy_price = price * (1 + slippage)
            buy_fee = budget * fee
            spend = min(budget, cash)

            quantity = (spend - buy_fee) / buy_price

            if quantity > 0:
                entry_cost = spend
                entry_price = buy_price
                cash -= spend

                trades.append({
                    "Tarih": date,
                    "İşlem": "AL",
                    "Fiyat": buy_price,
                    "Miktar": quantity,
                    "Net sonuç (TL)": 0.0,
                    "Neden": "Alış sinyali"
                })

        equity = cash + quantity * price
        equity_history.append({
            "Tarih": date,
            "Portföy (TL)": equity,
            "Fiyat": price
        })

    if quantity > 0:
        last_price = float(df.iloc[-1]["Close"])
        marked_value = quantity * last_price * (1 - slippage)
        marked_value *= (1 - fee)
        final_equity = cash + marked_value
    else:
        final_equity = cash

    profit = final_equity - initial_cash
    profit_pct = profit / initial_cash * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Başlangıç bakiyesi", f"{initial_cash:,.2f} TL")
    col2.metric(
        "Tahmini son portföy",
        f"{final_equity:,.2f} TL",
        f"{profit_pct:+.2f}%"
    )
    col3.metric("Tahmini sonuç", f"{profit:+,.2f} TL")

    history = pd.DataFrame(equity_history)
    fig = px.line(
        history,
        x="Tarih",
        y="Portföy (TL)",
        title="Sanal portföy değişimi"
    )
    st.plotly_chart(fig, use_container_width=True)

    price_fig = px.line(
        df,
        x="Date",
        y=["Close", "SMA_short", "SMA_long"],
        title="Fiyat ve hareketli ortalamalar"
    )
    st.plotly_chart(price_fig, use_container_width=True)

    st.subheader("📋 İşlem geçmişi")
    if trades:
        trade_df = pd.DataFrame(trades)
        st.dataframe(trade_df, use_container_width=True)
        st.download_button(
            "İşlem geçmişini CSV indir",
            trade_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="paper_bot_islemleri.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "Henüz işlem oluşmadı. Farklı gösterge ayarları "
            "veya başka bir fiyat CSV dosyası deneyebilirsin."
        )

    st.caption(
        "Not: Bu basit bir geçmiş veri simülasyonudur; "
        "gelecekteki performansı göstermez. Açık pozisyonun "
        "son değeri tahminidir. Vergiler ve tüm piyasa "
        "koşulları hesaba katılmamış olabilir."
    )

except Exception as e:
    st.error(f"Veri işlenirken hata oluştu: {e}")

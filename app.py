
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ccxt
from datetime import datetime, timezone

st.set_page_config(
    page_title="Paribu Paper Bot",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Paribu Çoklu Kripto Paper Bot")
st.caption(
    "Sanal analiz ve backtest sistemi. "
    "Gerçek emir göndermez. Kâr garantisi yoktur."
)

with st.sidebar:
    st.header("⚙️ Ayarlar")

    timeframe = st.selectbox(
        "Mum zaman dilimi",
        ["15m", "1h", "4h"],
        index=1
    )

    top_n = st.slider(
        "Taranacak en yüksek hacimli TRY paritesi",
        5, 30, 10
    )

    starting_balance = st.number_input(
        "Sanal sermaye (TL)",
        min_value=1000.0,
        value=10000.0,
        step=1000.0
    )

    risk_pct = st.slider(
        "İşlem başına risk (%)",
        0.25, 2.0, 0.5, 0.25
    )

    daily_loss_pct = st.slider(
        "Günlük zarar sınırı (%)",
        1.0, 10.0, 3.0, 0.5
    )

    fee_pct = st.number_input(
        "Tek yön komisyon (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.05
    )

    stop_pct = st.slider(
        "Zarar durdur (%)",
        1.0, 8.0, 2.0, 0.5
    )

    reward_ratio = st.slider(
        "Hedef risk/kazanç oranı",
        1.0, 4.0, 2.0, 0.5
    )

    run_scan = st.button(
        "🔎 Piyasayı tara ve test et",
        type="primary"
    )


def indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(
        span=20, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    fast = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    slow = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = fast - slow
    df["signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["volume_avg"] = df["volume"].rolling(20).mean()

    return df


def buy_signal(row):
    return bool(
        row["close"] > row["ema20"] > row["ema50"]
        and 45 <= row["rsi"] <= 65
        and row["macd"] > row["signal"]
        and row["volume"] >= row["volume_avg"]
    )


def backtest(df, balance, risk, stop, rr, fee):
    df = indicators(df).dropna().reset_index(drop=True)

    cash = float(balance)
    position = None
    trades = []
    equity = []

    daily_start = cash
    current_day = None
    daily_halt = False

    for i in range(1, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        day = pd.to_datetime(row["time"], unit="ms").date()

        if current_day != day:
            current_day = day
            daily_start = cash
            daily_halt = False

        if daily_start > 0:
            daily_change = (cash - daily_start) / daily_start * 100
            if daily_change <= -daily_loss_pct:
                daily_halt = True

        if position is not None:
            position["peak"] = max(position["peak"], price)

            trailing_stop = position["peak"] * 0.985
            stop_price = max(position["stop"], trailing_stop)
            target_price = position["target"]

            reason = None

            if price <= stop_price:
                reason = "Stop"
            elif price >= target_price:
                reason = "Hedef kâr"
            elif row["macd"] < row["signal"]:
                reason = "Çıkış sinyali"

            if reason:
                gross = (price - position["entry"]) * position["qty"]
                exit_fee = price * position["qty"] * fee / 100
                pnl = gross - exit_fee
                cash += pnl

                trades.append({
                    "Tarih": pd.to_datetime(
                        row["time"], unit="ms"
                    ),
                    "İşlem": "SAT",
                    "Fiyat": price,
                    "Net sonuç (TL)": pnl,
                    "Neden": reason
                })

                position = None

        if position is None and not daily_halt and buy_signal(row):
            entry = price
            stop_price = entry * (1 - stop / 100)
            risk_amount = cash * risk / 100
            risk_per_unit = entry - stop_price

            if risk_per_unit > 0 and cash > 0:
                qty_by_risk = risk_amount / risk_per_unit
                qty_by_cash = cash / (
                    entry * (1 + fee / 100)
                )
                qty = min(qty_by_risk, qty_by_cash)

                if qty > 0:
                    buy_fee = entry * qty * fee / 100
                    cash -= buy_fee

                    position = {
                        "entry": entry,
                        "qty": qty,
                        "stop": stop_price,
                        "target": entry * (
                            1 + (stop / 100) * rr
                        ),
                        "peak": entry
                    }

                    trades.append({
                        "Tarih": pd.to_datetime(
                            row["time"], unit="ms"
                        ),
                        "İşlem": "AL",
                        "Fiyat": entry,
                        "Net sonuç (TL)": -buy_fee,
                        "Neden": "Trend ve momentum"
                    })

        equity.append({
            "Tarih": pd.to_datetime(row["time"], unit="ms"),
            "Sanal nakit": cash
        })

    return pd.DataFrame(trades), pd.DataFrame(equity), cash


if run_scan:
    with st.spinner("Paribu piyasaları taranıyor..."):
        try:
            exchange = ccxt.paribu({
                "enableRateLimit": True,
                "timeout": 20000
            })

            markets = exchange.load_markets()
            tickers = exchange.fetch_tickers()

            candidates = []

            for symbol, market in markets.items():
                if not market.get("active", True):
                    continue

                if market.get("quote") != "TRY":
                    continue

                ticker = tickers.get(symbol, {})
                volume = ticker.get("quoteVolume")

                if volume is None:
                    last = ticker.get("last")
                    base_volume = ticker.get("baseVolume")
                    if last and base_volume:
                        volume = last * base_volume

                if volume and ticker.get("last"):
                    candidates.append({
                        "symbol": symbol,
                        "volume": float(volume),
                        "last": float(ticker["last"])
                    })

            candidates.sort(
                key=lambda x: x["volume"],
                reverse=True
            )
            candidates = candidates[:top_n]

            if not candidates:
                st.error(
                    "TRY pariteleri bulunamadı. "
                    "Borsa verisi veya sembol biçimi kontrol edilmeli."
                )
                st.stop()

            scan_rows = []
            all_trades = []
            all_equity = []

            progress = st.progress(0)

            for idx, item in enumerate(candidates):
                symbol = item["symbol"]

                try:
                    candles = exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        limit=300
                    )

                    if not candles or len(candles) < 80:
                        continue

                    df = pd.DataFrame(
                        candles,
                        columns=[
                            "time", "open", "high",
                            "low", "close", "volume"
                        ]
                    )

                    df = indicators(df)
                    last = df.iloc[-1]

                    signal = (
                        "AL sinyali"
                        if buy_signal(last)
                        else "BEKLE"
                    )

                    trades, equity, final_cash = backtest(
                        df,
                        starting_balance / max(len(candidates), 1),
                        risk_pct,
                        stop_pct,
                        reward_ratio,
                        fee_pct
                    )

                    scan_rows.append({
                        "Parite": symbol,
                        "Son fiyat": item["last"],
                        "24s hacim (TL)": item["volume"],
                        "RSI": round(float(last["rsi"]), 2),
                        "Sinyal": signal,
                        "Test sonucu (TL)": round(
                            final_cash -
                            starting_balance / max(len(candidates), 1),
                            2
                        ),
                        "Tamamlanan satış": int(
                            (trades["İşlem"] == "SAT").sum()
                        ) if not trades.empty else 0
                    })

                    if not trades.empty:
                        trades.insert(0, "Parite", symbol)
                        all_trades.append(trades)

                    progress.progress((idx + 1) / len(candidates))

                except Exception as err:
                    scan_rows.append({
                        "Parite": symbol,
                        "Sinyal": "Veri hatası",
                        "Hata": str(err)[:180]
                    })

            st.session_state["scan"] = pd.DataFrame(scan_rows)
            st.session_state["trades"] = (
                pd.concat(all_trades, ignore_index=True)
                if all_trades else pd.DataFrame()
            )

            st.success("Tarama tamamlandı.")

        except Exception as err:
            st.error(
                "Paribu veri bağlantısı başarısız oldu. "
                "Hata ayrıntısı:"
            )
            st.code(str(err))


if "scan" in st.session_state:
    st.subheader("🔍 Çoklu kripto tarama sonuçları")
    st.dataframe(
        st.session_state["scan"],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📒 Sanal işlem geçmişi")
    trades = st.session_state["trades"]

    if not trades.empty:
        st.dataframe(
            trades,
            use_container_width=True,
            hide_index=True
        )

        sells = trades[trades["İşlem"] == "SAT"]
        net = sells["Net sonuç (TL)"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Tamamlanan satış", len(sells))
        c2.metric("Net satış sonucu", f"{net:.2f} TL")
        c3.metric(
            "Kârlı satış oranı",
            f"{(sells['Net sonuç (TL)'] > 0).mean() * 100:.1f}%"
            if len(sells) else "Henüz yok"
        )
    else:
        st.info("Henüz tamamlanmış sanal işlem yok.")

st.divider()
st.caption(
    "Not: Geçmiş test sonuçları gelecekteki getiriyi garanti etmez. "
    "Bu sürüm gerçek emir göndermez. "
    "İz süren stop ve günlük zarar kontrolü, "
    "gerçek piyasada kesin fiyat garantisi vermez."
)

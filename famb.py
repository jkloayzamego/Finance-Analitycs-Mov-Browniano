
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================
# 1. DESCARGAR DATOS HISTORICOS
# =====================================

symbol = "AAPL"

stock_data = yf.download(
    symbol,
    start="2022-01-01",
    end="2026-01-01",
    auto_adjust=True
)

if stock_data.empty:
    raise ValueError("No se pudieron descargar los datos.")

# Compatibilidad con columnas MultiIndex
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

prices = stock_data["Close"].dropna()

# =====================================
# 2. CALCULAR RENDIMIENTOS
# =====================================

returns = np.log(prices / prices.shift(1)).dropna()

mu = returns.mean() * 252
sigma = returns.std() * np.sqrt(252)

S0 = float(prices.iloc[-1])

print("Precio inicial:", round(S0, 2))
print("Rendimiento anual:", round(mu * 100, 2), "%")
print("Volatilidad anual:", round(sigma * 100, 2), "%")

# =====================================
# 3. MOVIMIENTO BROWNIANO GEOMETRICO
# =====================================

np.random.seed(42)

dias = 252
simulaciones = 100

dt = 1 / 252

Z = np.random.normal(
    0, 1,
    (dias, simulaciones)
)

precios = np.zeros((dias + 1, simulaciones))
precios[0] = S0

for t in range(1, dias + 1):

    precios[t] = precios[t-1] * np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z[t-1]
    )

# =====================================
# 4. CALCULAR INTERVALOS DE CONFIANZA
# =====================================

media = np.mean(precios, axis=1)

limite_inferior = np.percentile(precios, 2.5, axis=1)
limite_superior = np.percentile(precios, 97.5, axis=1)

# =====================================
# 5. CREAR GRAFICO ANIMADO
# =====================================

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        "Precio historico de Apple",
        "Simulacion Movimiento Browniano Geometrico"
    ),
    vertical_spacing=0.15
)

# Grafico historico
fig.add_trace(
    go.Scatter(
        x=prices.index,
        y=prices.values,
        mode="lines",
        name="Precio historico"
    ),
    row=1, col=1
)

# Precio inicial de simulacion
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[S0],
        mode="lines",
        name="Trayectorias GBM"
    ),
    row=2, col=1
)

# Media esperada
fig.add_trace(
    go.Scatter(
        x=[0],
        y=[S0],
        mode="lines",
        name="Precio medio"
    ),
    row=2, col=1
)

# =====================================
# 6. CREAR FRAMES DE ANIMACION
# =====================================

frames = []

for i in range(1, dias + 1):

    trazas = []

    # 100 trayectorias
    for j in range(simulaciones):

        trazas.append(
            go.Scatter(
                x=list(range(i + 1)),
                y=precios[:i + 1, j],
                mode="lines",
                line=dict(width=1),
                showlegend=False
            )
        )

    # Media
    trazas.append(
        go.Scatter(
            x=list(range(i + 1)),
            y=media[:i + 1],
            mode="lines",
            line=dict(width=3),
            name="Media esperada"
        )
    )

    frames.append(
        go.Frame(
            data=trazas,
            traces=list(range(1, simulaciones + 2)),
            name=str(i)
        )
    )

fig.frames = frames

# =====================================
# 7. CONFIGURAR ANIMACION
# =====================================

fig.update_layout(
    title="APPLE (AAPL) - SIMULACION ESTOCASTICA",
    height=850,
    template="plotly_dark",
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="▶ INICIAR",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(
                                duration=40,
                                redraw=True
                            ),
                            transition=dict(duration=0),
                            fromcurrent=True
                        )
                    ]
                ),
                dict(
                    label="⏸ PAUSAR",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0),
                            mode="immediate"
                        )
                    ]
                )
            ]
        )
    ],
    xaxis2_title="Dias futuros",
    yaxis_title="Precio (USD)",
    showlegend=True
)

fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
fig.update_yaxes(title_text="Precio (USD)", row=2, col=1)

# =====================================
# 8. MOSTRAR Y GUARDAR
# =====================================

fig.show()

fig.write_html("movimiento_browniano_AAPL.html")

print("Simulacion terminada.")
print("Archivo guardado: movimiento_browniano_AAPL.html")

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Aleta de resfriamento - seleção de material", layout="wide")
st.title("Simulador de aleta de resfriamento — seleção de material")
st.caption("Aleta tipo pino em regime permanente: problema de valor de contorno (PVC).")


# ----------------------------- Utilidades de exibição -----------------------------
def mostrar_grafico(ch):
    try:
        st.altair_chart(ch, width="stretch")
    except Exception:
        st.altair_chart(ch, use_container_width=True)


def mostrar_tabela(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def dominio(valores, folga=0.08):
    """Domínio do eixo y ajustado à variação real dos dados."""
    vmin, vmax = float(np.min(valores)), float(np.max(valores))
    if np.isclose(vmin, vmax):
        d = abs(vmin) * 0.05 + 1e-9
        return [vmin - d, vmax + d]
    d = (vmax - vmin) * folga
    return [vmin - d, vmax + d]


# ----------------------------- Banco de materiais -----------------------------
C_MAT, C_K, C_RHO, C_PRECO, C_CORR = (
    "Material",
    "k (W/m·K)",
    "ρ (kg/m³)",
    "Preço (R$/kg)",
    "Resiste a H₂SO₄",
)

MATERIAIS_PADRAO = pd.DataFrame(
    [
        {C_MAT: "Cobre", C_K: 401.0, C_RHO: 8933.0, C_PRECO: 55.0, C_CORR: False},
        {C_MAT: "Alumínio 6061", C_K: 167.0, C_RHO: 2700.0, C_PRECO: 25.0, C_CORR: False},
        {C_MAT: "Latão", C_K: 110.0, C_RHO: 8530.0, C_PRECO: 45.0, C_CORR: False},
        {C_MAT: "Aço carbono", C_K: 51.0, C_RHO: 7850.0, C_PRECO: 8.0, C_CORR: False},
        {C_MAT: "Aço inox 316", C_K: 16.3, C_RHO: 8000.0, C_PRECO: 35.0, C_CORR: True},
        {C_MAT: "Titânio grau 2", C_K: 21.9, C_RHO: 4510.0, C_PRECO: 250.0, C_CORR: True},
    ]
)

with st.expander("Propriedades dos materiais (editáveis)", expanded=False):
    st.caption(
        "Valores típicos aproximados — confira com a literatura e ajuste os preços à sua realidade. "
        "Você pode editar as células e adicionar novos materiais na última linha."
    )
    try:
        materiais = st.data_editor(
            MATERIAIS_PADRAO, num_rows="dynamic", hide_index=True, key="tab_mat"
        )
    except Exception:
        materiais = st.data_editor(MATERIAIS_PADRAO, num_rows="dynamic", key="tab_mat")
    materiais = materiais.dropna(subset=[C_MAT, C_K, C_RHO])
    materiais = materiais[(materiais[C_K] > 0) & (materiais[C_RHO] > 0)].reset_index(drop=True)
    materiais[C_PRECO] = materiais[C_PRECO].fillna(0.0)
    materiais[C_CORR] = materiais[C_CORR].fillna(False).astype(bool)


# ----------------------------- Modelo físico -----------------------------
def geometria(D):
    """Perímetro e área da seção de um pino de diâmetro D (m)."""
    return np.pi * D, np.pi * D**2 / 4.0


def solucao_analitica(x, L, D, k, h, theta_b, ponta_conv):
    """Solução particular do PVC: perfil θ(x), calor q_a e parâmetro m."""
    P, Ac = geometria(D)
    m = np.sqrt(h * P / (k * Ac))
    hp = h / (m * k) if ponta_conv else 0.0
    den = np.cosh(m * L) + hp * np.sinh(m * L)
    theta = theta_b * (np.cosh(m * (L - x)) + hp * np.sinh(m * (L - x))) / den
    q = np.sqrt(h * P * k * Ac) * theta_b * (np.sinh(m * L) + hp * np.cosh(m * L)) / den
    return theta, q, m


def diferencas_finitas(N, L, D, k, h, theta_b, ponta_conv):
    """PVC por diferenças finitas centrais (2ª ordem), com nó fantasma na ponta."""
    P, Ac = geometria(D)
    m2 = h * P / (k * Ac)
    dx = L / (N - 1)
    x = np.linspace(0.0, L, N)
    diag = np.full(N, -(2.0 + m2 * dx**2))
    A = np.diag(diag) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
    b = np.zeros(N)
    # base: temperatura prescrita
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = theta_b
    # ponta: -k dθ/dx = h θ (ou dθ/dx = 0 se adiabática)
    beta = 2.0 * dx * h / k if ponta_conv else 0.0
    A[-1, :] = 0.0
    A[-1, -2] = 2.0
    A[-1, -1] = -(2.0 + m2 * dx**2 + beta)
    theta = np.linalg.solve(A, b)
    q = -k * Ac * (-3.0 * theta[0] + 4.0 * theta[1] - theta[2]) / (2.0 * dx)
    return x, theta, q


def avaliar(mat, D, L, h, Tb, Tinf, ponta_conv, N, Na):
    """Resolve o PVC para um material e calcula os indicadores de desempenho."""
    k, rho, preco = float(mat[C_K]), float(mat[C_RHO]), float(mat[C_PRECO])
    theta_b = Tb - Tinf
    P, Ac = geometria(D)
    x = np.linspace(0.0, L, 401)
    theta, q, m = solucao_analitica(x, L, D, k, h, theta_b, ponta_conv)
    xf, thf, qf = diferencas_finitas(N, L, D, k, h, theta_b, ponta_conv)
    th_an_nos, _, _ = solucao_analitica(xf, L, D, k, h, theta_b, ponta_conv)
    Aa = np.pi * D * L + (Ac if ponta_conv else 0.0)
    massa = rho * Ac * L
    return {
        "nome": str(mat[C_MAT]),
        "k": k,
        "rho": rho,
        "preco": preco,
        "corr": bool(mat[C_CORR]),
        "x": x,
        "T": theta + Tinf,
        "xf": xf,
        "Tf": thf + Tinf,
        "q": q,
        "qf": qf,
        "erro_T": float(np.max(np.abs(thf - th_an_nos))),
        "erro_q": float(abs(qf - q) / q * 100.0),
        "m": m,
        "mL": m * L,
        "hmk": h / (m * k),
        "Tponta": float(theta[-1] + Tinf),
        "eta": q / (h * Aa * theta_b),
        "eps": q / (h * Ac * theta_b),
        "massa": massa,
        "custo": preco * massa,
        "q_m": q / massa,
        "q_c": q / (preco * massa) if preco > 0 else np.nan,
        "Bi": h * (D / 2.0) / k,
        "q_inf": np.sqrt(h * P * k * Ac) * theta_b,
        "q_tot": Na * q,
    }


def curva_q_L(mat, D, L, h, theta_b, ponta_conv):
    """Calor dissipado em função do comprimento da aleta."""
    k = float(mat[C_K])
    _, _, m = solucao_analitica(0.0, L, D, k, h, theta_b, ponta_conv)
    Lmax = max(2.0 * L, 3.0 / m)
    Ls = np.linspace(Lmax / 200.0, Lmax, 200)
    qs = np.array([solucao_analitica(0.0, Li, D, k, h, theta_b, ponta_conv)[1] for Li in Ls])
    return Ls, qs


# ----------------------------- Desenho da aleta -----------------------------
PALETA = [(44, 123, 182), (171, 217, 233), (255, 255, 191), (253, 174, 97), (215, 25, 28)]


def cor_T(frac):
    """Escala de cor: azul (T∞) → vermelho (T_b)."""
    f = min(max(float(frac), 0.0), 1.0) * (len(PALETA) - 1)
    i = min(int(f), len(PALETA) - 2)
    t = f - i
    a, b = PALETA[i], PALETA[i + 1]
    return "rgb({},{},{})".format(*(round(a[j] + (b[j] - a[j]) * t) for j in range(3)))


def gradiente(gid, x, T, Tb, Tinf, ativo):
    if not ativo:
        return (
            f'<linearGradient id="{gid}"><stop offset="0" stop-color="#c9d3df"/>'
            f'<stop offset="1" stop-color="#c9d3df"/></linearGradient>'
        )
    L = x[-1]
    idx = np.linspace(0, len(x) - 1, 24).astype(int)
    stops = "".join(
        f'<stop offset="{x[i] / L:.4f}" stop-color="{cor_T((T[i] - Tinf) / (Tb - Tinf))}"/>'
        for i in idx
    )
    return f'<linearGradient id="{gid}" x1="0" x2="1" y1="0" y2="0">{stops}</linearGradient>'


def svg_aletas(linhas, Tb, Tinf, h, D_mm, L_mm, ativo):
    """Parede do tanque + uma aleta por material, coloridas pelo perfil de temperatura."""
    n = max(len(linhas), 1)
    unica = n == 1
    esp = 44 if unica else 24          # espessura desenhada da aleta
    passo = 150 if unica else 66       # distância vertical entre aletas
    topo = 95 if unica else 78
    Hf = topo + passo * (n - 1) + esp / 2 + 95
    W = 900
    x0, x1 = 120, 700                  # base e ponta da aleta no desenho

    css = (
        "<style>"
        ".calor{animation:sobe 2.4s ease-in infinite;}"
        "@keyframes sobe{0%{transform:translateY(0);opacity:0}"
        "25%{opacity:1}100%{transform:translateY(-26px);opacity:0}}"
        "</style>"
    )
    defs = [
        '<pattern id="hach" width="10" height="10" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="10" '
        'stroke="#8f9eae" stroke-width="3"/></pattern>'
    ]
    defs.append(
        '<linearGradient id="brilho" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0" stop-color="#000" stop-opacity="0.18"/>'
        '<stop offset="0.3" stop-color="#fff" stop-opacity="0.35"/>'
        '<stop offset="0.55" stop-color="#fff" stop-opacity="0"/>'
        '<stop offset="1" stop-color="#000" stop-opacity="0.28"/></linearGradient>'
    )
    defs.append(
        '<radialGradient id="brilho_ponta" cx="0.4" cy="0.35" r="0.75">'
        '<stop offset="0" stop-color="#fff" stop-opacity="0.35"/>'
        '<stop offset="1" stop-color="#000" stop-opacity="0.15"/></radialGradient>'
    )
    corpo = []
    y_ini = topo - esp / 2 - 30
    y_fim = topo + passo * (n - 1) + esp / 2 + 30

    # parede quente do tanque
    corpo.append(
        f'<rect x="70" y="{y_ini}" width="{x0 - 70}" height="{y_fim - y_ini}" '
        f'fill="{cor_T(1.0) if ativo else "#c9d3df"}" opacity="0.9"/>'
    )
    corpo.append(
        f'<rect x="70" y="{y_ini}" width="{x0 - 70}" height="{y_fim - y_ini}" fill="url(#hach)" opacity="0.35"/>'
    )
    corpo.append(
        f'<line x1="{x0}" y1="{y_ini}" x2="{x0}" y2="{y_fim}" stroke="#3b4a5a" stroke-width="4"/>'
    )
    corpo.append(
        f'<text x="95" y="{y_fim + 22}" font-size="12" text-anchor="middle" fill="#4a5a6a">'
        f"parede do tanque</text>"
    )
    corpo.append(
        f'<text x="95" y="{y_fim + 38}" font-size="13" font-weight="700" text-anchor="middle" '
        f'fill="#1f2a37">T_b = {Tb:.1f} °C</text>'
    )

    for i, ln in enumerate(linhas):
        yc = topo + passo * i
        gid = f"g{i}"
        defs.append(gradiente(gid, ln["x"], ln["T"], Tb, Tinf, ativo))
        # cilindro: corpo + base circular (junto à parede) + ponta circular, em perspectiva
        rx, ry = esp * 0.25, esp / 2
        yt, yb_ = yc - ry, yc + ry
        xc = x1 - rx                       # centro da elipse da ponta
        corpo_cil = (
            f"M{x0} {yt} H{xc} V{yb_} H{x0} "
            f"A{rx} {ry} 0 0 1 {x0} {yt} Z"
        )
        corpo.append(f'<path d="{corpo_cil}" fill="url(#{gid})"/>')
        corpo.append(f'<path d="{corpo_cil}" fill="url(#brilho)"/>')
        corpo.append(
            f'<path d="M{x0} {yt} H{xc} M{x0} {yb_} H{xc} M{x0} {yb_} A{rx} {ry} 0 0 1 {x0} {yt}" '
            f'fill="none" stroke="#3b4a5a" stroke-width="2"/>'
        )
        # metade oculta da base circular (tracejada)
        corpo.append(
            f'<path d="M{x0} {yt} A{rx} {ry} 0 0 1 {x0} {yb_}" fill="none" stroke="#3b4a5a" '
            f'stroke-width="1.4" stroke-dasharray="4 3" opacity="0.8"/>'
        )
        cor_ponta = cor_T((ln["T"][-1] - Tinf) / (Tb - Tinf)) if ativo else "#c9d3df"
        corpo.append(
            f'<ellipse cx="{xc}" cy="{yc}" rx="{rx}" ry="{ry}" fill="{cor_ponta}" '
            f'stroke="#3b4a5a" stroke-width="2"/>'
        )
        corpo.append(
            f'<ellipse cx="{xc}" cy="{yc}" rx="{rx}" ry="{ry}" fill="url(#brilho_ponta)"/>'
        )
        # calor saindo por convecção (opacidade ∝ excesso local de temperatura)
        if ativo:
            for j in range(9):
                fr = (j + 0.5) / 9
                xp = x0 + fr * (x1 - x0)
                th = (np.interp(fr * ln["x"][-1], ln["x"], ln["T"]) - Tinf) / (Tb - Tinf)
                y0 = yc - esp / 2 - 6
                path = f"M{xp:.1f} {y0:.1f} q4 -5 0 -10 q-4 -5 0 -10"
                corpo.append(
                    f'<path class="calor" d="{path}" stroke="{cor_T(th)}" stroke-width="2.6" '
                    f'fill="none" stroke-linecap="round" '
                    f'stroke-opacity="{0.15 + 0.85 * th:.2f}" style="animation-delay:{(j % 3) * 0.8:.1f}s"/>'
                )
        nome = ln["nome"]
        txt_q = f'q = {ln["q"]:.2f} W' if ativo else "q = —"
        txt_tp = f'T(L) = {ln["Tponta"]:.1f} °C' if ativo else "T(L) = —"
        fs = 15 if unica else 13
        dy = 18 if unica else 15
        corpo.append(
            f'<text x="{x1 + 16}" y="{yc - dy + 4}" font-size="{fs}" font-weight="700" '
            f'fill="#1f4e9a">{nome}</text>'
        )
        corpo.append(
            f'<text x="{x1 + 16}" y="{yc + 4}" font-size="{fs}" font-weight="700" '
            f'fill="#1f2a37">{txt_q}</text>'
        )
        corpo.append(
            f'<text x="{x1 + 16}" y="{yc + dy + 2}" font-size="12" fill="#4a5a6a">{txt_tp}</text>'
        )

    # cotas
    yc_ult = topo + passo * (n - 1) + esp / 2
    corpo.append(
        f'<line x1="{x0}" y1="{yc_ult + 22}" x2="{x1}" y2="{yc_ult + 22}" stroke="#6b7a8a" '
        f'stroke-width="1.2" marker-start="url(#seta)" marker-end="url(#seta)"/>'
    )
    corpo.append(
        f'<text x="{(x0 + x1) / 2}" y="{yc_ult + 38}" font-size="12" text-anchor="middle" '
        f'fill="#4a5a6a">L = {L_mm:.1f} mm   ·   D = {D_mm:.1f} mm   ·   x = 0 na base</text>'
    )
    defs.append(
        '<marker id="seta" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#6b7a8a"/></marker>'
    )

    # barra de cores
    yb = Hf - 34
    defs.append(
        '<linearGradient id="barra" x1="0" x2="1">'
        + "".join(f'<stop offset="{t:.2f}" stop-color="{cor_T(1 - t)}"/>' for t in np.linspace(0, 1, 11))
        + "</linearGradient>"
    )
    corpo.append(f'<rect x="300" y="{yb}" width="300" height="12" rx="3" fill="url(#barra)" stroke="#8f9eae"/>')
    corpo.append(
        f'<text x="292" y="{yb + 11}" font-size="12" text-anchor="end" fill="#4a5a6a">T_b = {Tb:.1f} °C</text>'
    )
    corpo.append(
        f'<text x="608" y="{yb + 11}" font-size="12" fill="#4a5a6a">T∞ = {Tinf:.1f} °C</text>'
    )

    # ar ambiente
    corpo.append(
        f'<text x="{W - 20}" y="28" font-size="13" font-weight="700" text-anchor="end" fill="#1f2a37">'
        f"Ar ambiente</text>"
    )
    corpo.append(
        f'<text x="{W - 20}" y="46" font-size="12" text-anchor="end" fill="#4a5a6a">'
        f"T∞ = {Tinf:.1f} °C · h = {h:.1f} W/m²·K</text>"
    )

    return (
        f'<svg viewBox="0 0 {W} {Hf:.0f}" style="width:100%;max-width:900px;display:block;margin:auto" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="sans-serif">{css}<defs>{"".join(defs)}</defs>'
        f'<rect width="{W}" height="{Hf:.0f}" rx="12" fill="#f4f7fb" stroke="#d5dde8"/>'
        f'{"".join(corpo)}</svg>'
    )


def mostrar_svg(svg, altura):
    components.html(f'<div style="margin:0">{svg}</div>', height=int(altura) + 10)


def altura_svg(n):
    unica = n == 1
    esp, passo, topo = (44, 150, 95) if unica else (24, 66, 78)
    return topo + passo * (n - 1) + esp / 2 + 95


# ----------------------------- Desenho do tanque com as aletas -----------------------------
def dimensoes_tanque(M, rho, HD):
    """Volume, diâmetro e altura de um tanque cilíndrico vertical cheio de solução."""
    V = M / rho
    Dt = (4.0 * V / (np.pi * HD)) ** (1.0 / 3.0)
    return V, Dt, HD * Dt


def arranjo_aletas(Na, Dt, Hf):
    """Fileiras horizontais × aletas por fileira, com passos parecidos nas duas direções."""
    nc = int(min(max(1, round(np.sqrt(Na * np.pi * Dt / Hf))), Na))
    nr = int(np.ceil(Na / nc))
    return nr, nc


def grad_perfil(gid, x, T, Tb, Tinf, inverter=False):
    """Gradiente com o perfil T(x) da aleta; inverter=True para aletas apontando para a esquerda."""
    idx = np.linspace(0, len(x) - 1, 16).astype(int)
    stops = "".join(
        f'<stop offset="{x[i] / x[-1]:.4f}" stop-color="{cor_T((T[i] - Tinf) / (Tb - Tinf))}"/>'
        for i in idx
    )
    x1, x2 = (1, 0) if inverter else (0, 1)
    return f'<linearGradient id="{gid}" x1="{x1}" x2="{x2}" y1="0" y2="0">{stops}</linearGradient>'


def svg_tanque(r, Tb, Tinf, D_mm, L_mm, Na, M, rho, HD):
    """Vista lateral do tanque com as aletas, vista superior e detalhe ampliado."""
    V, Dt, Ht = dimensoes_tanque(M, rho, HD)
    R = Dt / 2.0
    Df, Lf = D_mm / 1000.0, L_mm / 1000.0
    Hf = 0.8 * Ht                                   # faixa aletada: 10 % a 90 % da altura
    nr, nc = arranjo_aletas(Na, Dt, Hf)
    Sv, Sc = Hf / nr, np.pi * Dt / nc
    W, H = 900, 655
    cor_parede = cor_T(1.0)
    cor_ponta = cor_T((r["Tponta"] - Tinf) / (Tb - Tinf))

    defs = [
        grad_perfil("aD", r["x"], r["T"], Tb, Tinf),
        grad_perfil("aE", r["x"], r["T"], Tb, Tinf, inverter=True),
        '<linearGradient id="casco" x1="0" x2="1" y1="0" y2="0">'
        '<stop offset="0" stop-color="#000" stop-opacity="0.35"/>'
        '<stop offset="0.35" stop-color="#fff" stop-opacity="0.30"/>'
        '<stop offset="0.6" stop-color="#fff" stop-opacity="0"/>'
        '<stop offset="1" stop-color="#000" stop-opacity="0.40"/></linearGradient>',
        '<pattern id="hach2" width="8" height="8" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" '
        'stroke="#8f9eae" stroke-width="2.5"/></pattern>',
        '<marker id="seta2" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#6b7a8a"/></marker>',
    ]
    c = []

    # ---------------- vista lateral (elevação), em escala ----------------
    s = min(400.0 / Ht, 330.0 / (Dt + 2.0 * Lf))    # px por metro
    cx, Rs = 250.0, R * s
    ry = Rs * 0.16                                  # elipse de perspectiva do topo/fundo
    ytop = 105.0 + ry
    ybot = ytop + Ht * s
    t_px = max(Df * s, 1.2)                         # espessura desenhada (mínimo visível)

    aletas = []
    for k in range(Na):
        lin, col = divmod(k, nc)
        phi = 2.0 * np.pi * (col + 0.5 * (lin % 2)) / nc
        yk = ybot - (0.1 * Ht + (lin + 0.5) * Sv) * s
        aletas.append((np.cos(phi), np.sin(phi), yk))

    def desenha_aleta(cphi, sphi, yk, frente):
        xb = cx + Rs * sphi
        proj = Lf * s * sphi
        if abs(proj) >= 0.4:
            gid = "aD" if proj > 0 else "aE"
            c.append(
                f'<rect x="{min(xb, xb + proj):.2f}" y="{yk - t_px / 2:.2f}" width="{abs(proj):.2f}" '
                f'height="{t_px:.2f}" fill="url(#{gid})"/>'
            )
        if frente:
            rx_ = max(Df * s / 2 * abs(cphi), 0.7)
            c.append(
                f'<ellipse cx="{xb + proj:.2f}" cy="{yk:.2f}" rx="{rx_:.2f}" ry="{max(Df * s / 2, 0.9):.2f}" '
                f'fill="{cor_ponta}" stroke="#3b4a5a" stroke-width="0.4"/>'
            )

    # aletas de trás (o casco do tanque cobre o que fica atrás dele)
    for cphi, sphi, yk in aletas:
        if cphi < 0:
            desenha_aleta(cphi, sphi, yk, False)

    # casco cilíndrico
    casco = (
        f"M{cx - Rs:.1f} {ytop:.1f} V{ybot:.1f} "
        f"A{Rs:.1f} {ry:.1f} 0 0 0 {cx + Rs:.1f} {ybot:.1f} V{ytop:.1f} "
        f"A{Rs:.1f} {ry:.1f} 0 0 0 {cx - Rs:.1f} {ytop:.1f} Z"
    )
    c.append(f'<path d="{casco}" fill="{cor_parede}" opacity="0.92"/>')
    c.append(f'<path d="{casco}" fill="url(#casco)"/>')
    c.append(f'<path d="{casco}" fill="none" stroke="#3b4a5a" stroke-width="2"/>')
    c.append(
        f'<ellipse cx="{cx}" cy="{ytop:.1f}" rx="{Rs:.1f}" ry="{ry:.1f}" fill="#d9e1ea" '
        f'stroke="#3b4a5a" stroke-width="2"/>'
    )
    # faixa aletada (limites tracejados)
    for frac in (0.1, 0.9):
        yl = ybot - frac * Ht * s
        c.append(
            f'<path d="M{cx - Rs:.1f} {yl:.1f} A{Rs:.1f} {ry:.1f} 0 0 0 {cx + Rs:.1f} {yl:.1f}" '
            f'fill="none" stroke="#ffffff" stroke-width="1" stroke-dasharray="5 4" opacity="0.7"/>'
        )

    # aletas da frente
    frente = sorted([a for a in aletas if a[0] >= 0], key=lambda a: a[0])
    for cphi, sphi, yk in frente:
        desenha_aleta(cphi, sphi, yk, True)

    # cotas do tanque
    xh = cx - Rs - Lf * s - 22
    c.append(
        f'<line x1="{xh:.1f}" y1="{ytop:.1f}" x2="{xh:.1f}" y2="{ybot:.1f}" stroke="#6b7a8a" '
        f'stroke-width="1.2" marker-start="url(#seta2)" marker-end="url(#seta2)"/>'
    )
    c.append(
        f'<text x="{xh - 6:.1f}" y="{(ytop + ybot) / 2:.1f}" font-size="12" text-anchor="middle" fill="#4a5a6a" '
        f'transform="rotate(-90 {xh - 6:.1f} {(ytop + ybot) / 2:.1f})">H_t = {Ht:.2f} m</text>'
    )
    yd = ybot + ry + 20
    c.append(
        f'<line x1="{cx - Rs:.1f}" y1="{yd:.1f}" x2="{cx + Rs:.1f}" y2="{yd:.1f}" stroke="#6b7a8a" '
        f'stroke-width="1.2" marker-start="url(#seta2)" marker-end="url(#seta2)"/>'
    )
    c.append(
        f'<text x="{cx}" y="{yd + 16:.1f}" font-size="12" text-anchor="middle" fill="#4a5a6a">'
        f"D_t = {Dt:.2f} m</text>"
    )
    c.append(
        f'<text x="{cx}" y="{yd + 32:.1f}" font-size="11" text-anchor="middle" fill="#6b7a8a">'
        f"Vista lateral em escala (espessura mínima visível para aletas finas)</text>"
    )

    # região do detalhe marcada na vista lateral
    ydet = ybot - 0.5 * Ht * s
    hdet = min(max(3 * Sv * s, 16), 60)
    wdet = max(Lf * s + 10, 18)
    xdet = cx + Rs - 5
    c.append(
        f'<rect x="{xdet:.1f}" y="{ydet - hdet / 2:.1f}" width="{wdet:.1f}" height="{hdet:.1f}" '
        f'fill="none" stroke="#1f4e9a" stroke-width="1.5" stroke-dasharray="4 3"/>'
    )

    # títulos
    c.append(
        '<text x="20" y="28" font-size="14" font-weight="700" fill="#1f2a37">'
        "Tanque de mistura com as aletas</text>"
    )
    c.append(
        f'<text x="20" y="47" font-size="12" fill="#4a5a6a">V = {V:.2f} m³ '
        f"(M = {M:.0f} kg · ρ = {rho:.0f} kg/m³) · H/D = {HD:.2f}</text>"
    )
    c.append(
        f'<text x="20" y="64" font-size="12" fill="#4a5a6a">{Na} aleta{"s" if Na > 1 else ""} de {r["nome"]} · '
        f'{nr} fileira{"s" if nr > 1 else ""} × {nc} por fileira'
        f'{" (fileiras alternadas defasadas)" if nr > 1 else ""}</text>'
    )

    # ---------------- vista superior (planta) ----------------
    pcx, pcy, Rp = 730.0, 160.0, 72.0
    sp = Rp / R
    lp = max(Lf * sp, 3.0)
    tp = max(Df * sp, 1.0)
    c.append(
        f'<text x="{pcx}" y="28" font-size="13" font-weight="700" text-anchor="middle" fill="#1f2a37">'
        f"Vista superior</text>"
    )
    for col in range(nc):
        phi = 2.0 * np.pi * col / nc
        x1_, y1_ = pcx + Rp * np.sin(phi), pcy - Rp * np.cos(phi)
        x2_, y2_ = pcx + (Rp + lp) * np.sin(phi), pcy - (Rp + lp) * np.cos(phi)
        gid = f"pl{col}"
        defs.append(
            f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{x1_:.2f}" y1="{y1_:.2f}" '
            f'x2="{x2_:.2f}" y2="{y2_:.2f}"><stop offset="0" stop-color="{cor_parede}"/>'
            f'<stop offset="1" stop-color="{cor_ponta}"/></linearGradient>'
        )
        c.append(
            f'<line x1="{x1_:.2f}" y1="{y1_:.2f}" x2="{x2_:.2f}" y2="{y2_:.2f}" '
            f'stroke="url(#{gid})" stroke-width="{tp:.2f}"/>'
        )
    c.append(
        f'<circle cx="{pcx}" cy="{pcy}" r="{Rp}" fill="{cor_parede}" opacity="0.92" stroke="#3b4a5a" stroke-width="2"/>'
    )
    c.append(f'<circle cx="{pcx}" cy="{pcy}" r="{Rp - 5}" fill="#d9e1ea" stroke="#3b4a5a" stroke-width="1"/>')
    c.append(
        f'<text x="{pcx}" y="{pcy + Rp + lp + 20:.1f}" font-size="12" text-anchor="middle" fill="#4a5a6a">'
        f'{nc} aleta{"s" if nc > 1 else ""} por fileira · passo circunferencial = {Sc * 1000:.1f} mm</text>'
    )

    # ---------------- detalhe ampliado ----------------
    bx, by, bw, bh = 560.0, 285.0, 320.0, 250.0
    c.append(
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="#ffffff" stroke="#1f4e9a" '
        f'stroke-width="1.5" stroke-dasharray="4 3"/>'
    )
    c.append(
        f'<line x1="{xdet + wdet:.1f}" y1="{ydet:.1f}" x2="{bx:.1f}" y2="{by + bh / 2:.1f}" '
        f'stroke="#1f4e9a" stroke-width="1" stroke-dasharray="4 3"/>'
    )
    c.append(
        f'<text x="{bx + 12}" y="{by + 20}" font-size="13" font-weight="700" fill="#1f2a37">'
        f"Detalhe ampliado (corte da parede)</text>"
    )
    nd = int(min(3, nr))
    sd = min(160.0 / Lf, 40.0 / Df)                 # escala do detalhe pelo comprimento da aleta
    gap = Sv * sd                                   # passo vertical na escala do detalhe
    fora_escala = gap > 75.0
    if fora_escala:
        gap = 75.0                                  # passo grande: desenhado encurtado
    xw = bx + 40
    yc0 = by + 127
    c.append(f'<rect x="{xw - 22}" y="{by + 44}" width="22" height="{bh - 59}" fill="{cor_parede}" opacity="0.9"/>')
    c.append(f'<rect x="{xw - 22}" y="{by + 44}" width="22" height="{bh - 59}" fill="url(#hach2)" opacity="0.35"/>')
    c.append(f'<line x1="{xw}" y1="{by + 44}" x2="{xw}" y2="{by + bh - 15}" stroke="#3b4a5a" stroke-width="2.5"/>')
    ys = [yc0 + (j - (nd - 1) / 2) * gap for j in range(nd)]
    td = max(Df * sd, 1.5)
    for yj in ys:
        c.append(
            f'<rect x="{xw}" y="{yj - td / 2:.2f}" width="{Lf * sd:.2f}" height="{td:.2f}" '
            f'fill="url(#aD)" stroke="#3b4a5a" stroke-width="0.8"/>'
        )
    # cotas do detalhe
    y_l = ys[-1] + td / 2 + 16
    c.append(
        f'<line x1="{xw}" y1="{y_l:.1f}" x2="{xw + Lf * sd:.1f}" y2="{y_l:.1f}" stroke="#6b7a8a" '
        f'stroke-width="1.2" marker-start="url(#seta2)" marker-end="url(#seta2)"/>'
    )
    c.append(
        f'<text x="{xw + Lf * sd / 2:.1f}" y="{y_l + 15:.1f}" font-size="12" text-anchor="middle" '
        f'fill="#4a5a6a">L = {L_mm:.1f} mm</text>'
    )
    xt = xw + Lf * sd + 8
    c.append(
        f'<text x="{xt:.1f}" y="{ys[0] + 4:.1f}" font-size="12" fill="#4a5a6a">D = {D_mm:.1f} mm</text>'
    )
    if nd > 1:
        xs_ = xt + 4
        c.append(
            f'<line x1="{xs_:.1f}" y1="{ys[-2]:.1f}" x2="{xs_:.1f}" y2="{ys[-1]:.1f}" stroke="#6b7a8a" '
            f'stroke-width="1.2" marker-start="url(#seta2)" marker-end="url(#seta2)"/>'
        )
        c.append(
            f'<text x="{xs_ + 6:.1f}" y="{(ys[-2] + ys[-1]) / 2 - 3:.1f}" font-size="12" fill="#4a5a6a">'
            f"passo vertical</text>"
        )
        c.append(
            f'<text x="{xs_ + 6:.1f}" y="{(ys[-2] + ys[-1]) / 2 + 12:.1f}" font-size="12" fill="#4a5a6a">'
            f"= {Sv * 1000:.1f} mm</text>"
        )
        if fora_escala:
            c.append(
                f'<text x="{bx + 12}" y="{by + 36}" font-size="11" fill="#6b7a8a">'
                f"(passo vertical encurtado no desenho)</text>"
            )
            for j in range(nd - 1):
                ym = (ys[j] + ys[j + 1]) / 2
                c.append(
                    f'<path d="M{xw - 28} {ym - 4:.1f} l8 -4 l8 8 l8 -8 l8 8 l8 -4" fill="none" '
                    f'stroke="#3b4a5a" stroke-width="1.5"/>'
                )

    # barra de cores (T_b à esquerda, como na figura das aletas)
    yb = H - 26
    defs.append(
        '<linearGradient id="barra2" x1="0" x2="1">'
        + "".join(f'<stop offset="{t:.2f}" stop-color="{cor_T(1 - t)}"/>' for t in np.linspace(0, 1, 11))
        + "</linearGradient>"
    )
    c.append(f'<rect x="300" y="{yb}" width="300" height="12" rx="3" fill="url(#barra2)" stroke="#8f9eae"/>')
    c.append(
        f'<text x="292" y="{yb + 11}" font-size="12" text-anchor="end" fill="#4a5a6a">T_b = {Tb:.1f} °C</text>'
    )
    c.append(f'<text x="608" y="{yb + 11}" font-size="12" fill="#4a5a6a">T∞ = {Tinf:.1f} °C</text>')

    svg = (
        f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:900px;display:block;margin:auto" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif"><defs>{"".join(defs)}</defs>'
        f'<rect width="{W}" height="{H}" rx="12" fill="#f4f7fb" stroke="#d5dde8"/>'
        f'{"".join(c)}</svg>'
    )
    info = dict(V=V, Dt=Dt, Ht=Ht, nr=nr, nc=nc, Sv=Sv, Sc=Sc)
    return svg, H, info


# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Tipo de simulação")
modo = st.sidebar.radio(
    "Escolha a simulação",
    ["Aleta única", "Comparação de materiais"],
    label_visibility="collapsed",
)

st.sidebar.header("Geometria (pino)")
D_mm = st.sidebar.number_input("Diâmetro D (mm)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
L_mm = st.sidebar.number_input("Comprimento L (mm)", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)

st.sidebar.header("Condições térmicas")
Tb = st.sidebar.number_input("Temperatura da base T_b (°C)", value=80.0, step=1.0)
Tinf = st.sidebar.number_input("Temperatura do ar T∞ (°C)", value=25.0, step=1.0)
h = st.sidebar.number_input(
    "Coeficiente de convecção h (W/m²·K)", min_value=0.5, max_value=5000.0, value=25.0, step=5.0
)
ponta = st.sidebar.radio("Condição na ponta (x = L)", ["Convecção na ponta", "Ponta adiabática"])
ponta_conv = ponta == "Convecção na ponta"
Na = st.sidebar.number_input("Número de aletas", min_value=1, max_value=10000, value=100, step=10)

st.sidebar.header("Tanque (figura no fim da página)")
M_t = st.sidebar.number_input("Massa de solução no tanque M (kg)", min_value=1.0, value=5000.0, step=100.0)
rho_t = st.sidebar.number_input("Massa específica da solução ρ (kg/m³)", min_value=500.0, value=1110.0, step=10.0)
HD_t = st.sidebar.number_input("Razão altura/diâmetro do tanque H/D", min_value=0.3, max_value=5.0, value=1.5, step=0.1)

lista_mat = materiais[C_MAT].astype(str).tolist()

if modo == "Aleta única":
    st.sidebar.header("Material")
    mat_sel = st.sidebar.selectbox("Material da aleta", lista_mat, index=min(1, max(len(lista_mat) - 1, 0)))
    st.sidebar.header("Solução numérica")
    N = st.sidebar.slider("Nós da malha (diferenças finitas)", min_value=5, max_value=201, value=21, step=2)
else:
    st.sidebar.header("Materiais")
    mats_sel = st.sidebar.multiselect("Materiais a comparar", lista_mat, default=lista_mat)
    exige_corr = st.sidebar.checkbox("Exigir resistência a respingos de H₂SO₄", value=False)
    criterio = st.sidebar.selectbox(
        "Critério de escolha",
        [
            "Calor por aleta q (W)",
            "Calor por massa q/m (W/kg)",
            "Calor por custo q/C (W/R$)",
            "Efetividade ε",
            "Eficiência η",
        ],
    )
    N = 51

try:
    calcular = st.sidebar.button("Calcular", type="primary", width="stretch")
except Exception:
    calcular = st.sidebar.button("Calcular", type="primary", use_container_width=True)

# ----------------------------- Cálculo -----------------------------
if calcular:
    erro = None
    if Tb <= Tinf:
        erro = "A temperatura da base deve ser maior que a do ar (T_b > T∞) para que a aleta dissipe calor."
    elif len(lista_mat) == 0:
        erro = "Cadastre pelo menos um material na tabela."
    elif modo == "Comparação de materiais" and len(mats_sel) == 0:
        erro = "Selecione pelo menos um material para comparar."
    if erro:
        st.session_state["res"] = {"erro": erro}
    else:
        D, L = D_mm / 1000.0, L_mm / 1000.0
        params = dict(
            D_mm=D_mm, L_mm=L_mm, Tb=Tb, Tinf=Tinf, h=h, ponta=ponta, Na=int(Na), N=N,
            M_t=M_t, rho_t=rho_t, HD_t=HD_t,
        )
        if modo == "Aleta única":
            linha = materiais[materiais[C_MAT].astype(str) == mat_sel].iloc[0]
            r = avaliar(linha, D, L, h, Tb, Tinf, ponta_conv, N, int(Na))
            Ls, qs = curva_q_L(linha, D, L, h, Tb - Tinf, ponta_conv)
            st.session_state["res"] = {
                "modo": modo, "params": params, "r": r, "Ls": Ls, "qs": qs,
            }
        else:
            sub = materiais[materiais[C_MAT].astype(str).isin(mats_sel)]
            resultados, curvas = [], []
            for _, linha in sub.iterrows():
                r = avaliar(linha, D, L, h, Tb, Tinf, ponta_conv, N, int(Na))
                resultados.append(r)
                Ls, qs = curva_q_L(linha, D, L, h, Tb - Tinf, ponta_conv)
                curvas.append((r["nome"], Ls, qs))
            st.session_state["res"] = {
                "modo": modo, "params": params, "lista": resultados, "curvas": curvas,
                "criterio": criterio, "exige_corr": exige_corr,
            }

res = st.session_state.get("res")

# ----------------------------- Figura no topo -----------------------------
if res and "erro" not in res and res["modo"] == modo:
    p = res["params"]
    linhas = [res["r"]] if modo == "Aleta única" else res["lista"]
    mostrar_svg(svg_aletas(linhas, p["Tb"], p["Tinf"], p["h"], p["D_mm"], p["L_mm"], True), altura_svg(len(linhas)))
else:
    if modo == "Aleta única":
        ficticias = [{"nome": "—", "x": np.array([0, 1]), "T": np.array([0, 0]), "q": 0, "Tponta": 0}]
    else:
        ficticias = [{"nome": "—", "x": np.array([0, 1]), "T": np.array([0, 0]), "q": 0, "Tponta": 0}] * 3
    mostrar_svg(
        svg_aletas(ficticias, max(Tb, Tinf + 1), Tinf, h, D_mm, L_mm, False), altura_svg(len(ficticias))
    )

if not res:
    st.info("Defina os parâmetros na barra lateral e clique em **Calcular**.")
    st.stop()
if "erro" in res:
    st.error(res["erro"])
    st.stop()
if res["modo"] != modo:
    st.info("O tipo de simulação mudou. Clique em **Calcular** para simular neste modo.")
    st.stop()

p = res["params"]
st.caption(
    f"Resultado calculado com: D = {p['D_mm']:.1f} mm · L = {p['L_mm']:.1f} mm · "
    f"T_b = {p['Tb']:.1f} °C · T∞ = {p['Tinf']:.1f} °C · h = {p['h']:.1f} W/m²·K · "
    f"{p['ponta'].lower()} · {p['Na']} aletas"
)
ponta_conv_res = p["ponta"] == "Convecção na ponta"


# ----------------------------- Equações -----------------------------
def bloco_equacoes(r=None):
    with st.expander("Modelo matemático (PVC) e solução", expanded=False):
        st.markdown("**Equação diferencial** (balanço de energia no elemento $dx$):")
        st.latex(r"\frac{d^2\theta}{dx^2} - m^2\,\theta = 0, \qquad \theta = T - T_\infty, \qquad m^2 = \frac{hP}{kA_c} = \frac{4h}{kD}")
        st.markdown("**Condições de contorno:**")
        st.latex(r"\theta(0) = \theta_b = T_b - T_\infty")
        if ponta_conv_res:
            st.latex(r"-k\left.\frac{d\theta}{dx}\right|_{x=L} = h\,\theta(L)")
            st.markdown("**Solução particular:**")
            st.latex(r"\frac{\theta(x)}{\theta_b} = \frac{\cosh[m(L-x)] + \dfrac{h}{mk}\sinh[m(L-x)]}{\cosh(mL) + \dfrac{h}{mk}\sinh(mL)}")
            st.markdown("**Calor dissipado:**")
            st.latex(r"q_a = \sqrt{hPkA_c}\;\theta_b\;\frac{\sinh(mL) + \dfrac{h}{mk}\cosh(mL)}{\cosh(mL) + \dfrac{h}{mk}\sinh(mL)}")
        else:
            st.latex(r"\left.\frac{d\theta}{dx}\right|_{x=L} = 0")
            st.markdown("**Solução particular:**")
            st.latex(r"\frac{\theta(x)}{\theta_b} = \frac{\cosh[m(L-x)]}{\cosh(mL)}")
            st.markdown("**Calor dissipado:**")
            st.latex(r"q_a = \sqrt{hPkA_c}\;\theta_b\;\tanh(mL)")
        st.markdown("**Indicadores:**")
        st.latex(r"\eta_a = \frac{q_a}{h\,A_a\,\theta_b} \qquad \varepsilon_a = \frac{q_a}{h\,A_c\,\theta_b}")
        st.markdown("**Diferenças finitas** (nós internos e nó da ponta com nó fantasma):")
        st.latex(r"\theta_{i-1} - \left(2 + m^2\Delta x^2\right)\theta_i + \theta_{i+1} = 0")
        if r is not None:
            st.markdown("**Valores para o material selecionado:**")
            st.latex(
                rf"m = {r['m']:.3f}\ \text{{m}}^{{-1}} \qquad mL = {r['mL']:.3f} \qquad "
                rf"\frac{{h}}{{mk}} = {r['hmk']:.4f} \qquad Bi = {r['Bi']:.2e}"
            )


# ============================= MODO: ALETA ÚNICA =============================
if modo == "Aleta única":
    r = res["r"]
    c = st.columns(4)
    c[0].metric("Calor por aleta q", f"{r['q']:.3f} W")
    c[1].metric("Temperatura na ponta T(L)", f"{r['Tponta']:.2f} °C")
    c[2].metric("Eficiência η", f"{r['eta'] * 100:.1f} %")
    c[3].metric("Efetividade ε", f"{r['eps']:.1f}")
    c = st.columns(4)
    c[0].metric(f"Calor total ({p['Na']} aletas)", f"{r['q_tot']:.1f} W")
    c[1].metric("Massa por aleta", f"{r['massa'] * 1000:.2f} g")
    c[2].metric("Custo por aleta", f"R$ {r['custo']:.3f}")
    c[3].metric("mL", f"{r['mL']:.3f}")

    if r["Bi"] > 0.1:
        st.warning(f"Bi = {r['Bi']:.3f} > 0,1: a hipótese de condução unidimensional fica questionável.")
    if r["eps"] < 2:
        st.warning(f"Efetividade ε = {r['eps']:.2f} < 2: a aleta praticamente não se justifica.")

    bloco_equacoes(r)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Perfil de temperatura T(x)")
        df_an = pd.DataFrame({"x (mm)": r["x"] * 1000, "T (°C)": r["T"], "Solução": "Analítica"})
        df_fd = pd.DataFrame({"x (mm)": r["xf"] * 1000, "T (°C)": r["Tf"], "Solução": "Diferenças finitas"})
        if len(df_fd) > 41:
            df_fd = df_fd.iloc[np.linspace(0, len(df_fd) - 1, 41).astype(int)]
        ydom = dominio(np.concatenate([r["T"], r["Tf"]]))
        escala = alt.Scale(domain=["Analítica", "Diferenças finitas"], range=["#1f4e9a", "#d7191c"])
        linha_an = alt.Chart(df_an).mark_line(strokeWidth=3).encode(
            x=alt.X("x (mm):Q", title="Posição x (mm)"),
            y=alt.Y("T (°C):Q", title="Temperatura T (°C)", scale=alt.Scale(domain=ydom)),
            color=alt.Color("Solução:N", scale=escala, legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("x (mm):Q", format=".2f"), alt.Tooltip("T (°C):Q", format=".3f")],
        )
        pts_fd = alt.Chart(df_fd).mark_point(size=45, filled=True).encode(
            x="x (mm):Q", y="T (°C):Q",
            color=alt.Color("Solução:N", scale=escala),
            tooltip=[alt.Tooltip("x (mm):Q", format=".2f"), alt.Tooltip("T (°C):Q", format=".3f")],
        )
        mostrar_grafico((linha_an + pts_fd).properties(height=340))
        st.caption(
            f"Validação ({p['N']} nós): erro máximo em T = {r['erro_T']:.2e} °C · "
            f"erro em q = {r['erro_q']:.3f} %"
        )
    with col2:
        st.subheader("Calor dissipado × comprimento")
        df_L = pd.DataFrame({"L (mm)": res["Ls"] * 1000, "q (W)": res["qs"]})
        base = alt.Chart(df_L).mark_line(strokeWidth=3, color="#1f4e9a").encode(
            x=alt.X("L (mm):Q", title="Comprimento da aleta L (mm)"),
            y=alt.Y("q (W):Q", title="Calor dissipado q (W)"),
            tooltip=[alt.Tooltip("L (mm):Q", format=".1f"), alt.Tooltip("q (W):Q", format=".3f")],
        )
        assint = alt.Chart(pd.DataFrame({"q": [r["q_inf"]]})).mark_rule(
            strokeDash=[6, 4], color="#6b7a8a"
        ).encode(y="q:Q")
        atual = alt.Chart(pd.DataFrame({"L": [p["L_mm"]], "q": [r["q"]]})).mark_point(
            size=110, filled=True, color="#d7191c"
        ).encode(x="L:Q", y="q:Q")
        mostrar_grafico((base + assint + atual).properties(height=340))
        st.caption(
            f"Ponto vermelho: aleta atual. Linha tracejada: limite da aleta infinita "
            f"q∞ = {r['q_inf']:.3f} W. Acima de mL ≈ 2,5 aumentar L quase não aumenta q."
        )

# ============================= MODO: COMPARAÇÃO =============================
else:
    lista = res["lista"]
    crit = res["criterio"]
    chave = {
        "Calor por aleta q (W)": "q",
        "Calor por massa q/m (W/kg)": "q_m",
        "Calor por custo q/C (W/R$)": "q_c",
        "Efetividade ε": "eps",
        "Eficiência η": "eta",
    }[crit]

    tab = pd.DataFrame(
        [
            {
                "Material": r["nome"],
                "k (W/m·K)": r["k"],
                "q (W)": r["q"],
                f"q total, {p['Na']} aletas (W)": r["q_tot"],
                "T(L) (°C)": r["Tponta"],
                "η (%)": r["eta"] * 100,
                "ε": r["eps"],
                "Massa (g)": r["massa"] * 1000,
                "Custo (R$)": r["custo"],
                "q/m (W/kg)": r["q_m"],
                "q/C (W/R$)": r["q_c"],
                "Resiste a H₂SO₄": "sim" if r["corr"] else "não",
                "_crit": r[chave],
                "_corr": r["corr"],
            }
            for r in lista
        ]
    )
    aptos = tab[tab["_corr"]] if res["exige_corr"] else tab
    aptos = aptos.dropna(subset=["_crit"])
    if len(aptos) == 0:
        st.warning("Nenhum material selecionado atende às restrições impostas.")
    else:
        melhor = aptos.sort_values("_crit", ascending=False).iloc[0]
        st.success(
            f"**Material recomendado pelo critério “{crit}”: {melhor['Material']}**"
            + (" (entre os resistentes a H₂SO₄)" if res["exige_corr"] else "")
        )

    st.subheader("Ranking")
    ranking = tab.sort_values("_crit", ascending=False, na_position="last").drop(columns=["_crit", "_corr"])
    ranking.insert(0, "Posição", range(1, len(ranking) + 1))
    fmt = {
        "k (W/m·K)": "{:.1f}", "q (W)": "{:.3f}", f"q total, {p['Na']} aletas (W)": "{:.1f}",
        "T(L) (°C)": "{:.2f}", "η (%)": "{:.1f}", "ε": "{:.1f}", "Massa (g)": "{:.2f}",
        "Custo (R$)": "{:.3f}", "q/m (W/kg)": "{:.1f}", "q/C (W/R$)": "{:.1f}",
    }
    mostrar_tabela(ranking.style.format(fmt, na_rep="—"))

    bloco_equacoes()

    st.subheader("Perfis de temperatura")
    df_perf = pd.concat(
        [pd.DataFrame({"x (mm)": r["x"] * 1000, "T (°C)": r["T"], "Material": r["nome"]}) for r in lista]
    )
    ydom = dominio(df_perf["T (°C)"])
    mostrar_grafico(
        alt.Chart(df_perf).mark_line(strokeWidth=3).encode(
            x=alt.X("x (mm):Q", title="Posição x (mm)"),
            y=alt.Y("T (°C):Q", title="Temperatura T (°C)", scale=alt.Scale(domain=ydom)),
            color=alt.Color("Material:N", legend=alt.Legend(orient="right")),
            tooltip=["Material", alt.Tooltip("x (mm):Q", format=".2f"), alt.Tooltip("T (°C):Q", format=".2f")],
        ).properties(height=360)
    )

    st.subheader("Indicadores por material")
    barras = [
        ("q (W)", "Calor por aleta q (W)", "q (W)"),
        ("q/m (W/kg)", "Calor por massa q/m (W/kg)", "q/m (W/kg)"),
        ("q/C (W/R$)", "Calor por custo q/C (W/R$)", "q/C (W/R$)"),
        ("ε", "Efetividade ε", "ε (–)"),
    ]
    cols = st.columns(2)
    for i, (col, titulo, eixo) in enumerate(barras):
        dfb = tab[["Material", col, "_corr"]].dropna()
        dfb = dfb.assign(Apto=np.where(dfb["_corr"] | (not res["exige_corr"]), "atende", "não atende"))
        ch = alt.Chart(dfb).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Material:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{col}:Q", title=eixo),
            color=alt.Color(
                "Apto:N",
                scale=alt.Scale(domain=["atende", "não atende"], range=["#1f4e9a", "#c9d3df"]),
                legend=None,
            ),
            tooltip=["Material", alt.Tooltip(f"{col}:Q", format=".3f")],
        ).properties(title=titulo, height=280)
        with cols[i % 2]:
            mostrar_grafico(ch)
    if res["exige_corr"]:
        st.caption("Barras cinza: materiais que não atendem à exigência de resistência a H₂SO₄.")

    st.subheader("Calor dissipado × comprimento")
    df_c = pd.concat(
        [pd.DataFrame({"L (mm)": Ls * 1000, "q (W)": qs, "Material": nome}) for nome, Ls, qs in res["curvas"]]
    )
    df_c = df_c[df_c["L (mm)"] <= 3 * p["L_mm"]]
    linhas_L = alt.Chart(df_c).mark_line(strokeWidth=3).encode(
        x=alt.X("L (mm):Q", title="Comprimento da aleta L (mm)"),
        y=alt.Y("q (W):Q", title="Calor dissipado q (W)"),
        color=alt.Color("Material:N"),
        tooltip=["Material", alt.Tooltip("L (mm):Q", format=".1f"), alt.Tooltip("q (W):Q", format=".3f")],
    )
    regua = alt.Chart(pd.DataFrame({"L": [p["L_mm"]]})).mark_rule(strokeDash=[6, 4], color="#6b7a8a").encode(x="L:Q")
    mostrar_grafico((linhas_L + regua).properties(height=360))
    st.caption("Linha tracejada: comprimento atual. Materiais de baixa condutividade saturam com L pequeno.")

    biq = [r for r in lista if r["Bi"] > 0.1]
    if biq:
        st.warning("Bi > 0,1 para: " + ", ".join(r["nome"] for r in biq) + " — a hipótese 1D fica questionável.")

# ============================= TANQUE COM AS ALETAS =============================
st.divider()
st.subheader("Tanque de mistura com as aletas")
if modo == "Aleta única":
    r_fig = res["r"]
else:
    r_fig = lista[0]
    if len(aptos) > 0:
        r_fig = next(r for r in lista if r["nome"] == melhor["Material"])
M_t = p.get("M_t", 5000.0)
rho_t = p.get("rho_t", 1110.0)
HD_t = p.get("HD_t", 1.5)
svg_t, alt_t, inf_t = svg_tanque(r_fig, p["Tb"], p["Tinf"], p["D_mm"], p["L_mm"], p["Na"], M_t, rho_t, HD_t)
mostrar_svg(svg_t, alt_t)
st.caption(
    f"Tanque cilíndrico vertical com a massa de solução do simulador do tanque de mistura "
    f"(M = {M_t:.0f} kg, ρ = {rho_t:.0f} kg/m³ → V = {inf_t['V']:.2f} m³), "
    f"D_t = {inf_t['Dt']:.2f} m e H_t = {inf_t['Ht']:.2f} m. As {p['Na']} aletas ficam entre 10 % e 90 % "
    f"da altura, em {inf_t['nr']} fileiras de {inf_t['nc']} aletas"
    + (" (material recomendado na comparação)." if modo != "Aleta única" else ".")
)
if min(inf_t["Sv"], inf_t["Sc"]) < p["D_mm"] / 1000.0:
    st.error("As aletas não cabem na parede com esse diâmetro: o passo ficou menor que D.")
elif min(inf_t["Sv"], inf_t["Sc"]) < 2 * p["D_mm"] / 1000.0:
    st.warning("Aletas muito próximas (passo < 2D): a hipótese de h uniforme fica questionável.")

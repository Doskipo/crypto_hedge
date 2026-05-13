import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONS MATEMÀTIQUES (BLACK-SCHOLES)
# ==========================================
def bs_price(S, K, T, r, sigma):
    if T <= 0: return np.maximum(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    if T <= 0: return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1)

def bs_gamma(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.pdf(d1) / (S * sigma * np.sqrt(T))

# ==========================================
# 2. MOTOR DE SIMULACIÓ I COBERTURA
# ==========================================
def run_scenario(amb_crash=False):
    S0 = 65000.0; STRIKE_1 = 65000.0; STRIKE_2 = 68000.0
    DIES = 30; VOL = 0.50; R = 0.05
    
    TICKS_PER_DIA = 24 * 60  
    passos_totals = DIES * TICKS_PER_DIA
    dt = 1.0 / (365.0 * TICKS_PER_DIA)
    passos_entre_rebalancos = int(TICKS_PER_DIA / 1) # Rebalanç Diari
    
    # Utilitzem la mateixa llavor (seed) perquè els moviments normals siguin idèntics
    np.random.seed(42)
    Z = np.random.normal(0, 1, passos_totals)
    S_path = np.zeros(passos_totals + 1)
    S_path[0] = S0
    
    tick_del_crash = int(15 * TICKS_PER_DIA)
    
    # 1. Generació del preu
    for i in range(passos_totals):
        if amb_crash and i == tick_del_crash:
            S_path[i+1] = S_path[i] * 0.85 # Crash sever del -15%
        else:
            S_path[i+1] = S_path[i] * np.exp((R - 0.5 * VOL**2)*dt + VOL * np.sqrt(dt) * Z[i])
            
    temps_dies = np.linspace(0, DIES, passos_totals + 1)
    
    # 2. Configuració Inicial
    T_inicial = DIES / 365.0
    prima_inicial = bs_price(S0, STRIKE_1, T_inicial, R, VOL)
    
    error_delta = np.zeros(passos_totals + 1)
    efectiu_d = prima_inicial
    pos_btc_d = 0.0
    
    error_dg = np.zeros(passos_totals + 1)
    efectiu_dg = prima_inicial
    pos_btc_dg = 0.0
    pos_op2_dg = 0.0
    
    # 3. Bucle de Cobertura
    for i in range(passos_totals + 1):
        T_restant = max(0.00001, (DIES - temps_dies[i]) / 365.0)
        S_actual = S_path[i]
        
        # MTM
        deute_op1 = bs_price(S_actual, STRIKE_1, T_restant, R, VOL)
        error_delta[i] = (efectiu_d + pos_btc_d * S_actual) - deute_op1
        
        valor_op2 = pos_op2_dg * bs_price(S_actual, STRIKE_2, T_restant, R, VOL)
        error_dg[i] = (efectiu_dg + pos_btc_dg * S_actual + valor_op2) - deute_op1
        
        # Rebalanç Diari
        if i % passos_entre_rebalancos == 0 and i < passos_totals:
            # Només Delta
            d1 = bs_delta(S_actual, STRIKE_1, T_restant, R, VOL)
            efectiu_d -= (d1 - pos_btc_d) * S_actual
            pos_btc_d = d1
            
            # Delta-Gamma
            g1 = bs_gamma(S_actual, STRIKE_1, T_restant, R, VOL)
            g2 = bs_gamma(S_actual, STRIKE_2, T_restant, R, VOL)
            
            q_op2_nova = min(g1 / g2, 50.0) if g2 > 1e-6 else 0.0
            preu_op2 = bs_price(S_actual, STRIKE_2, T_restant, R, VOL)
            efectiu_dg -= (q_op2_nova - pos_op2_dg) * preu_op2
            pos_op2_dg = q_op2_nova
            
            d2 = bs_delta(S_actual, STRIKE_2, T_restant, R, VOL)
            delta_neta = (-1 * d1) + (pos_op2_dg * d2)
            q_btc_nova = -delta_neta
            
            efectiu_dg -= (q_btc_nova - pos_btc_dg) * S_actual
            pos_btc_dg = q_btc_nova
            
    return temps_dies, S_path, error_delta, error_dg

# ==========================================
# 3. EXECUTAR ELS DOS ESCENARIS
# ==========================================
print("Executant Escenari 1: Mercat Normal...")
t_norm, S_norm, err_d_norm, err_dg_norm = run_scenario(amb_crash=False)

print("Executant Escenari 2: Mercat amb Crash del 15%...")
t_crash, S_crash, err_d_crash, err_dg_crash = run_scenario(amb_crash=True)

# ==========================================
# 4. VISUALITZACIÓ (GRÀFICS SEPARATS)
# ==========================================

# --- FIGURA 1: ESCENARI SENSE CRASH ---
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(t_norm, S_norm, color='#1f77b4', linewidth=1.5)
ax1.set_title('Escenari 1: Trajectòria GBM Normal', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)

ax2.plot(t_norm[::50], err_d_norm[::50], color='red', linewidth=1.8, label='Només Delta')
ax2.plot(t_norm[::50], err_dg_norm[::50], color='green', linewidth=1.8, label='Delta-Gamma')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_title('Evolució de l\'Error PnL (Sense Crash)', fontweight='bold')
ax2.set_xlabel('Dies transcorreguts')
ax2.set_ylabel('Error PnL ($)')
ax2.grid(True, alpha=0.3)
ax2.legend()

fig1.tight_layout()

# --- FIGURA 2: ESCENARI AMB CRASH ---
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax3.plot(t_crash, S_crash, color='#1f77b4', linewidth=1.5)
ax3.axvline(x=15, color='red', linestyle='--', linewidth=1.5, label='Crash del 15%')
ax3.set_title('Escenari 2: Trajectòria GBM amb Crash Extrem', fontweight='bold')
ax3.set_ylabel('Preu BTC ($)')
ax3.grid(True, alpha=0.3)
ax3.legend()

ax4.plot(t_crash[::50], err_d_crash[::50], color='red', linewidth=1.8, label='Només Delta')
ax4.plot(t_crash[::50], err_dg_crash[::50], color='green', linewidth=1.8, label='Delta-Gamma')
ax4.axvline(x=15, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
ax4.axhline(y=0, color='black', linewidth=1)
ax4.set_title('Evolució de l\'Error PnL (Amb Crash al Dia 15)', fontweight='bold')
ax4.set_xlabel('Dies transcorreguts')
ax4.set_ylabel('Error PnL ($)')
ax4.grid(True, alpha=0.3)
ax4.legend()

fig2.tight_layout()

# Mostrem ambdues figures a la vegada
plt.show()
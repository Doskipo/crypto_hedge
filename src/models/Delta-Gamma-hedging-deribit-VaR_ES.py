import numpy as np
import pandas as pd
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
# 2. MOTOR DE RISC (FULL MC PER DELTA-GAMMA)
# ==========================================
def full_mc_var_es_dg(S, K1, K2, T, r, sigma, pos_btc_actual, pos_op2_actual, N_paths=10000, alpha=0.99, horizon_days=1):
    """
    Càlcul exacte del VaR i ES fent una revaluació completa de la cartera Delta-Gamma.
    Cartera: Curts d'1 Call ATM + Llargs de 'pos_op2' Calls OTM + Llargs de 'pos_btc' Bitcoins.
    """
    if T <= 0: return 0.0, 0.0
    
    dt_risk = horizon_days / 365.0
    
    # 1. VALOR INICIAL DE LA CARTERA 
    val_op1 = bs_price(S, K1, T, r, sigma)
    val_op2 = bs_price(S, K2, T, r, sigma)
    valor_inicial = -val_op1 + (pos_op2_actual * val_op2) + (pos_btc_actual * S)
    
    # 2. SIMULEM S_NEXT (10,000 escenaris)
    Z = np.random.standard_normal(N_paths)
    S_next = S * np.exp((r - 0.5 * sigma**2) * dt_risk + sigma * np.sqrt(dt_risk) * Z)
    
    # 3. VALOR FINAL DE LA CARTERA (Amb la posició congelada)
    val_op1_next = bs_price(S_next, K1, T - dt_risk, r, sigma)
    val_op2_next = bs_price(S_next, K2, T - dt_risk, r, sigma)
    valor_final_array = -val_op1_next + (pos_op2_actual * val_op2_next) + (pos_btc_actual * S_next)
    
    # 4. EQUACIÓ DE PÈRDUA
    losses = valor_inicial - valor_final_array
    
    # 5. EXTRAIEM MÈTRIQUES
    losses_sorted = np.sort(losses)
    pos_var = int(np.ceil(alpha * N_paths)) - 1
    
    var = losses_sorted[pos_var]
    es = np.mean(losses_sorted[pos_var:])
    
    return max(var, 0), max(es, 0)

# ==========================================
# 3. CARREGAR DADES I CONFIGURACIÓ
# ==========================================
print("Carregant dades de Deribit...")
try:
   df = pd.read_csv('/Marc/Universitat/MAMME/2n Quadrimestre/Quantitative Finances/Treball/dades_deribit.csv', parse_dates=['timestamp'])
except FileNotFoundError:
    print("ERROR: No s'ha trobat 'dades_deribit.csv'.")
    exit()

S0 = df.iloc[0]['price']
STRIKE_1 = S0              # Venem Call ATM
STRIKE_2 = S0 * 1.05       # Comprem Call 5% OTM per cobrir Gamma

T_TOTAL_DIES = 30.0
T_TOTAL_ANYS = T_TOTAL_DIES / 365.0
R = 0.0
FREQ_HORES = 24            # Rebalançem 1 cop al dia
FEE = 0.0

# ==========================================
# 4. CONFIGURACIÓ INICIAL (El Dia 0)
# ==========================================
iv_inicial = df.iloc[0]['iv']
prima_1 = bs_price(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)

# A. Cobertura Gamma Inicial
g1_inicial = bs_gamma(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)
g2_inicial = bs_gamma(S0, STRIKE_2, T_TOTAL_ANYS, R, iv_inicial)
pos_op2 = min(g1_inicial / g2_inicial, 50.0) if g2_inicial > 1e-6 else 0.0
prima_2 = bs_price(S0, STRIKE_2, T_TOTAL_ANYS, R, iv_inicial)

# B. Cobertura Delta Inicial
d1_inicial = bs_delta(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)
d2_inicial = bs_delta(S0, STRIKE_2, T_TOTAL_ANYS, R, iv_inicial)
pos_btc = d1_inicial - (pos_op2 * d2_inicial)

# C. Efectiu Inicial (Ingrés Op1 - Cost Op2 - Cost BTC)
efectiu = prima_1 - (pos_op2 * prima_2) - (pos_btc * S0)

llista_errors = []
llista_temps = []
llista_var = [] 
llista_es = []  

print("Inici Backtest Delta-Gamma + Risc (Full Monte Carlo)...")
np.random.seed(42) 

# ==========================================
# 5. BUCLE DE COBERTURA I RISC
# ==========================================
for i in range(1, len(df) - 1):
    S_actual = df.iloc[i]['price']
    iv_actual = df.iloc[i]['iv'] 

    t_passat = (df.iloc[i]['timestamp'] - df.iloc[0]['timestamp']).total_seconds() / (365 * 24 * 3600)
    T_restant = max(0.00001, T_TOTAL_ANYS - t_passat)
    
    # --- Mark-to-Market PnL ---
    deute_op1 = bs_price(S_actual, STRIKE_1, T_restant, R, iv_actual)
    valor_op2 = pos_op2 * bs_price(S_actual, STRIKE_2, T_restant, R, iv_actual)
    valor_cartera = efectiu + (pos_btc * S_actual) + valor_op2
    
    llista_errors.append(valor_cartera - deute_op1)
    llista_temps.append(df.iloc[i]['timestamp'])
    
    # --- CÀLCUL DEL RISC (Abans de rebalancejar) ---
    var_99, es_99 = full_mc_var_es_dg(
        S=S_actual, K1=STRIKE_1, K2=STRIKE_2, T=T_restant, r=R, sigma=iv_actual, 
        pos_btc_actual=pos_btc, pos_op2_actual=pos_op2, 
        alpha=0.99, N_paths=10000
    )
    llista_var.append(var_99)
    llista_es.append(es_99)
    
    # --- Rebalanç ---
    if i % FREQ_HORES == 0:
        
        # 1. Actualitzar Gamma
        gamma_1 = bs_gamma(S_actual, STRIKE_1, T_restant, R, iv_actual)
        gamma_2 = bs_gamma(S_actual, STRIKE_2, T_restant, R, iv_actual)
        
        q_op2_nova = min(gamma_1 / gamma_2, 50.0) if gamma_2 > 1e-6 else 0.0
        op_op2 = q_op2_nova - pos_op2
        preu_op2 = bs_price(S_actual, STRIKE_2, T_restant, R, iv_actual)
        
        efectiu -= (op_op2 * preu_op2)
        pos_op2 = q_op2_nova
        
        # 2. Actualitzar Delta
        delta_1 = bs_delta(S_actual, STRIKE_1, T_restant, R, iv_actual)
        delta_2 = bs_delta(S_actual, STRIKE_2, T_restant, R, iv_actual)
        delta_neta = (-1 * delta_1) + (pos_op2 * delta_2)
        
        q_btc_nova = -delta_neta
        op_btc = q_btc_nova - pos_btc
        
        efectiu -= (op_btc * S_actual)
        pos_btc = q_btc_nova

# ==========================================
# 6. VISUALITZACIÓ (3 PANELS)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# 1. Preu del Bitcoin
ax1.plot(df['timestamp'][:-1], df['price'][:-1], color='#1f77b4', linewidth=1.5)
ax1.set_title('Preu Real del Bitcoin (Deribit)', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)

# 2. Error PnL
ax2.plot(llista_temps, llista_errors, color='purple', linewidth=1.5, label='Error PnL (Delta-Gamma)')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_title('Evolució de l\'Error de Cobertura PnL', fontweight='bold')
ax2.set_ylabel('Error PnL ($)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Evolució de Mètriques de Risc (VaR & ES)
ax3.plot(llista_temps, llista_var, color='orange', linewidth=1.5, label='1-Day VaR (99%) - Delta-Gamma')
ax3.plot(llista_temps, llista_es, color='darkred', linestyle='--', linewidth=1.5, label='1-Day Expected Shortfall (99%) - Delta-Gamma')
ax3.set_title('Risc Diari amb Cobertura Delta-Gamma (Full Monte Carlo)', fontweight='bold')
ax3.set_xlabel('Data')
ax3.set_ylabel('Risc Potencial ($)')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()
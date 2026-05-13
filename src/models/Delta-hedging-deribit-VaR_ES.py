import numpy as np
import pandas as pd
import scipy.stats as si
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONS MATEMÀTIQUES (BLACK-SCHOLES)
# ==========================================
def bs_price(S, K, T, r, sigma):
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    if T <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1)

def bs_gamma(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_theta(S, K, T, r, sigma):
    if T <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * si.norm.cdf(d2)
    return term1 - term2

# ==========================================
# 2. MOTOR DE RISC (PARTIAL MC AMB DELTA DRIFT)
# ==========================================
def full_mc_var_es(S, K, T, r, sigma, pos_btc_actual, N_paths=10000, alpha=0.99, horizon_days=1):
    """
    Càlcul exacte del VaR i ES fent una revaluació completa (Full Monte Carlo) de la cartera.
    La cartera és: Curts d'1 Call + Llargs de 'pos_btc_actual' Bitcoins.
    """
    if T <= 0: return 0.0, 0.0
    
    dt_risk = horizon_days / 365.0
    
    # 1. VALOR INICIAL DE LA CARTERA (ignorant l'efectiu, ja que l'efectiu no té risc de preu)
    # Valor = -Preu_Call + (Posició_BTC * Preu_Actual)
    valor_inicial = -bs_price(S, K, T, r, sigma) + (pos_btc_actual * S)
    
    # 2. SIMULEM S_NEXT (10,000 escenaris)
    Z = np.random.standard_normal(N_paths)
    S_next = S * np.exp((r - 0.5 * sigma**2) * dt_risk + sigma * np.sqrt(dt_risk) * Z)
    
    # 3. VALOR FINAL DE LA CARTERA
    # ATENCIÓ: Utilitzem 'pos_btc_actual'. Per què? Perquè encara que rebalancegem exactament a S_next, 
    # el rebalanç és un intercanvi equivalent (Cash per BTC) que no altera el valor total 
    # de la cartera en aquell precís instant. Tota la pèrdua o guany del salt ve de pos_btc_actual.
    valor_final_array = -bs_price(S_next, K, T - dt_risk, r, sigma) + (pos_btc_actual * S_next)
    
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

# ==========================================
# 3. CARREGAR DADES I CONFIGURACIÓ
# ==========================================
S0 = df.iloc[0]['price']
STRIKE_1 = S0              
T_TOTAL_DIES = 30.0
T_TOTAL_ANYS = T_TOTAL_DIES / 365.0
R = 0.0
FREQ_HORES = 300
FEE = 0.0

# --- EL FIX: COBERTURA INICIAL AL DIA 0 ---
iv_inicial = df.iloc[0]['iv']
prima_1 = bs_price(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)
delta_inicial = bs_delta(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)

efectiu = prima_1 - (delta_inicial * S0) # Paguem el hedge inicial
pos_btc = delta_inicial                  # Comencem 100% coberts

llista_errors = []
llista_temps = []
llista_var = [] 
llista_es = []  

print("Inici Backtest Delta + Risc amb Delta Drift...")
np.random.seed(42) 

# ==========================================
# 4. BUCLE DE COBERTURA
# ==========================================
# Canviem el rang per començar des de 1, ja que el Dia 0 ja està processat
for i in range(1, len(df) - 1):
    S_actual = df.iloc[i]['price']
    iv_actual = df.iloc[i]['iv'] 

    t_passat = (df.iloc[i]['timestamp'] - df.iloc[0]['timestamp']).total_seconds() / (365 * 24 * 3600)
    T_restant = max(0.00001, T_TOTAL_ANYS - t_passat)
    
    # --- Mark-to-Market PnL ---
    deute_op1 = bs_price(S_actual, STRIKE_1, T_restant, R, iv_actual)
    valor_cartera = efectiu + (pos_btc * S_actual)
    llista_errors.append(valor_cartera - deute_op1)
    llista_temps.append(df.iloc[i]['timestamp'])
    
    # --- CÀLCUL DEL RISC (Abans de rebalancejar, quan la Delta està més desajustada) ---
    var_99, es_99 =  full_mc_var_es(
        S=S_actual, K=STRIKE_1, T=T_restant, r=R, sigma=iv_actual, 
        pos_btc_actual=pos_btc, alpha=0.99, N_paths=10000
    )
    llista_var.append(var_99)
    llista_es.append(es_99)
    
    # --- Rebalanç ---
    if i % FREQ_HORES == 0:
        q_btc_nova = bs_delta(S_actual, STRIKE_1, T_restant, R, iv_actual)
        op_btc = q_btc_nova - pos_btc
        
        efectiu -= (op_btc * S_actual)
        pos_btc = q_btc_nova

# ==========================================
# 5. VISUALITZACIÓ (3 PANELS)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# 1. Preu del Bitcoin
ax1.plot(df['timestamp'][:-1], df['price'][:-1], color='#1f77b4', linewidth=1.5)
ax1.set_title('Preu Real del Bitcoin (Deribit)', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)

# 2. Error PnL
ax2.plot(llista_temps, llista_errors, color='red', linewidth=1.5, label='Error PnL')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_title('Evolució de l\'Error de Cobertura PnL', fontweight='bold')
ax2.set_ylabel('Error PnL ($)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Evolució de Mètriques de Risc (VaR & ES)
ax3.plot(llista_temps, llista_var, color='orange', linewidth=1.5, label='1-Day VaR (99%) amb Delta Drift')
ax3.plot(llista_temps, llista_es, color='darkred', linestyle='--', linewidth=1.5, label='1-Day Expected Shortfall (99%) amb Delta Drift')
ax3.set_title('Risc Diari considerant la Delta Residual abans de Rebalancejar', fontweight='bold')
ax3.set_xlabel('Data')
ax3.set_ylabel('Risc Potencial ($)')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()
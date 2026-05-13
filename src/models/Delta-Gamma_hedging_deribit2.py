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

# ==========================================
# 2. CARREGAR DADES I PARÀMETRES
# ==========================================
print("Carregant dades de Deribit...")
try:
   df = pd.read_csv('/Marc/Universitat/MAMME/2n Quadrimestre/Quantitative Finances/Treball/dades_deribit.csv', parse_dates=['timestamp'])
except FileNotFoundError:
    print("ERROR: No s'ha trobat 'dades_deribit.csv'.")
    exit()

# Paràmetres del contracte
S0 = df.iloc[0]['price']
STRIKE_1 = S0-5000              # Venem aquesta opció Call (ATM)
STRIKE_2 = S0 * 0.85       # Comprem aquesta opció (5% OTM) per cobrir Gamma

T_TOTAL_DIES = 30.0
T_TOTAL_ANYS = T_TOTAL_DIES / 365.0
R = 0.0

FREQ_HORES = 24  # Freqüència de rebalanç (p. ex., cada 360 períodes/minuts)
FEE = 0           # 3 bps per a Deribit (ex: 0.0003)

# ==========================================
# 3. CONFIGURACIÓ INICIAL (El Dia 0)
# ==========================================
iv_inicial = df.iloc[0]['iv']
prima_1 = bs_price(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)

efectiu = prima_1  # Ingressos inicials
pos_btc = 0.0      
pos_op2 = 0.0      # Comencem sense opcions de cobertura
comissions_acum = 0.0

llista_errors = []
llista_temps = []
llista_target_btc = []  # NOU: Quant BTC necessitem teòricament
llista_cartera_btc = [] # NOU: Quant BTC tenim realment

print(f"Inici Backtest Delta-Gamma (Dades Reals):")
print(f"- Venem Call {STRIKE_1:.0f} | Ingrés: ${prima_1:.2f}")
print(f"- Comprem Call {STRIKE_2:.0f} per cobrir Gamma.")
print(f"- Rebalanç cada {FREQ_HORES} períodes.\n")

# ==========================================
# 4. BUCLE DE COBERTURA
# ==========================================
for i in range(len(df) - 1):
    S_actual = df.iloc[i]['price']
    iv_actual = df.iloc[i]['iv'] 
    
    t_passat = (df.iloc[i]['timestamp'] - df.iloc[0]['timestamp']).total_seconds() / (365 * 24 * 3600)
    T_restant = max(0.00001, T_TOTAL_ANYS - t_passat)
    
    # --- Guardem l'error actual (Mark-to-Market) ---
    deute_op1 = bs_price(S_actual, STRIKE_1, T_restant, R, iv_actual)
    valor_op2 = pos_op2 * bs_price(S_actual, STRIKE_2, T_restant, R, iv_actual)
    valor_cartera = efectiu + (pos_btc * S_actual) + valor_op2
    
    llista_errors.append(valor_cartera - deute_op1)
    llista_temps.append(df.iloc[i]['timestamp'])
    
    # --- Calculem la Delta Teòrica (Target BTC) ---
    # Som curts de l'Op1 (-Delta1) i llargs de l'Op2 (+ pos_op2 * Delta2)
    delta_1 = bs_delta(S_actual, STRIKE_1, T_restant, R, iv_actual)
    delta_2 = bs_delta(S_actual, STRIKE_2, T_restant, R, iv_actual)
    target_btc = delta_1 - (pos_op2 * delta_2)
    llista_target_btc.append(target_btc)
    
    # --- Rebalançem només quan toca ---
    if i % FREQ_HORES == 0:
        
        # 1. Neutralitzar Gamma
        gamma_1 = bs_gamma(S_actual, STRIKE_1, T_restant, R, iv_actual)
        gamma_2 = bs_gamma(S_actual, STRIKE_2, T_restant, R, iv_actual)
        
        # Límit de seguretat de 50 contractes per evitar explosions d'iliquiditat
        if gamma_2 > 1e-6:
            q_op2_nova = min(gamma_1 / gamma_2, 50.0)
        else:
            q_op2_nova = 0.0
            
        op_op2 = q_op2_nova - pos_op2
        preu_op2 = bs_price(S_actual, STRIKE_2, T_restant, R, iv_actual)
        fee_op2 = abs(op_op2) * S_actual * FEE
        
        efectiu -= (op_op2 * preu_op2 + fee_op2)
        comissions_acum += fee_op2
        pos_op2 = q_op2_nova
        
        # 2. Neutralitzar Delta (amb la nova posició d'opcions)
        delta_2_new = bs_delta(S_actual, STRIKE_2, T_restant, R, iv_actual)
        delta_neta = (-1 * delta_1) + (pos_op2 * delta_2_new)
        
        q_btc_nova = -delta_neta
        op_btc = q_btc_nova - pos_btc
        fee_btc = abs(op_btc) * S_actual * FEE
        
        efectiu -= (op_btc * S_actual + fee_btc)
        comissions_acum += fee_btc
        pos_btc = q_btc_nova

    # Guardem la posició real després d'avaluar el rebalanç
    llista_cartera_btc.append(pos_btc)

# ==========================================
# 5. RESULTATS FINALS I GRÀFICS
# ==========================================
S_final = df.iloc[-1]['price']
deute_op1 = max(S_final - STRIKE_1, 0)
valor_final_op2 = pos_op2 * max(S_final - STRIKE_2, 0)
print(deute_op1)
valor_cartera_final = efectiu + (pos_btc * S_final) + valor_final_op2
error_total = valor_cartera_final - deute_op1
error_mercat_pur = error_total + comissions_acum

print("--- RESULTATS AL VENCIMENT (Dades Reals) ---")
print(f"Pèrdua per Comissions: -${comissions_acum:.2f}")
print(f"Error de Mercat Pur:    ${error_mercat_pur:.2f}")
print(f"----------------------------------------")
print(f"ERROR TOTAL:            ${error_total:.2f}")

# ==========================================
# 6. VISUALITZACIÓ (3 PANELS)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Preu del Bitcoin
ax1.plot(df['timestamp'], df['price'], color='#1f77b4', linewidth=1.5)
ax1.set_title('Preu Real del Bitcoin (Deribit)', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)

# 2. Error de Cobertura
ax2.plot(llista_temps, llista_errors, color='purple', linewidth=1.5, label='Error PnL (Delta-Gamma)')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_title(f'Evolució de l\'Error PnL (Rebalanç cada {FREQ_HORES} períodes)')
ax2.set_ylabel('Error PnL ($)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Target BTC vs Posició Real
ax3.plot(llista_temps, llista_target_btc, color='gray', linestyle='--', linewidth=1.5, label='Target BTC (Contínua)')
ax3.plot(llista_temps, llista_cartera_btc, color='orange', drawstyle='steps-post', linewidth=1.5, label='Posició Real BTC (Esglaonada)')
ax3.set_title('Evolució de la Delta: Target vs. Posició Real (Delta-Gamma)', fontweight='bold')
ax3.set_xlabel('Data')
ax3.set_ylabel('Delta / BTC')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()
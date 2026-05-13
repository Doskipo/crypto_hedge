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

# ==========================================
# 2. CARREGAR DADES I PARÀMETRES
# ==========================================
print("Carregant dades de Deribit...")
try:
   df = pd.read_csv('/Marc/Universitat/MAMME/2n Quadrimestre/Quantitative Finances/Treball/dades_deribit.csv', parse_dates=['timestamp'])
except FileNotFoundError:
    print("ERROR: No s'ha trobat 'dades_deribit.csv'.")
    exit()

S0 = df.iloc[0]['price']
STRIKE_1 = S0              
# T_TOTAL_DIES = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / (24 * 3600)
T_TOTAL_DIES = 30
print(f"Total dies de dades: {T_TOTAL_DIES:.2f}")
T_TOTAL_ANYS = T_TOTAL_DIES / 365.0
R = 0.0
FEE = 0 # 3 bps per a Deribit (canvia-ho a 0 si vols ignorar comissions)

# ==========================================
# 3. FUNCIÓ DE COBERTURA PARAMETRITZADA
# ==========================================
def run_deribit_hedge(freq_steps):
    """
    Simula la cobertura Delta rebalancejant cada 'freq_steps' files del CSV.
    Retorna la llista d'errors PnL per a poder-la dibuixar.
    """
    iv_inicial = df.iloc[0]['iv']
    prima_1 = bs_price(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)

    efectiu = prima_1  
    pos_btc = 0.0      
    comissions_acum = 0.0

    llista_errors = []
    
    # Bucle principal sobre les dades històriques
    for i in range(len(df) - 1):
        S_actual = df.iloc[i]['price']
        iv_actual = df.iloc[i]['iv'] 
        
        t_passat = (df.iloc[i]['timestamp'] - df.iloc[0]['timestamp']).total_seconds() / (365 * 24 * 3600)
        T_restant = max(0.00001, T_TOTAL_ANYS - t_passat)
        
        # 1. Marquem a Mercat (Mark-to-Market) ABANS de rebalancejar
        deute_op1 = bs_price(S_actual, STRIKE_1, T_restant, R, iv_actual)
        valor_cartera = efectiu + (pos_btc * S_actual)
        
        llista_errors.append(valor_cartera - deute_op1)
        
        # 2. Rebalançem només quan toca segons la freqüència
        if i % freq_steps == 0:
            delta_1 = bs_delta(S_actual, STRIKE_1, T_restant, R, iv_actual)
            q_btc_nova = delta_1
            
            op_btc = q_btc_nova - pos_btc
            fee_btc = abs(op_btc) * S_actual * FEE
            
            efectiu -= (op_btc * S_actual + fee_btc)
            comissions_acum += fee_btc
            pos_btc = q_btc_nova
            
    # Assegurem que l'últim punt coincideixi en longitud
    llista_errors.append(llista_errors[-1]) 
    return llista_errors, comissions_acum

# ==========================================
# 4. EXECUTAR PER A DIFERENTS FREQÜÈNCIES
# ==========================================
# Suposant que el teu CSV té 1 fila per hora:
# 1 = Cada hora, 4 = Cada 4 hores, 12 = Cada 12 hores, 24 = Un cop al dia
frequencies = [1, 12, 24, 120, 240]  # Rebalanç cada hora, cada 12 hores, cada dia, cada 5 dies, cada 10 dies
labels = ['Rebalanç cada hora', 'Rebalanç cada 12 hores', 'Rebalanç cada dia', 'Rebalanç cada 5 dies']
colors = ['black', 'blue', 'orange', 'red']

resultats = {}

print("Executant simulacions...")
for freq, label in zip(frequencies, labels):
    errors_pnl, fees = run_deribit_hedge(freq)
    resultats[label] = errors_pnl
    print(f"Completat: {label} | Error PnL Final: ${errors_pnl[-1]:.2f} | Comissions totals: ${fees:.2f}")

# ==========================================
# 5. VISUALITZACIÓ GRÀFICA
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Preu Real del Bitcoin
ax1.plot(df['timestamp'], df['price'], color='#1f77b4', linewidth=1.5)
ax1.set_title('Preu Real del Bitcoin (Deribit)', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)

# Errors de Cobertura Superposats
temps_plot = df['timestamp'].tolist()
for label, color in zip(labels, colors):
    ax2.plot(temps_plot, resultats[label], label=label, color=color, linewidth=1.5, alpha=0.8)

ax2.axhline(y=0, color='black', linewidth=1, linestyle='--')
ax2.set_title('Evolució de l\'Error PnL per Freqüència de Rebalanç', fontweight='bold')
ax2.set_xlabel('Data')
ax2.set_ylabel('Error PnL ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
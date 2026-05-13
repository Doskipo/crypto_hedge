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

# Paràmetres del contracte
S0 = df.iloc[0]['price']
STRIKE_1 = S0              # Venem aquesta opció Call (In-The-Money)
T_TOTAL_DIES = 30.0
T_TOTAL_ANYS = T_TOTAL_DIES / 365.0

print(f"Dies totals: {T_TOTAL_DIES}")
print(f"Anys totals: {T_TOTAL_ANYS}")
R = 0.0

FREQ_HORES = 1  # Freqüència de rebalanç
FEE = 0        # 3 bps per a Deribit (ex: 0.0003)

# ==========================================
# 3. CONFIGURACIÓ INICIAL (El Dia 0)
# ==========================================
iv_inicial = df.iloc[0]['iv']
prima_1 = bs_price(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)

efectiu = prima_1  # Ingressos inicials per la venda de la Call
pos_btc = 0.0      # Comencem amb 0 Bitcoins
comissions_acum = 0.0

llista_errors = []
llista_temps = []
llista_delta_teorica = [] # NOU: Per guardar la Delta exacta a cada tick
llista_delta_cartera = [] # NOU: Per guardar la quantitat real de BTC que tenim
deute = 0.0
print(f"Inici Backtest Delta Simple (Dades Reals):")
print(f"- Venem Call {STRIKE_1:.0f} | Ingrés: ${prima_1:.2f}")
print(f"- Rebalanç cada {FREQ_HORES} períodes.\n")

# ==========================================
# 4. BUCLE DE COBERTURA
# ==========================================
for i in range(len(df) - 1):
    S_actual = df.iloc[i]['price']
    iv_actual = df.iloc[i]['iv'] 
    
    t_passat = (df.iloc[i]['timestamp'] - df.iloc[0]['timestamp']).total_seconds() / (365 * 24 * 3600)
    T_restant = max(0.00001, T_TOTAL_ANYS - t_passat)
    
    # --- Guardem l'error actual abans de rebalancejar ---
    deute_op1 = bs_price(S_actual, STRIKE_1, T_restant, R, iv_actual)
    valor_cartera = efectiu + (pos_btc * S_actual)
    
    llista_errors.append(valor_cartera - deute_op1)
    llista_temps.append(df.iloc[i]['timestamp'])
    
    # NOU: Calculem la Delta teòrica per a l'anàlisi, encara que no rebalancem
    delta_teorica = bs_delta(S_actual, STRIKE_1, T_restant, R, iv_actual)
    llista_delta_teorica.append(delta_teorica)
    
    # --- Rebalançem només quan toca ---
    if i % FREQ_HORES == 0:
        
        # Com estem CURTS de l'opció (-1), la nostra delta neta inicial és -delta_teorica.
        # Per neutralitzar-la, necessitem comprar +delta_teorica Bitcoins.
        q_btc_nova = delta_teorica
        op_btc = q_btc_nova - pos_btc
        fee_btc = abs(op_btc) * S_actual * FEE
        
        # Paguem els Bitcoins i les comissions
        efectiu -= (op_btc * S_actual + fee_btc)
        comissions_acum += fee_btc
        pos_btc = q_btc_nova

    # NOU: Guardem la posició real que mantenim durant aquest tick
    llista_delta_cartera.append(pos_btc)

# ==========================================
# 5. RESULTATS FINALS I GRÀFICS
# ==========================================
S_final = df.iloc[-1]['price']
deute_op1 = max(S_final - STRIKE_1, 0) # Payoff real al venciment
print(deute)
valor_cartera_final = efectiu + (pos_btc * S_final)
error_total = valor_cartera_final - deute_op1
error_mercat_pur = error_total + comissions_acum

print("--- RESULTATS AL VENCIMENT (Dades Reals) ---")
print(f"Pèrdua per Comissions: -${comissions_acum:.2f}")
print(f"Error de Mercat Pur:    ${error_mercat_pur:.2f}")
print(f"----------------------------------------")
print(f"ERROR TOTAL:            ${error_total:.2f}")

# ==========================================
# 6. VISUALITZACIÓ (Ara amb 3 gràfics)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1. Preu del Bitcoin
ax1.plot(df['timestamp'], df['price'], color='#1f77b4', linewidth=1.5)
ax1.set_title('Preu Real del Bitcoin (Deribit)', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)

# 2. Error de Cobertura
ax2.plot(llista_temps, llista_errors, color='red', linewidth=1.5, label='Error PnL')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_title(f'Evolució de l\'Error PnL (Rebalanç cada {FREQ_HORES} períodes)')
ax2.set_ylabel('Error PnL ($)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Evolució de la Delta vs Posició de la Cartera
ax3.plot(llista_temps, llista_delta_teorica, color='gray', linestyle='--', linewidth=1.5, label='Delta Teòrica (Contínua)')
# L'estil 'steps-post' fa que la línia es mantingui plana fins al proper rebalanç
ax3.plot(llista_temps, llista_delta_cartera, color='green', drawstyle='steps-post', linewidth=1.5, label='Posició Real BTC (Esglaonada)')
ax3.set_title('Evolució de la Delta vs. Posició Real a la Cartera', fontweight='bold')
ax3.set_xlabel('Data')
ax3.set_ylabel('Delta / BTC')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.show()
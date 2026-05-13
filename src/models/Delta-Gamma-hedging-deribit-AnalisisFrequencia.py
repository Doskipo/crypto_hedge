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
    print("ERROR: No s'ha trobat 'dades_deribit.csv'. Assegura't de posar la ruta correcta.")
    # Creem dades falses només perquè el codi no falli si el proves sense l'arxiu
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
    df = pd.DataFrame({'timestamp': dates, 'price': np.linspace(65000, 68000, 1000) + np.random.randn(1000)*500, 'iv': 0.5})

S0 = df.iloc[0]['price']
STRIKE_1 = S0              # Opció Venuda (ATM)
STRIKE_2 = S0 * 1.05       # Opció Comprada (5% OTM) per cobrir Gamma

T_TOTAL_DIES = 30.0
T_TOTAL_ANYS = T_TOTAL_DIES / 365.0
R = 0.0
FEE = 0.0000  # Posa 0.0003 si vols veure l'impacte devastador de les comissions a alta freqüència

# ==========================================
# 3. FUNCIÓ DE COBERTURA PARAMETRITZADA
# ==========================================
def run_dg_deribit(freq_steps):
    iv_inicial = df.iloc[0]['iv']
    prima_1 = bs_price(S0, STRIKE_1, T_TOTAL_ANYS, R, iv_inicial)

    efectiu = prima_1  
    pos_btc = 0.0      
    pos_op2 = 0.0      
    comissions_acum = 0.0

    llista_errors = []
    llista_target_btc = []
    llista_cartera_btc = []
    
    # Bucle principal sobre les dades històriques
    for i in range(len(df) - 1):
        S_actual = df.iloc[i]['price']
        iv_actual = df.iloc[i]['iv'] 
        
        t_passat = (df.iloc[i]['timestamp'] - df.iloc[0]['timestamp']).total_seconds() / (365 * 24 * 3600)
        T_restant = max(0.00001, T_TOTAL_ANYS - t_passat)
        
        # --- Mark-to-Market ABANS de rebalancejar ---
        deute_op1 = bs_price(S_actual, STRIKE_1, T_restant, R, iv_actual)
        valor_op2 = pos_op2 * bs_price(S_actual, STRIKE_2, T_restant, R, iv_actual)
        valor_cartera = efectiu + (pos_btc * S_actual) + valor_op2
        
        llista_errors.append(valor_cartera - deute_op1)
        
        # --- Càlcul de la Delta Teòrica (Target) ---
        delta_1 = bs_delta(S_actual, STRIKE_1, T_restant, R, iv_actual)
        delta_2 = bs_delta(S_actual, STRIKE_2, T_restant, R, iv_actual)
        target_btc = delta_1 - (pos_op2 * delta_2)
        llista_target_btc.append(target_btc)
        
        # --- Rebalançem només quan toca ---
        if i % freq_steps == 0:
            
            # 1. Gamma Hedging
            gamma_1 = bs_gamma(S_actual, STRIKE_1, T_restant, R, iv_actual)
            gamma_2 = bs_gamma(S_actual, STRIKE_2, T_restant, R, iv_actual)
            
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
            
            # 2. Delta Hedging
            delta_2_new = bs_delta(S_actual, STRIKE_2, T_restant, R, iv_actual)
            delta_neta = (-1 * delta_1) + (pos_op2 * delta_2_new)
            
            q_btc_nova = -delta_neta
            op_btc = q_btc_nova - pos_btc
            fee_btc = abs(op_btc) * S_actual * FEE
            
            efectiu -= (op_btc * S_actual + fee_btc)
            comissions_acum += fee_btc
            pos_btc = q_btc_nova

        llista_cartera_btc.append(pos_btc)
            
    return llista_errors, llista_target_btc, llista_cartera_btc, comissions_acum

# ==========================================
# 4. EXECUTAR PER A DIFERENTS FREQÜÈNCIES
# ==========================================
# Assumint que el teu CSV té resolució d'1 hora per fila:
frequencies = [1, 4, 12, 24]
labels = ['Cada hora (1)', 'Cada 4 hores (4)', 'Cada 12 hores (12)', 'Diari (24)']
colors = ['black', 'blue', 'orange', 'red']

resultats_errors = {}
resultats_carteres = {}
target_referencia = None

print("Executant simulacions comparatives Delta-Gamma...")
for freq, label in zip(frequencies, labels):
    err, tgt, cart, fees = run_dg_deribit(freq)
    resultats_errors[label] = err
    resultats_carteres[label] = cart
    if target_referencia is None:
        target_referencia = tgt # El target teòric és independent de la freqüència de rebalanç
        
    print(f"Completat: {label:<20} | Error Final: ${err[-1]:>8.2f} | Comissions: ${fees:.2f}")

# Extraiem l'eix de temps correcte (traient la darrera fila per quadrar amb el bucle)
temps_plot = df['timestamp'].iloc[:-1]

# ==========================================
# 5. VISUALITZACIÓ (3 PANELS)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# Panel 1: Preu del Bitcoin Real
ax1.plot(df['timestamp'], df['price'], color='#1f77b4', linewidth=1.5)
ax1.set_title('Preu Real del Bitcoin (Deribit)', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)

# Panel 2: Errors de Cobertura per Freqüència
for label, color in zip(labels, colors):
    ax2.plot(temps_plot, resultats_errors[label], label=label, color=color, linewidth=1.5, alpha=0.8)

ax2.axhline(y=0, color='black', linewidth=1, linestyle='--')
ax2.set_title('Evolució de l\'Error PnL (Delta-Gamma) per Freqüència', fontweight='bold')
ax2.set_ylabel('Error PnL ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Tracking de la Delta
ax3.plot(temps_plot, target_referencia, color='gray', linestyle='--', linewidth=2, label='Target BTC (Delta Teòrica)')

# Mostrem l'esglaonat de les freqüències més baixes per veure l'efecte de fricció clarament
ax3.plot(temps_plot, resultats_carteres['Diari (24)'], color='red', drawstyle='steps-post', linewidth=1.5, alpha=0.8, label='Posició Real BTC (Diari)')
ax3.plot(temps_plot, resultats_carteres['Cada 4 hores (4)'], color='blue', drawstyle='steps-post', linewidth=1.5, alpha=0.8, label='Posició Real BTC (Cada 4 hores)')

ax3.set_title('Tracking de la Delta: Target vs. Posicions Reals', fontweight='bold')
ax3.set_xlabel('Data')
ax3.set_ylabel('Delta / BTC')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONS MATEMÀTIQUES
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
# 2. GENERACIÓ DEL MERCAT (AMB CRASH)
# ==========================================
S0 = 65000.0; STRIKE_1 = 65000.0; STRIKE_2 = 68000.0
DIES = 30; VOL = 0.50; R = 0.05; FEE = 0

TICKS_PER_DIA = 24 * 60  # Resolució: 1 minut
passos_totals = DIES * TICKS_PER_DIA
dt = 1.0 / (365.0 * TICKS_PER_DIA)

np.random.seed(42)
Z = np.random.normal(0, 1, passos_totals)
S_path = np.zeros(passos_totals + 1)
S_path[0] = S0

tick_del_crash = int(15 * TICKS_PER_DIA) # Crash al dia 15

for i in range(passos_totals):
    if i == tick_del_crash:
        S_path[i+1] = S_path[i] * 0.95 # Crash sobtat del -15%
    else:
        S_path[i+1] = S_path[i] * np.exp((R - 0.5 * VOL**2)*dt + VOL * np.sqrt(dt) * Z[i])

temps_dies = np.linspace(0, DIES, passos_totals + 1)

# ==========================================
# 3. FUNCIÓ DE COBERTURA DELTA-GAMMA
# ==========================================
def run_dg_hedge(rebalancos_per_dia):
    passos_entre_rebalancos = int(TICKS_PER_DIA / rebalancos_per_dia)
    
    error_cobertura = np.zeros(passos_totals + 1)
    target_btc = np.zeros(passos_totals + 1)
    cartera_btc = np.zeros(passos_totals + 1)
    
    # Configuració Dia 0
    T_inicial = DIES / 365.0
    efectiu = bs_price(S0, STRIKE_1, T_inicial, R, VOL)
    pos_btc = 0.0
    pos_op2 = 0.0
    
    for i in range(passos_totals + 1):
        T_restant = max(0.00001, (DIES - temps_dies[i]) / 365.0)
        S_actual = S_path[i]
        
        # A. Mark-to-Market
        deute_op1 = bs_price(S_actual, STRIKE_1, T_restant, R, VOL)
        valor_op2 = pos_op2 * bs_price(S_actual, STRIKE_2, T_restant, R, VOL)
        valor_cartera = efectiu + (pos_btc * S_actual) + valor_op2
        error_cobertura[i] = valor_cartera - deute_op1
        
        # B. Calculem la Delta Target (Quant BTC necessitem teòricament ARA MATEIX)
        d1 = bs_delta(S_actual, STRIKE_1, T_restant, R, VOL)
        d2 = bs_delta(S_actual, STRIKE_2, T_restant, R, VOL)
        target_btc[i] = d1 - (pos_op2 * d2)
        
        # C. Rebalanç
        if i % passos_entre_rebalancos == 0 and i < passos_totals:
            # 1. Gamma Hedging
            g1 = bs_gamma(S_actual, STRIKE_1, T_restant, R, VOL)
            g2 = bs_gamma(S_actual, STRIKE_2, T_restant, R, VOL)
            
            # Límit de seguretat (Max 50 contractes) per quan G2 s'acosta a zero
            if g2 > 1e-6:
                q_op2_nova = min(g1 / g2, 50.0) 
            else:
                q_op2_nova = 0.0
                
            op_op2 = q_op2_nova - pos_op2
            preu_op2 = bs_price(S_actual, STRIKE_2, T_restant, R, VOL)
            efectiu -= (op_op2 * preu_op2 + abs(op_op2)*S_actual*FEE)
            pos_op2 = q_op2_nova
            
            # 2. Delta Hedging
            d2_new = bs_delta(S_actual, STRIKE_2, T_restant, R, VOL)
            delta_neta = (-1 * d1) + (pos_op2 * d2_new)
            
            q_btc_nova = -delta_neta
            op_btc = q_btc_nova - pos_btc
            efectiu -= (op_btc * S_actual + abs(op_btc)*S_actual*FEE)
            pos_btc = q_btc_nova
            
        cartera_btc[i] = pos_btc
            
    return error_cobertura, target_btc, cartera_btc

# ==========================================
# 4. ANÀLISI DE FREQÜÈNCIES
# ==========================================
freqs_to_test = [1, 2, 6, 24, 24*60]
labels = ['Diari (1/dia)', 'Cada 12h (2/dia)', 'Cada 4h (6/dia)', 'Horari (24/dia)', 'Continu (1/minut)']
colors = ['red', 'orange', 'green', 'blue', 'black']

errors_dict, targets_dict, carteres_dict = {}, {}, {}

print("Executant simulacions Delta-Gamma...")
for freq, label in zip(freqs_to_test, labels):
    err, tgt, cart = run_dg_hedge(freq)
    errors_dict[label] = err
    targets_dict[label] = tgt
    carteres_dict[label] = cart
    print(f"Completat: {label} \t| Error Final: ${err[-1]:.2f}")

# ==========================================
# 5. VISUALITZACIÓ (3 PANELS)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# Panel 1: Preu
ax1.plot(temps_dies, S_path, color='#1f77b4', linewidth=1.5)
ax1.axvline(x=15, color='red', linestyle='--', linewidth=1.5, label='Crash del 15%')
ax1.set_title('Trajectòria Simulada del Bitcoin (Amb Crash al Dia 15)', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel 2: Error
for label, color in zip(labels, colors):
    ax2.plot(temps_dies[::50], errors_dict[label][::50], label=f'{label}', color=color, linewidth=1.5, alpha=0.8)
ax2.axvline(x=15, color='red', linestyle='--', linewidth=1.5)
ax2.axhline(y=0, color='black', linewidth=1, linestyle='--')
ax2.set_title('Evolució de l\'Error de Cobertura (Delta-Gamma)', fontweight='bold')
ax2.set_ylabel('Error PnL ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Target BTC vs Posició Real
ax3.plot(temps_dies[::50], targets_dict['Continu (1/minut)'][::50], color='gray', linestyle='--', linewidth=2, label='Target BTC (Delta Teòrica)')
ax3.plot(temps_dies, carteres_dict['Diari (1/dia)'], color='red', drawstyle='steps-post', linewidth=1.5, alpha=0.8, label='Posició Real BTC (Diari)')
ax3.plot(temps_dies, carteres_dict['Horari (24/dia)'], color='blue', drawstyle='steps-post', linewidth=1.5, alpha=0.8, label='Posició Real BTC (Horari)')

ax3.axvline(x=15, color='red', linestyle='--', linewidth=1.5)
ax3.set_title('Evolució de la Cobertura: BTC Target vs Posició Real', fontweight='bold')
ax3.set_xlabel('Dies transcorreguts')
ax3.set_ylabel('Quantitat de BTC')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
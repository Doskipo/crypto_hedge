import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONS DE BLACK-SCHOLES
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

# ==========================================
# 2. GENERACIÓ DEL MERCAT MESTRE (1 MINUT RESOLUTION)
# ==========================================
S0 = 65000.0; K = 65000.0; DIES = 30; VOL = 0.50; R = 0.0
TICKS_PER_DIA = 24 * 60  # Resolució base: 1 minut
passos_totals = DIES * TICKS_PER_DIA
dt = 1.0 / (365.0 * TICKS_PER_DIA)

np.random.seed(42)
Z = np.random.normal(0, 1, passos_totals)
S_path = np.zeros(passos_totals + 1)
S_path[0] = S0

for i in range(passos_totals):
    S_path[i+1] = S_path[i] * np.exp((R - 0.5 * VOL**2)*dt + VOL * np.sqrt(dt) * Z[i])

temps_dies = np.linspace(0, DIES, passos_totals + 1)

# ==========================================
# 3. FUNCIÓ DE COBERTURA
# ==========================================
def run_delta_hedge(rebalancos_per_dia):
    passos_entre_rebalancos = int(TICKS_PER_DIA / rebalancos_per_dia)
    
    valor_teoric_opcio = np.zeros(passos_totals + 1)
    valor_cartera = np.zeros(passos_totals + 1)
    error = np.zeros(passos_totals + 1)
    
    # NOU: Arrays per guardar l'evolució de la Delta
    delta_teorica = np.zeros(passos_totals + 1)
    delta_cartera = np.zeros(passos_totals + 1)
    
    # Setup Dia 0
    T_inicial = DIES / 365.0
    valor_teoric_opcio[0] = bs_price(S0, K, T_inicial, R, VOL)
    pos_btc = bs_delta(S0, K, T_inicial, R, VOL)
    efectiu = valor_teoric_opcio[0] - (pos_btc * S0)
    
    delta_teorica[0] = pos_btc
    delta_cartera[0] = pos_btc
    
    for i in range(1, passos_totals + 1):
        T_restant = max(0.00001, (DIES - temps_dies[i]) / 365.0) 
        
        # MTM
        valor_teoric_opcio[i] = bs_price(S_path[i], K, T_restant, R, VOL)
        valor_cartera[i] = efectiu + (pos_btc * S_path[i])
        error[i] = valor_cartera[i] - valor_teoric_opcio[i]
        
        # NOU: Calculem la Delta teòrica exacta
        d_teorica = bs_delta(S_path[i], K, T_restant, R, VOL)
        delta_teorica[i] = d_teorica
        
        # Rebalanç
        if i % passos_entre_rebalancos == 0 and i < passos_totals:
            quantitat_a_comprar = d_teorica - pos_btc
            efectiu -= quantitat_a_comprar * S_path[i]
            pos_btc = d_teorica
            
        # NOU: Guardem la posició real després d'un possible rebalanç
        delta_cartera[i] = pos_btc
            
    return error, delta_teorica, delta_cartera

# ==========================================
# 4. ANÀLISI DE FREQÜÈNCIES
# ==========================================
freqs_to_test = [1, 2, 6, 24, 24*60] # 1x Dia, 2x Dia, 6x Dia, 1x Hora, 1x Minut
labels = ['Diari (1/dia)', 'Cada 12h (2/dia)', 'Cada 4h (6/dia)', 'Horari (24/dia)', 'Continu (1/minut)']
colors = ['red', 'orange', 'green', 'blue', 'black']

errors_dict = {}
delta_cartera_dict = {}

print("Executant simulacions...")
for freq, label in zip(freqs_to_test, labels):
    err, d_teo, d_cart = run_delta_hedge(freq)
    errors_dict[label] = err
    delta_cartera_dict[label] = d_cart
    print(f"Completat: {label} \t| Error Final: ${err[-1]:.2f}")

# La Delta teòrica és independent de la freqüència de rebalanç, així que la podem agafar de qualsevol execució
delta_teorica_referencia = d_teo 

# ==========================================
# 5. VISUALITZACIÓ (3 PANELS)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# --- Subplot 1: Preu del Subjacent ---
ax1.plot(temps_dies, S_path, color='#1f77b4', linewidth=1.5)
ax1.set_title('Trajectòria Simulada del Bitcoin (Mètode GBM)', fontweight='bold')
ax1.set_ylabel('Preu BTC ($)')
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Evolució de l'Error PnL ---
for label, color in zip(labels, colors):
    # Dibuixem només un de cada 50 punts per optimitzar el gràfic
    ax2.plot(temps_dies[::50], errors_dict[label][::50], label=f'{label}', color=color, linewidth=1.5, alpha=0.8)

ax2.axhline(y=0, color='black', linewidth=1, linestyle='--')
ax2.set_title('Evolució de l\'Error de Cobertura per Freqüència de Rebalanç', fontweight='bold')
ax2.set_ylabel('Error PnL ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- Subplot 3: Delta Teòrica vs Posició Real ---
ax3.plot(temps_dies[::50], delta_teorica_referencia[::50], color='gray', linestyle='--', linewidth=2, label='Delta Teòrica (Contínua)')

# Dibuixem només un parell de freqüències representatives per no saturar el gràfic
ax3.plot(temps_dies, delta_cartera_dict['Diari (1/dia)'], color='red', drawstyle='steps-post', linewidth=1.5, alpha=0.8, label='Posició Real (Rebalanç Diari)')
ax3.plot(temps_dies, delta_cartera_dict['Horari (24/dia)'], color='blue', drawstyle='steps-post', linewidth=1.5, alpha=0.8, label='Posició Real (Rebalanç Horari)')

ax3.set_title('Evolució de la Delta: Comparativa de Seguiment (Tracking)', fontweight='bold')
ax3.set_xlabel('Dies transcorreguts')
ax3.set_ylabel('Delta / BTC')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
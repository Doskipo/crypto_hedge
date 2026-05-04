import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONS DE BLACK-SCHOLES
# ==========================================
def bs_price(S, K, T, r, sigma):
    """Calcula el preu just (Valor Teòric) de l'opció Call"""
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    """Calcula la Delta de l'opció Call"""
    if T <= 0: return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1)

# ==========================================
# 2. VARIABLES DEL TEU EXPERIMENT
# ==========================================
# PROVA DE CANVIAR AQUESTS DOS NÚMEROS!
TICKS_MERCAT_PER_DIA = 24*60    # El mercat es mou 24 cops al dia (cada hora)
REBALANCOS_PER_DIA = 24     # Nosaltres només ajustem la Delta 1 cop al dia

# Paràmetres del contracte
S0 = 65000.0
K = 65000.0
DIES = 30
VOL = 0.50
R = 0.0

# Matemàtiques del temps
passos_totals = DIES * TICKS_MERCAT_PER_DIA
dt = 1.0 / (365.0 * TICKS_MERCAT_PER_DIA)
passos_entre_rebalancos = int(TICKS_MERCAT_PER_DIA / REBALANCOS_PER_DIA)

# ==========================================
# 3. GENERACIÓ DEL MERCAT (EL CAMÍ GBM)
# ==========================================
np.random.seed(42) # Perquè el gràfic sempre sigui igual i puguem comparar
Z = np.random.normal(0, 1, passos_totals)
S = np.zeros(passos_totals + 1)
S[0] = S0

# Generem tot el camí del Bitcoin pas a pas
for i in range(passos_totals):
    S[i+1] = S[i] * np.exp((R - 0.5 * VOL**2)*dt + VOL * np.sqrt(dt) * Z[i])

# ==========================================
# 4. LA SIMULACIÓ DE COBERTURA
# ==========================================
temps_dies = np.linspace(0, DIES, passos_totals + 1)
valor_teoric_opcio = np.zeros(passos_totals + 1)
valor_la_meva_cartera = np.zeros(passos_totals + 1)
error_de_discretitzacio = np.zeros(passos_totals + 1)

# El Dia 0 (Configuració inicial)
T_inicial = DIES / 365.0
valor_teoric_opcio[0] = bs_price(S0, K, T_inicial, R, VOL)
posicio_btc = bs_delta(S0, K, T_inicial, R, VOL)

# Creem una cartera "Autofinançada" (El que ingressen de la prima menys el que ens costa el primer BTC)
efectiu = valor_teoric_opcio[0] - (posicio_btc * S0)
valor_la_meva_cartera[0] = efectiu + (posicio_btc * S0)

# El bucle del temps
for i in range(1, passos_totals + 1):
    # Protecció matemàtica pels últims minuts abans de caducar
    T_restant = max(0.00001, (DIES - temps_dies[i]) / 365.0) 
    
    # A. Actualitzem on està el mercat (Valor Teòric)
    valor_teoric_opcio[i] = bs_price(S[i], K, T_restant, R, VOL)
    
    # B. Actualitzem la nostra cartera (que pot tenir la Delta desactualitzada)
    valor_la_meva_cartera[i] = efectiu + (posicio_btc * S[i])
    
    # C. Calculem l'error en aquest instant
    error_de_discretitzacio[i] = valor_la_meva_cartera[i] - valor_teoric_opcio[i]
    
    # D. Toca rebalancejar? Només comprem/venem si coincideix amb el nostre rellotge
    if i % passos_entre_rebalancos == 0 and i < passos_totals:
        nova_delta = bs_delta(S[i], K, T_restant, R, VOL)
        quantitat_a_comprar = nova_delta - posicio_btc
        
        # Paguem els Bitcoins nous o cobrem si en venem (Sense comissions)
        efectiu -= quantitat_a_comprar * S[i]
        posicio_btc = nova_delta

# ==========================================
# 5. VISUALITZACIÓ (MATPLOTLIB)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Gràfic Superior: Comparativa de Valors
ax1.plot(temps_dies, valor_teoric_opcio, label='Valor Teòric de l\'Opció', color='black', linewidth=2)
ax1.plot(temps_dies, valor_la_meva_cartera, label='Valor de la nostra Cartera', color='red', linestyle='--', linewidth=1.5)
ax1.set_title(f'Cobertura Discreta vs Contínua\n(Mercat: {TICKS_MERCAT_PER_DIA} ticks/dia | Rebalanç: {REBALANCOS_PER_DIA} cops/dia)', fontweight='bold')
ax1.set_ylabel('Valor en $')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gràfic Inferior: L'Error pur
ax2.plot(temps_dies, error_de_discretitzacio, label='Error de Discretització (El que guanyes/perds per anar tard)', color='purple')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_title('Evolució de l\'Error de Cobertura')
ax2.set_xlabel('Dies transcorreguts')
ax2.set_ylabel('Error en $')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Error residual final al venciment: ${error_de_discretitzacio[-1]:.2f}")

import requests
import pandas as pd
from datetime import datetime, timedelta

def obtenir_dades_deribit():
    print("Connectant amb l'API de Deribit...")
    
    # Descarreguem els últims 30 dies
    data_final = datetime.now()
    data_inici = data_final - timedelta(days=30)
    print(datetime.now())
    ts_inici = int(data_inici.timestamp() * 1000)
    ts_final = int(data_final.timestamp() * 1000)
    
    # 1. Preu del BTC (Resolució: 1 hora = 60 minuts)
    url_preu = f"https://deribit.com/api/v2/public/get_tradingview_chart_data?instrument_name=BTC-PERPETUAL&start_timestamp={ts_inici}&end_timestamp={ts_final}&resolution=1"
    resposta_preu = requests.get(url_preu).json()
    
    df_preu = pd.DataFrame({
        'timestamp_ms': resposta_preu['result']['ticks'],
        'price': resposta_preu['result']['close']
    })

    # 2. Volatilitat Implícita - DVOL (Resolució: 1 hora = 3600 segons)
    url_dvol = f"https://deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp={ts_inici}&end_timestamp={ts_final}&resolution=60"
    resposta_dvol = requests.get(url_dvol).json()
    
    dades_dvol = resposta_dvol['result']['data']
    df_dvol = pd.DataFrame(dades_dvol, columns=['timestamp_ms', 'open', 'high', 'low', 'close_iv'])
    
    # Passem la IV de percentatge a decimal (ex: 55 -> 0.55) per a Black-Scholes
    df_dvol['iv'] = df_dvol['close_iv'] / 100.0

    # 3. Fusionar i guardar
    df_final = pd.merge(df_preu, df_dvol[['timestamp_ms', 'iv']], on='timestamp_ms', how='inner')
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp_ms'], unit='ms')
    df_final = df_final[['timestamp', 'price', 'iv']]
    
    df_final.to_csv('C:/Marc/Universitat/MAMME/2n Quadrimestre/Quantitative Finances/Treball/dades_deribit.csv', index=False)
    print(f"Èxit! S'ha creat l'arxiu 'dades_deribit.csv' amb {len(df_final)} files de dades reals.")

obtenir_dades_deribit()
"""
BTC Price Action Tracker
Calistir: streamlit run btc_dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="BTC Tracker", layout="wide", page_icon="B")

st.markdown("""
<style>
.aciklama {
    background:#1a1d2e; border-left:3px solid #00b894;
    padding:12px 16px; border-radius:0 8px 8px 0;
    margin-bottom:14px; font-size:.9rem; color:#dfe6e9; line-height:1.6;
}
.uyari {
    background:#2d1f1f; border-left:3px solid #e17055;
    padding:12px 16px; border-radius:0 8px 8px 0;
    margin-bottom:10px; font-size:.9rem; color:#fab1a0;
}
.strateji {
    background:#1a2e1f; border-left:3px solid #fdcb6e;
    padding:12px 16px; border-radius:0 8px 8px 0;
    margin-bottom:10px; font-size:.9rem; color:#ffeaa7; line-height:1.6;
}
.baslik { font-size:1.5rem; font-weight:700; color:#00b894; margin-bottom:4px; }
.kart { background:#1a1d2e; padding:14px; border-radius:8px; text-align:center; margin-bottom:8px; }
.kart_deger { font-size:1.35rem; font-weight:700; color:#fff; }
.kart_etiket { font-size:.73rem; color:#636e72; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

HALVING = [pd.Timestamp('2016-07-09'), pd.Timestamp('2020-05-11'), pd.Timestamp('2024-04-20')]
SONRAKI_HALVING = pd.Timestamp('2028-03-01')


# ── VERİ ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Binance 4H veri cekiliyor...")
def veri_cek():
    url, start, all_data = 'https://api.binance.com/api/v3/klines', 1502942400000, []
    while True:
        data = requests.get(url, params={'symbol':'BTCUSDT','interval':'4h',
                                         'startTime':start,'limit':1000}).json()
        if not data or isinstance(data, dict): break
        all_data.extend(data)
        start = data[-1][0] + 1
        if len(data) < 1000: break
        time.sleep(0.1)
    df = pd.DataFrame(all_data, columns=['tarih','acilis','yuksek','dusuk','kapanis','hacim',
                                          'kz','qh','is','tab','taq','ign'])
    df['tarih'] = pd.to_datetime(df['tarih'], unit='ms')
    df.set_index('tarih', inplace=True)
    for k in ['acilis','yuksek','dusuk','kapanis','hacim']:
        df[k] = df[k].astype(float)
    return df[['acilis','yuksek','dusuk','kapanis','hacim']]


@st.cache_data(show_spinner="Pivotlar hesaplaniyor...")
def hesapla(_df, wf_n, sp_order):
    def wf(df, n):
        h, l, nb = df['yuksek'].values, df['dusuk'].values, len(df)
        ft, fd = np.full(nb, False), np.full(nb, False)
        for i in range(n, nb-n):
            if h[i] > max(h[i-n:i]) and h[i] > max(h[i+1:i+n+1]): ft[i] = True
            if l[i] < min(l[i-n:i]) and l[i] < min(l[i+1:i+n+1]): fd[i] = True
        return np.where(ft)[0], np.where(fd)[0]

    def sp(df, order):
        return (argrelextrema(df['yuksek'].values, np.greater_equal, order=order)[0],
                argrelextrema(df['dusuk'].values,  np.less_equal,    order=order)[0])

    def olustur(df, ti, di):
        t = pd.DataFrame({'tarih':df.index[ti], 'fiyat':df['yuksek'].iloc[ti].values, 'tip':'TEPE'})
        d = pd.DataFrame({'tarih':df.index[di], 'fiyat':df['dusuk'].iloc[di].values,  'tip':'DIP'})
        return pd.concat([t,d]).sort_values('tarih').reset_index(drop=True)

    def temizle(pdf):
        rows, out, i = pdf.to_dict('records'), [], 0
        while i < len(rows):
            tip, grup = rows[i]['tip'], []
            while i < len(rows) and rows[i]['tip'] == tip:
                grup.append(rows[i]); i += 1
            out.append(max(grup, key=lambda x:x['fiyat']) if tip=='TEPE'
                       else min(grup, key=lambda x:x['fiyat']))
        return pd.DataFrame(out).reset_index(drop=True)

    def hesapla_pivot(pdf):
        df = pdf.copy()
        df['sonraki_tarih']     = df['tarih'].shift(-1)
        df['sonraki_fiyat']     = df['fiyat'].shift(-1)
        df['sonraki_tip']       = df['tip'].shift(-1)
        df['sure_gun']          = (df['sonraki_tarih']-df['tarih']).dt.total_seconds()/86400
        df['fiyat_degisim_pct'] = (df['sonraki_fiyat']-df['fiyat'])/df['fiyat']*100
        df['hiz_pct_gun']       = df['fiyat_degisim_pct']/df['sure_gun']
        df['faz'] = np.where((df['tip']=='DIP')&(df['sonraki_tip']=='TEPE'),'BOGA',
                    np.where((df['tip']=='TEPE')&(df['sonraki_tip']=='DIP'),'AYI','KARMA'))
        df['hareket']      = df['fiyat'].diff()
        df['prev_hareket'] = df['hareket'].shift(1)
        df['oran_pct']     = (df['hareket']/df['prev_hareket'].abs())*100
        df['ms_etiket']    = '—'
        ti = df[df['tip']=='TEPE'].index.tolist()
        di = df[df['tip']=='DIP'].index.tolist()
        for p,idx in enumerate(ti):
            if p==0: continue
            df.loc[idx,'ms_etiket'] = 'HH' if df.loc[idx,'fiyat']>df.loc[ti[p-1],'fiyat'] else 'LH'
        for p,idx in enumerate(di):
            if p==0: continue
            df.loc[idx,'ms_etiket'] = 'HL' if df.loc[idx,'fiyat']>df.loc[di[p-1],'fiyat'] else 'LL'
        df['halving_dongu'] = np.where(df['tarih']<HALVING[1],'Dongu 1 (2017-2020)',
                              np.where(df['tarih']<HALVING[2],'Dongu 2 (2020-2024)','Dongu 3 (2024-)'))
        return df.dropna(subset=['sure_gun'])

    wft, wfd = wf(_df, wf_n)
    spt, spd = sp(_df, sp_order)
    return (hesapla_pivot(temizle(olustur(_df,wft,wfd))),
            hesapla_pivot(temizle(olustur(_df,spt,spd))),
            wft, wfd, spt, spd)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("## BTC Price Action Tracker")
st.sidebar.markdown("---")

sayfa = st.sidebar.radio("Sayfa:", [
    "0 — Ana Ekran",
    "1 — Istatistiksel Analiz",
    "2 — Korelasyon",
    "3 — Market Structure",
    "4 — ZigZag Oran Analizi",
    "5 — Moving Average Analizi",
])

st.sidebar.markdown("---")
st.sidebar.markdown("#### Pivot Parametreleri")
st.sidebar.caption("N buyudukce az ama daha guclu pivot uretir.")
yontem = st.sidebar.selectbox("Yontem:", ["Williams Fractal","Sabit Pencere"])
if yontem == "Williams Fractal":
    wf_n, sp_order = st.sidebar.select_slider("WF N:", [2,5,10,20,50], value=10), 50
    yks = f"WF (N={wf_n})"
else:
    wf_n, sp_order = 10, st.sidebar.select_slider("SP order:", [20,50,100,150,200], value=50)
    yks = f"SP (order={sp_order})"

st.sidebar.markdown("---")
st.sidebar.markdown("#### Zaman Araligi")
aralik = st.sidebar.selectbox("Donem:", [
    "Tum Gecmis (2017-)","Son 1 Yil","Son 2 Yil",
    "Dongu 1 (2017-2020)","Dongu 2 (2020-2024)","Dongu 3 (2024-)"])


# ── VERİ YÜKLEMESİ ────────────────────────────────────────────────────────────

df_tam = veri_cek()
wf_piv_tam, sp_piv_tam, wf_t, wf_d, sp_t, sp_d = hesapla(df_tam, wf_n, sp_order)
piv_tam  = wf_piv_tam if yontem=="Williams Fractal" else sp_piv_tam

bugun = df_tam.index[-1]
if aralik=="Son 1 Yil":       bas = bugun-pd.Timedelta(days=365)
elif aralik=="Son 2 Yil":     bas = bugun-pd.Timedelta(days=730)
elif aralik=="Dongu 1 (2017-2020)": bas, bugun = pd.Timestamp('2017-01-01'), HALVING[1]
elif aralik=="Dongu 2 (2020-2024)": bas, bugun = HALVING[1], HALVING[2]
elif aralik=="Dongu 3 (2024-)":     bas = HALVING[2]
else:                          bas = df_tam.index[0]

df      = df_tam[(df_tam.index>=bas)&(df_tam.index<=bugun)]
piv_df  = piv_tam[(piv_tam['tarih']>=bas)&(piv_tam['tarih']<=bugun)].reset_index(drop=True)


# ── YARDIMCILAR ───────────────────────────────────────────────────────────────

def kart(col, deger, etiket, renk="#ffffff"):
    col.markdown(f'<div class="kart"><div class="kart_deger" style="color:{renk}">{deger}</div>'
                 f'<div class="kart_etiket">{etiket}</div></div>', unsafe_allow_html=True)

def acik(m): st.markdown(f'<div class="aciklama">{m}</div>', unsafe_allow_html=True)
def uyar(m): st.markdown(f'<div class="uyari">⚠️ {m}</div>', unsafe_allow_html=True)
def strat(m): st.markdown(f'<div class="strateji">💡 {m}</div>', unsafe_allow_html=True)

# ── MOVING AVERAGE YARDIMCILARI ──────────────────────────────────────────────

MA_SECENEKLERI = [9, 12, 15, 20, 21, 50, 100, 144, 200, 288, 300, 365]

def ohlc_resample(df, periyot):
    if periyot == "Gunluk":
        rule = "1D"
    else:
        rule = "1W"

    out = df.resample(rule).agg({
        'acilis': 'first',
        'yuksek': 'max',
        'dusuk': 'min',
        'kapanis': 'last',
        'hacim': 'sum'
    }).dropna()

    return out

def ma_hesapla(seri, ma_tipi, uzunluk):
    if ma_tipi == "SMA":
        return seri.rolling(window=uzunluk).mean()
    elif ma_tipi == "EMA":
        return seri.ewm(span=uzunluk, adjust=False).mean()
    return pd.Series(index=seri.index, dtype=float)


# ── SAYFA 0 — ANA EKRAN ───────────────────────────────────────────────────────

if sayfa == "0 — Ana Ekran":
    st.markdown('<div class="baslik">BTC Price Action Tracker — Mevcut Durum</div>', unsafe_allow_html=True)
    acik("Bu sayfa <b>simdi ne durumdasin?</b> sorusunu yanıtlar. Son pivot analiz edilir, "
         "tarihsel ortalamalarla kiyaslanir, hedef fiyat ve kalan sure tahmin edilir. "
         "Uyari sistemi anormal durumlari isaret eder, strateji kutulari icgoru uretir.")

    son_fiyat  = df_tam['kapanis'].iloc[-1]
    son_pivot  = piv_tam.iloc[-2] if len(piv_tam)>=2 else piv_tam.iloc[-1]
    bugun_ts   = pd.Timestamp(datetime.now())
    pivot_gun  = (bugun_ts - son_pivot['tarih']).days
    mevcut_faz = "BOGA" if son_pivot['tip']=='DIP' else "AYI"

    boga_df  = piv_tam[piv_tam['faz']=='BOGA']
    ayi_df   = piv_tam[piv_tam['faz']=='AYI']
    ort_bs, ort_as   = boga_df['sure_gun'].mean(), ayi_df['sure_gun'].mean()
    ort_bh, ort_ah   = boga_df['fiyat_degisim_pct'].mean(), ayi_df['fiyat_degisim_pct'].mean()
    med_bh, med_ah   = boga_df['fiyat_degisim_pct'].median(), ayi_df['fiyat_degisim_pct'].median()
    std_bs, std_as   = boga_df['sure_gun'].std(), ayi_df['sure_gun'].std()

    kalan      = (ort_bs if mevcut_faz=='BOGA' else ort_as) - pivot_gun
    hedef_ort  = son_pivot['fiyat'] * (1+(ort_bh if mevcut_faz=='BOGA' else ort_ah)/100)
    hedef_med  = son_pivot['fiyat'] * (1+(med_bh if mevcut_faz=='BOGA' else med_ah)/100)
    hedef_ust  = son_pivot['fiyat'] * (1+(ort_bh+(std_bs/ort_bs*ort_bh) if mevcut_faz=='BOGA'
                                         else ort_ah-(std_as/ort_as*abs(ort_ah)))/100)
    halving_gun  = (bugun_ts-HALVING[2]).days
    sonraki_gun  = (SONRAKI_HALVING-bugun_ts).days

    # Son MS etiket
    son_et = piv_tam[piv_tam['ms_etiket'].isin(['HH','HL','LH','LL'])]['ms_etiket'].iloc[-1] \
             if len(piv_tam[piv_tam['ms_etiket'].isin(['HH','HL','LH','LL'])]) > 0 else '—'

    # Gecis matrisi (Markov) — son etiketten sonra ne bekleniyor
    renk_map = {'HH':'#00b894','HL':'#74b9ff','LH':'#e17055','LL':'#d63031','—':'#b2bec3'}
    etiketler = ['HH','HL','LH','LL']
    seq_tam   = piv_tam[piv_tam['ms_etiket'].isin(etiketler)]['ms_etiket'].tolist()
    matrix_c  = pd.DataFrame(0, index=etiketler, columns=etiketler)
    for i in range(len(seq_tam)-1):
        matrix_c.loc[seq_tam[i], seq_tam[i+1]] += 1
    matrix_p = matrix_c.div(matrix_c.sum(axis=1).replace(0,1), axis=0)*100

    # ── Metrik kartları ─────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kart(c1, f"${son_fiyat:,.0f}", "Son Fiyat")
    fr = "#00b894" if mevcut_faz=="BOGA" else "#e17055"
    kart(c2, "BOGA" if mevcut_faz=="BOGA" else "AYI", "Mevcut Faz", fr)
    kart(c3, f"{pivot_gun} gun", "Son Pivot Oncesi")
    kart(c4, f"{max(0,kalan):.0f} gun" if kalan>0 else f"{abs(kalan):.0f} gun asıldı", "Ort. Kalan Sure", "#fdcb6e")
    kart(c5, f"${hedef_ort:,.0f}", "Hedef (Ort.)", "#74b9ff")
    kart(c6, f"${hedef_med:,.0f}", "Hedef (Medyan)", "#a29bfe")

    st.markdown("---")

    # ── Durum Açıklaması ────────────────────────────────────────────────────
    acik(
        f"<b>Mevcut Durum:</b> Son pivot <b>{son_pivot['tip']}</b> "
        f"({son_pivot['tarih'].strftime('%d %b %Y')}, ${son_pivot['fiyat']:,.0f}). "
        f"BTC su an <b>{'BOGA' if mevcut_faz=='BOGA' else 'AYI'} fazinda</b>, "
        f"<b>{pivot_gun} gundur</b> devam ediyor.<br><br>"
        f"Tarihsel ort. {'boga' if mevcut_faz=='BOGA' else 'ayi'} suresi: "
        f"<b>{ort_bs if mevcut_faz=='BOGA' else ort_as:.0f} gun</b> "
        f"(std: {std_bs if mevcut_faz=='BOGA' else std_as:.0f} gun). "
        f"{'Henuz ortalamaya ulasilmadi.' if kalan>0 else '<b>Ortalama sure asildi.</b>'}<br><br>"
        f"Tarihsel ort. hareket {ort_bh if mevcut_faz=='BOGA' else ort_ah:.1f}% uygulanirsa "
        f"hedef <b>${hedef_ort:,.0f}</b>, medyan bazli hedef <b>${hedef_med:,.0f}</b>. "
        f"Son MS etiketi: <b style='color:{renk_map.get(son_et, chr(35)+chr(98)+chr(50)+chr(98)+chr(101)+chr(99)+chr(51))}'>{son_et}</b>."
    )

    # ── Strateji Kutuları ────────────────────────────────────────────────────
    st.markdown("#### Strateji Sinyalleri")

    # Boga fazi long stratejisi
    if mevcut_faz == "BOGA":
        boga_asilma_orani = pivot_gun / ort_bs if ort_bs > 0 else 0
        if boga_asilma_orani < 0.5:
            strat(
                f"<b>LONG — Erken Boga Fazi:</b> Mevcut boga fazi tarihsel ortalamanin "
                f"%{boga_asilma_orani*100:.0f}'inde. Trend hala erken evresinde. "
                f"Hedefe uzaklik: ${hedef_ort-son_fiyat:,.0f} (%{(hedef_ort/son_fiyat-1)*100:.1f}). "
                f"Stop-loss icin son dip referansi: ${son_pivot['fiyat']:,.0f}."
            )
        elif boga_asilma_orani < 1.0:
            strat(
                f"<b>LONG — Olgun Boga Fazi:</b> Tarihsel ortalamanin %{boga_asilma_orani*100:.0f}'inde. "
                f"Kar alma bolgesi yaklasıyor. Medyan hedef ${hedef_med:,.0f}, ort. hedef ${hedef_ort:,.0f}. "
                f"Pozisyon buyutme yerine mevcut pozisyonu koruma zamani."
            )
        else:
            strat(
                f"<b>DİKKAT — Uzun Boga Fazi:</b> Tarihsel ortalama sure asildi ({pivot_gun} > {ort_bs:.0f} gun). "
                f"Her yeni gunden sonra donus riski artar. Yeni long girmek yerine mevcut poziyon "
                f"icin trailing stop ve kar alma hedefleri belirlenmeli."
            )
    else:  # AYI
        ayi_asilma_orani = pivot_gun / ort_as if ort_as > 0 else 0
        if ayi_asilma_orani < 0.5:
            strat(
                f"<b>SHORT — Erken Ayi Fazi:</b> Ayi fazi tarihsel ortalamanin "
                f"%{ayi_asilma_orani*100:.0f}'inde. Short pozisyon icin hala vakit var. "
                f"Hedef dip: ${hedef_ort:,.0f} (ort.), ${hedef_med:,.0f} (medyan). "
                f"Stop-loss icin son tepe referansi: ${son_pivot['fiyat']:,.0f}."
            )
        elif ayi_asilma_orani < 1.0:
            strat(
                f"<b>SHORT — Olgun Ayi Fazi:</b> Tarihsel ortalamanin %{ayi_asilma_orani*100:.0f}'inde. "
                f"Dip bolgesine yaklasiliyor. Short kapat veya azalt, long giris noktasi taramaya basla. "
                f"Medyan dip hedefi: ${hedef_med:,.0f}."
            )
        else:
            strat(
                f"<b>DIP TARAMASI — Uzun Ayi Fazi:</b> Ortalama ayi suresi asildi ({pivot_gun} > {ort_as:.0f} gun). "
                f"Dip olustum riski yukseliyor. Kademeli long giris icin seviyeleri izle. "
                f"Short pozisyonlarda stop-loss sikistir."
            )

    # Markov bazlı sonraki sinyal
    if son_et in etiketler:
        en_cok_sonraki = matrix_p.loc[son_et].idxmax()
        en_cok_pct     = matrix_p.loc[son_et].max()
        strat(
            f"<b>Markov Sinyali — Sonraki Etiket Tahmini:</b> Son etiket "
            f"<b style='color:{renk_map[son_et]}'>{son_et}</b>. "
            f"Tarihsel olarak bu etiketten sonra en sik "
            f"<b style='color:{renk_map[en_cok_sonraki]}'>{en_cok_sonraki}</b> geldi "
            f"(%{en_cok_pct:.0f} olasilikla, {int(matrix_c.loc[son_et,en_cok_sonraki])} kez). "
            f"{'Bu boga yapisinin devamına isaret eder.' if en_cok_sonraki in ['HH','HL'] else 'Bu zayiflama veya ayi yapisina isaret eder.'}"
        )

    # ── Uyarı Sistemi ────────────────────────────────────────────────────────
    st.markdown("#### Uyari Sistemi")
    uyarilar = []
    if kalan < 0:
        uyarilar.append(f"Mevcut faz tarihsel ortalama sureyi <b>{abs(kalan):.0f} gun</b> asti.")
    if pivot_gun > (ort_bs if mevcut_faz=='BOGA' else ort_as) + (std_bs if mevcut_faz=='BOGA' else std_as):
        uyarilar.append(f"Sure ort + 1 std'yi gecti ({pivot_gun} gun). Istatistiksel olarak asiri uzun faz.")
    if halving_gun < 180:
        uyarilar.append(f"Son halvingden <b>{halving_gun} gun</b> gecti. Halving etkisi tam oturmamis olabilir.")
    if sonraki_gun < 365:
        uyarilar.append(f"Sonraki halvinge <b>{sonraki_gun} gun</b> kaldi. Onceki halvinglerde bu donemde volatilite artti.")
    if not uyarilar:
        st.success("Su an aktif uyari yok.")
    else:
        for u in uyarilar: uyar(u)

    # ── ZigZag Oran — Son Hareket ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ZigZag Oran — Son Hareket Nerede?")
    acik(
        "Son pivot hareketinin bir oncekine orani. "
        "+100% uzeri HH bolgesi (guclu yukselis), -100% altı LL bolgesi (derin dusus). "
        "Yuzdelik: tarihin kac %'inden buyuk bir hareket oldugunu gosterir."
    )
    piv_z = piv_tam.copy()
    piv_z['hareket']      = piv_z['fiyat'].diff()
    piv_z['prev_hareket'] = piv_z['hareket'].shift(1)
    piv_z['oran_pct']     = (piv_z['hareket']/piv_z['prev_hareket'].abs())*100
    temiz_z = piv_z.dropna(subset=['oran_pct'])
    if len(temiz_z) >= 2:
        son_oran = temiz_z['oran_pct'].iloc[-1]
        perc     = (temiz_z['oran_pct'] < son_oran).mean()*100
        kat      = ("HH Bolgesi" if son_oran>100 else "HL Bolgesi" if son_oran>0
                    else "LH Bolgesi" if son_oran>-100 else "LL Bolgesi")
        r        = "#00b894" if son_oran>100 else "#e17055" if son_oran<-100 else "#fdcb6e"
        c1,c2,c3 = st.columns(3)
        kart(c1, f"%{son_oran:.1f}", "Son ZigZag Orani", r)
        kart(c2, f"%{perc:.0f} yuzdelik", "Tarihsel Konumu")
        kart(c3, kat, "Siniflandirma", r)

        strat(
            f"<b>ZigZag Strateji:</b> Son hareket %{son_oran:.1f} ile tarihin %{perc:.0f} yuzdeligiyle. "
            f"{'Guclu bir hareket — momentum devam edebilir ancak asiri uzama riski var.' if perc>80 else 'Ortalama bir hareket — trendin devamı veya donusu her ikisi de olasi.' if perc>40 else 'Zayif bir hareket — karsi trend girisi icin izle.'}"
        )

    # ── Son 90 Gün ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Tum Gecmis — Pivot Grafigi")
    son90  = df_tam
    p90    = piv_tam
    fig_90 = go.Figure()
    fig_90.add_trace(go.Candlestick(
        x=son90.index, open=son90['acilis'], high=son90['yuksek'],
        low=son90['dusuk'], close=son90['kapanis'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350', showlegend=False))
    tp90 = p90[p90['tip']=='TEPE']
    dp90 = p90[p90['tip']=='DIP']
    fig_90.add_trace(go.Scatter(x=tp90['tarih'], y=tp90['fiyat'], mode='markers+text',
                                text=[f"${v:,.0f}" for v in tp90['fiyat']],
                                textposition='top center', textfont=dict(size=9, color='#e17055'),
                                marker=dict(symbol='triangle-down', color='#e17055', size=12), name='Tepe'))
    fig_90.add_trace(go.Scatter(x=dp90['tarih'], y=dp90['fiyat'], mode='markers+text',
                                text=[f"${v:,.0f}" for v in dp90['fiyat']],
                                textposition='bottom center', textfont=dict(size=9, color='#00b894'),
                                marker=dict(symbol='triangle-up', color='#00b894', size=12), name='Dip'))
    fig_90.update_layout(template='plotly_dark', height=420, xaxis_rangeslider_visible=False,
                         legend=dict(orientation='h', y=1.02))
    st.plotly_chart(fig_90, use_container_width=True)

    # ── Halving Döngü Tablosu ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Halving Donguleri — Boga Fazi Ozeti")
    acik(
        "<b>Halving:</b> Yaklasik 4 yilda bir BTC madenci odulu yariya iner, arz kisitlanir. "
        "Tarihsel olarak her halvingden 12-18 ay icinde yeni zirve goruldu. "
        "Asagidaki tablo her dongu icindeki boga fazlarinin istatistiksel ozetini verir. "
        "Donguler arasi ortalama sure ve hareket kuculuyor mu? Piyasa olgunlasiyor mu?"
    )
    halving_ozet = piv_tam[piv_tam['faz']=='BOGA'].groupby('halving_dongu').agg(
        Sayi=('sure_gun','count'),
        Ort_Sure_gun=('sure_gun','mean'),
        Med_Sure_gun=('sure_gun','median'),
        Ort_Hareket_pct=('fiyat_degisim_pct','mean'),
        Med_Hareket_pct=('fiyat_degisim_pct','median'),
        Ort_Hiz=('hiz_pct_gun','mean')
    ).round(1)
    st.dataframe(halving_ozet, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    kart(c1, f"{halving_gun} gun", "Son Halvingden (20 Nis 2024)", "#fdcb6e")
    kart(c2, f"{sonraki_gun} gun", "Sonraki Halvinge (~Mar 2028)", "#74b9ff")
    kart(c3, f"%{min(100,halving_gun/1460*100):.0f}", "Dongu Ilerlemesi (4yil=100%)", "#a29bfe")
    kart(c4, HALVING[2].strftime('%d %b %Y'), "Son Halving", "#b2bec3")

    # ── Proje Açıklaması ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Bu Tracker Nasil Calisir?")
    acik(
        "Bu tracker BTC'nin dongusel hareketini sayisal hale getirir. "
        "Price action metodolojisi kullanarak dip-tepe zamanlaması anlasilir, "
        "vadeli islemler ve opsiyon pozisyonlari icin tarihe dayali referans noktalari uretilir.<br><br>"
        "<b>Veri:</b> Binance Public API — 4H BTCUSDT, 2017'den buyana, API key gerektirmez.<br>"
        "<b>Pivot:</b> Williams Fractal (N) veya Sabit Pencere (order) — sidebar'dan ayarlanir.<br>"
        "<b>ZigZag:</b> Ardisik ayni tip pivotlar temizlenir, her zaman DIP-TEPE-DIP zinciri olusur.<br>"
        "<b>MS:</b> HH/HL/LH/LL etiketleri ile trend yapisi belirlenir.<br>"
        "<b>Markov:</b> Etiketten etikete gecis olasılıkları hesaplanir.<br><br>"
        "<b>Dikkat:</b> Tum sayilar tarihsel ortalamalardir. Piyasa gecmisin tekrarini garanti etmez. "
        "Pozisyon kararları tamamen kullaniciya aittir."
    )
    adimlar = [
        ("Istatistiksel Analiz","Boga/Ayi Fazi Sure ve Hareket Istatistikleri",
         "Her boga/ayi icin sure, yuzde hareket, gunluk hiz. Ortalama + medyan + std."),
        ("Korelasyon","Pearson Katsayisi ile Iliski Olcumu",
         "Uzun boga daha buyuk hareket mi? Uzun boga uzun ayi getirir mi?"),
        ("Market Structure","HH/HL/LH/LL + Markov + Pattern Analizi",
         "Gecis matrisi, run analizi, 4'lu pattern, N streak sonrasi ne geldi?"),
        ("ZigZag Oran","Her Hareketin Bir Oncekine Orani",
         "+100% = HH esigi, -100% = LL esigi. Dagilim ve son hareketin yuzdeligi."),
    ]
    for baslik, alt, ack in adimlar:
        with st.expander(f"- {baslik} — {alt}"):
            acik(ack)


# ── SAYFA 1 — İSTATİSTİK ──────────────────────────────────────────────────────

elif sayfa == "1 — Istatistiksel Analiz":
    st.markdown('<div class="baslik">Istatistiksel Analiz</div>', unsafe_allow_html=True)
    acik(
        "<b>Boga fazi:</b> Dipten tepeye — fiyat yukseliyor.<br>"
        "<b>Ayi fazi:</b> Tepeden dibe — fiyat dusuyor.<br><br>"
        "Her faz icin <b>sure</b> (kac gun), <b>fiyat degisim</b> (%), "
        "<b>hiz</b> (%/gun) hesaplanir. "
        "Medyan asiri uclerden etkilenmez — ikisini birlikte oku."
    )
    col1, col2 = st.columns(2)
    for faz, col, emoji in [('BOGA',col1,'🟢'),('AYI',col2,'🔴')]:
        fv = piv_df[piv_df['faz']==faz]
        with col:
            st.markdown(f"#### {emoji} {faz} — {len(fv)} donem")
            if len(fv)==0: st.info("Bu aralikta yeterli veri yok."); continue
            tablo = pd.DataFrame({
                'Metrik':  ['Sure (gun)','Degisim (%)','Hiz (%/gun)'],
                'Ort':     [f"{fv['sure_gun'].mean():.1f}",       f"{fv['fiyat_degisim_pct'].mean():.1f}%",   f"{fv['hiz_pct_gun'].mean():.3f}"],
                'Medyan':  [f"{fv['sure_gun'].median():.1f}",     f"{fv['fiyat_degisim_pct'].median():.1f}%", f"{fv['hiz_pct_gun'].median():.3f}"],
                'Std':     [f"{fv['sure_gun'].std():.1f}",        f"{fv['fiyat_degisim_pct'].std():.1f}%",   f"{fv['hiz_pct_gun'].std():.3f}"],
                'Min':     [f"{fv['sure_gun'].min():.1f}",        f"{fv['fiyat_degisim_pct'].min():.1f}%",   f"{fv['hiz_pct_gun'].min():.3f}"],
                'Max':     [f"{fv['sure_gun'].max():.1f}",        f"{fv['fiyat_degisim_pct'].max():.1f}%",   f"{fv['hiz_pct_gun'].max():.3f}"],
            })
            st.dataframe(tablo, use_container_width=True, hide_index=True)
    st.markdown("---")
    acik("<b>Histogram:</b> Bir cubuk ne kadar uzunsa o aralikta o kadar cok faz yasandi. "
         "Sola yigili = cogu faz kisa, saga cekik kuyruk = birkac cok uzun faz var.")
    fig = make_subplots(rows=1, cols=3, subplot_titles=['Sure (gun)','Degisim (%)','Hiz (%/gun)'])
    for faz, renk in [('BOGA','#00b894'),('AYI','#e17055')]:
        sub = piv_df[piv_df['faz']==faz]
        if len(sub)==0: continue
        for ci, kolon in enumerate(['sure_gun','fiyat_degisim_pct','hiz_pct_gun'],1):
            fig.add_trace(go.Histogram(x=sub[kolon].abs(), name=faz, marker_color=renk,
                                       opacity=0.65, showlegend=(ci==1)), row=1, col=ci)
    fig.update_layout(template='plotly_dark', height=380, barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)


# ── SAYFA 2 — KORELASYON ──────────────────────────────────────────────────────

elif sayfa == "2 — Korelasyon":
    st.markdown('<div class="baslik">Korelasyon Analizleri</div>', unsafe_allow_html=True)
    acik(
        "<b>Korelasyon</b> iki degerin birbiriyle iliski derecesini olcer (-1 ile +1).<br>"
        "+1: Biri artinca digeri de artar | 0: Iliski yok | -1: Biri artinca digeri azalir<br>"
        "0.3 uzeri zayif, 0.5 uzeri orta, 0.7 uzeri guclu iliski."
    )
    boga = piv_df[piv_df['faz']=='BOGA'].reset_index(drop=True)
    ayi  = piv_df[piv_df['faz']=='AYI'].reset_index(drop=True)
    n    = min(len(boga), len(ayi))
    if n < 3:
        st.warning("Bu aralikta korelasyon icin yeterli veri yok.")
    else:
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=['Boga: Sure vs % Hareket',
                                            'Ayi: Sure vs % Dusus',
                                            'Boga Suresi vs Sonraki Ayi Suresi'])
        fig.add_trace(go.Scatter(x=boga['sure_gun'], y=boga['fiyat_degisim_pct'], mode='markers',
                                 marker=dict(color='#00b894',size=9,opacity=0.8), showlegend=False), row=1,col=1)
        fig.add_trace(go.Scatter(x=ayi['sure_gun'], y=ayi['fiyat_degisim_pct'].abs(), mode='markers',
                                 marker=dict(color='#e17055',size=9,opacity=0.8), showlegend=False), row=1,col=2)
        fig.add_trace(go.Scatter(x=boga['sure_gun'].iloc[:n], y=ayi['sure_gun'].iloc[:n],
                                 mode='markers+text', text=[str(i+1) for i in range(n)],
                                 textposition='top center', marker=dict(color='#fdcb6e',size=9),
                                 showlegend=False), row=1,col=3)
        fig.update_layout(template='plotly_dark', height=420)
        st.plotly_chart(fig, use_container_width=True)
        r1 = boga['sure_gun'].corr(boga['fiyat_degisim_pct'])
        r2 = ayi['sure_gun'].corr(ayi['fiyat_degisim_pct'].abs())
        r3 = boga['sure_gun'].iloc[:n].corr(ayi['sure_gun'].iloc[:n])
        c1,c2,c3 = st.columns(3)
        for col, r, et in [(c1,r1,'Boga: Sure-Hareket'),(c2,r2,'Ayi: Sure-Dusus'),(c3,r3,'Boga-Ayi Sure')]:
            rr = '#00b894' if abs(r)>0.5 else '#fdcb6e' if abs(r)>0.3 else '#b2bec3'
            kart(col, f"{r:.3f}", et, rr)
        st.markdown("---")
        acik(
            f"Boga sure-hareket: <b>{r1:.2f}</b> — "
            f"{'Uzun boga daha buyuk yukselis uretti.' if r1>0.3 else 'Sure ile hareket buyuklugu arasinda belirgin iliski yok.'}<br>"
            f"Ayi sure-dusus: <b>{r2:.2f}</b> — "
            f"{'Uzun ayi daha derin dusus yaratti.' if r2>0.3 else 'Sure ile dusus derinligi arasinda belirgin iliski yok.'}<br>"
            f"Boga suresi → ayi suresi: <b>{r3:.2f}</b> — "
            f"{'Uzun boga sonrasi uzun ayi gelme egilimi var.' if r3>0.3 else 'Boga uzunlugu sonraki ayinin uzunlugunu belirlemez.'}"
        )


# ── SAYFA 3 — MARKET STRUCTURE ────────────────────────────────────────────────

elif sayfa == "3 — Market Structure":
    st.markdown('<div class="baslik">Market Structure — HH / HL / LH / LL</div>', unsafe_allow_html=True)
    renk_map = {'HH':'#00b894','HL':'#74b9ff','LH':'#e17055','LL':'#d63031','—':'#b2bec3'}
    acik(
        "Her yeni pivot bir oncekiyle karsilastirilir:<br>"
        f"<b style='color:{renk_map['HH']}'>HH:</b> Yeni tepe onceki tepeden yuksek — alicilar kontrolde<br>"
        f"<b style='color:{renk_map['HL']}'>HL:</b> Yeni dip onceki dipten yuksek — her dususte alim var<br>"
        f"<b style='color:{renk_map['LH']}'>LH:</b> Yeni tepe onceki tepeden dusuk — yukari ivme zayifliyor<br>"
        f"<b style='color:{renk_map['LL']}'>LL:</b> Yeni dip onceki dipten dusuk — satis baskisi gucleniyor<br><br>"
        "<b>Strateji:</b> Art arda HH+HL = long agirlikli. LH sonrasi HL gelmezse LL riski artar."
    )
    tp_ms = piv_df[piv_df['tip']=='TEPE'].copy().reset_index(drop=True)
    dp_ms = piv_df[piv_df['tip']=='DIP'].copy().reset_index(drop=True)
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['kapanis'], mode='lines',
                             line=dict(color='#636e72',width=1), opacity=0.5, name='Fiyat'))
    for _, row in tp_ms.iterrows():
        et = row.get('ms_etiket','—')
        fig.add_trace(go.Scatter(x=[row['tarih']], y=[row['fiyat']], mode='markers+text',
                                 text=[et], textposition='top center',
                                 textfont=dict(size=10, color=renk_map.get(et,'#b2bec3')),
                                 marker=dict(color=renk_map.get(et,'#b2bec3'), size=11, symbol='triangle-down'),
                                 showlegend=False))
    for _, row in dp_ms.iterrows():
        et = row.get('ms_etiket','—')
        fig.add_trace(go.Scatter(x=[row['tarih']], y=[row['fiyat']], mode='markers+text',
                                 text=[et], textposition='bottom center',
                                 textfont=dict(size=10, color=renk_map.get(et,'#b2bec3')),
                                 marker=dict(color=renk_map.get(et,'#b2bec3'), size=11, symbol='triangle-up'),
                                 showlegend=False))
    fig.update_layout(template='plotly_dark', height=600, yaxis_type='log', title=f'{yks} | {aralik}')
    st.plotly_chart(fig, use_container_width=True)
    tum_et = pd.concat([tp_ms[['ms_etiket']], dp_ms[['ms_etiket']]]).dropna()
    say    = tum_et['ms_etiket'].value_counts()
    c1,c2,c3,c4 = st.columns(4)
    for col, et in [(c1,'HH'),(c2,'HL'),(c3,'LH'),(c4,'LL')]:
        v = say.get(et,0); pct = v/say.sum()*100 if say.sum()>0 else 0
        kart(col, f"{v}  (%{pct:.0f})", et, renk_map[et])

    # ── Box Plot — TEPE→DIP ve DIP→TEPE ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Etiket Bazli Fiyat Hareketi")
    acik(
        "<b>Sol — TEPE sonrasi kac % dusuyor?</b> HH tepesinden sonra daha az mi, LH'dan sonra daha cok mu dusus geliyor?<br>"
        "<b>Sag — DIP sonrasi kac % yukseliyor?</b> HL dipten sonra guclu toparlanma mi, LL'den sonra zayif mi?"
    )
    tepe_yon = piv_df[piv_df['tip']=='TEPE'].copy()
    dip_yon  = piv_df[piv_df['tip']=='DIP'].copy()
    if len(tepe_yon)>0 and len(dip_yon)>0:
        fig_box = make_subplots(rows=1, cols=2, subplot_titles=[
            'TEPE → DIP: Tepeden sonra % dusus?',
            'DIP → TEPE: Dipten sonra % yukselis?'])
        for et in ['HH','LH']:
            sub = tepe_yon[tepe_yon['ms_etiket']==et].dropna(subset=['fiyat_degisim_pct'])
            if sub.empty: continue
            fig_box.add_trace(go.Box(y=sub['fiyat_degisim_pct'], name=et,
                                     marker_color=renk_map[et], boxpoints='all',
                                     jitter=0.3, pointpos=-1.5), row=1, col=1)
        for et in ['HL','LL']:
            sub = dip_yon[dip_yon['ms_etiket']==et].dropna(subset=['fiyat_degisim_pct'])
            if sub.empty: continue
            fig_box.add_trace(go.Box(y=sub['fiyat_degisim_pct'], name=et,
                                     marker_color=renk_map[et], boxpoints='all',
                                     jitter=0.3, pointpos=-1.5, showlegend=False), row=1, col=2)
        fig_box.add_hline(y=0, line_dash='dash', line_color='#b2bec3', opacity=0.4)
        fig_box.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig_box, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            rows = [{'Etiket':et,'Sayi':len(s),'Ort%':f"{s['fiyat_degisim_pct'].mean():.1f}%",
                     'Medyan%':f"{s['fiyat_degisim_pct'].median():.1f}%",'Std':f"{s['fiyat_degisim_pct'].std():.1f}%"}
                    for et in ['HH','LH']
                    for s in [tepe_yon[tepe_yon['ms_etiket']==et].dropna(subset=['fiyat_degisim_pct'])]
                    if not s.empty]
            if rows: st.markdown("**TEPE → DIP**"); st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with col2:
            rows = [{'Etiket':et,'Sayi':len(s),'Ort%':f"{s['fiyat_degisim_pct'].mean():.1f}%",
                     'Medyan%':f"{s['fiyat_degisim_pct'].median():.1f}%",'Std':f"{s['fiyat_degisim_pct'].std():.1f}%"}
                    for et in ['HL','LL']
                    for s in [dip_yon[dip_yon['ms_etiket']==et].dropna(subset=['fiyat_degisim_pct'])]
                    if not s.empty]
            if rows: st.markdown("**DIP → TEPE**"); st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Pattern Analizi ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Pattern Analizi")
    piv_seq = piv_df[piv_df['ms_etiket'].isin(['HH','HL','LH','LL'])].reset_index(drop=True)
    if len(piv_seq) < 6:
        st.warning("Pattern analizi icin yeterli pivot yok. Daha genis aralik secin.")
    else:
        seq = piv_seq['ms_etiket'].tolist()
        etiketler = ['HH','HL','LH','LL']

        # Markov
        st.markdown("##### 1. Markov Gecis Matrisi")
        acik("Satirlar: simdi ne var. Sutunlar: sonra ne geldi. "
             "HH goruldukten sonra en cok ne geliyor? Bu gecis olasılıkları pozisyon kararini etkiler.")
        matrix_c = pd.DataFrame(0, index=etiketler, columns=etiketler)
        for i in range(len(seq)-1):
            if seq[i] in etiketler and seq[i+1] in etiketler:
                matrix_c.loc[seq[i], seq[i+1]] += 1
        matrix_p = matrix_c.div(matrix_c.sum(axis=1).replace(0,1), axis=0)*100
        fig_heat = go.Figure(data=go.Heatmap(
            z=matrix_p.values, x=[f'→ {e}' for e in etiketler], y=etiketler,
            colorscale='RdYlGn',
            text=[[f"{matrix_c.iloc[i,j]} kez\n%{matrix_p.iloc[i,j]:.0f}" for j in range(4)] for i in range(4)],
            texttemplate="%{text}", textfont=dict(size=12), zmin=0, zmax=100))
        fig_heat.update_layout(template='plotly_dark', height=360,
                               title='Gecis Matrisi — Sonraki etiket kac % ihtimalle geliyor?',
                               xaxis_title='Sonraki', yaxis_title='Mevcut')
        st.plotly_chart(fig_heat, use_container_width=True)
        yorumlar = []
        for et in etiketler:
            if matrix_c.loc[et].sum()==0: continue
            en_cok = matrix_p.loc[et].idxmax(); pct_val = matrix_p.loc[et].max()
            yorumlar.append(f"<b style='color:{renk_map[et]}'>{et}</b> → en cok "
                            f"<b style='color:{renk_map[en_cok]}'>{en_cok}</b> (%{pct_val:.0f}, {int(matrix_c.loc[et,en_cok])} kez)")
        acik(" | ".join(yorumlar))

        # Run Analizi
        st.markdown("---")
        st.markdown("##### 2. Art Arda Tekrar (Run) Analizi")
        acik("Ayni etiket ard arda kac kere geldi ve bittikten sonra ne geldi? "
             "'3 kez ust uste HH geldikten sonra ne oldu?' sorusunu yanıtlar.")
        runs, i = [], 0
        while i < len(seq):
            cur, length = seq[i], 0
            while i < len(seq) and seq[i] == cur: length += 1; i += 1
            runs.append({'etiket':cur,'uzunluk':length,'sonraki':seq[i] if i<len(seq) else None})
        run_df = pd.DataFrame(runs)
        col1, col2 = st.columns(2)
        with col1:
            run_ozet = run_df.groupby('etiket')['uzunluk'].value_counts().unstack(fill_value=0)
            run_ozet.columns = [f'{c}x' for c in run_ozet.columns]
            st.markdown("**Kac kez ard arda?**"); st.dataframe(run_ozet, use_container_width=True)
        with col2:
            rsnr = run_df.dropna(subset=['sonraki']).groupby(['etiket','sonraki']).size().unstack(fill_value=0)
            st.markdown("**Run bittikten sonra ne geldi?**"); st.dataframe(rsnr, use_container_width=True)

        # 4'lü Pattern
        st.markdown("---")
        st.markdown("##### 3. En Sık Görülen 4'lu Diziler")
        acik("Ard arda 4 pivot kombinasyonu. Yesil = boga agirlikli, kirmizi = ayi agirlikli.")
        patterns_4 = {}
        for i in range(len(seq)-3):
            pat = '-'.join(seq[i:i+4])
            patterns_4[pat] = patterns_4.get(pat,0) + 1
        pat_df = pd.DataFrame([{'Pattern':k,'Sayi':v} for k,v in
                               sorted(patterns_4.items(), key=lambda x:-x[1])]).head(15)
        def pren(p):
            ep = p.split('-')
            b = sum(1 for e in ep if e in ['HH','HL']); a = sum(1 for e in ep if e in ['LH','LL'])
            return '#00b894' if b>a else '#e17055' if a>b else '#fdcb6e'
        fig_pat = go.Figure(go.Bar(x=pat_df['Sayi'], y=pat_df['Pattern'], orientation='h',
                                   marker_color=[pren(p) for p in pat_df['Pattern']],
                                   text=pat_df['Sayi'], textposition='outside'))
        fig_pat.update_layout(template='plotly_dark', height=500,
                              title='En Sik 4lu Diziler', xaxis_title='Kac kez', yaxis_autorange='reversed')
        st.plotly_chart(fig_pat, use_container_width=True)

        # N Streak
        st.markdown("---")
        st.markdown("##### 4. N Kez Ust Uste HH veya LL Sonrasi Ne Oldu?")
        acik("Ust uste HH veya LL sayisi arttikca sonraki etikette ne degisiyor? "
             "Trend ne kadar surerse sursin bir donus olasiligi var mi?")
        col1, col2 = st.columns(2)
        for col, hedef, baslik_s in [(col1,'HH','HH Streak'),(col2,'LL','LL Streak')]:
            with col:
                st.markdown(f"**{baslik_s} sonrasi ne geldi?**")
                sonuclar = []
                for ns in range(1,6):
                    for i in range(len(seq)-ns):
                        if all(seq[i+j]==hedef for j in range(ns)):
                            si = i+ns
                            if si < len(seq): sonuclar.append({'n':f'{ns}x','sonraki':seq[si]})
                if sonuclar:
                    sdf = pd.DataFrame(sonuclar)
                    ct  = pd.crosstab(sdf['n'], sdf['sonraki'])
                    ctp = ct.div(ct.sum(axis=1), axis=0)*100
                    st.dataframe(ctp.round(0).astype(int).astype(str)+'%', use_container_width=True)
                else:
                    st.info("Yeterli veri yok.")


# ── SAYFA 4 — ZİGZAG ORAN ────────────────────────────────────────────────────

elif sayfa == "4 — ZigZag Oran Analizi":
    st.markdown('<div class="baslik">ZigZag Oran Analizi</div>', unsafe_allow_html=True)
    acik(
        "<b>Formul:</b> oran = (mevcut hareket / |onceki hareket|) x 100<br><br>"
        "10 birim dustukten sonra 5 birim yukseldi → +50% (HL, zayif toparlanma)<br>"
        "10 birim dustukten sonra 15 birim yukseldi → +150% (HH, guclu hamle)<br>"
        "10 birim yukseldikten sonra 20 birim dustu → -200% (LL, sert satis)<br><br>"
        f"<b style='color:#00b894'>+100% uzeri:</b> HH bolgesi<br>"
        f"<b style='color:#fdcb6e'>0% ile +100%:</b> HL bolgesi<br>"
        f"<b style='color:#e17055'>-100% ile 0%:</b> LH bolgesi<br>"
        f"<b style='color:#d63031'>-100% altı:</b> LL bolgesi"
    )
    piv_z = piv_df.copy()
    piv_z['hareket']      = piv_z['fiyat'].diff()
    piv_z['prev_hareket'] = piv_z['hareket'].shift(1)
    piv_z['oran_pct']     = (piv_z['hareket']/piv_z['prev_hareket'].abs())*100
    temiz = piv_z.dropna(subset=['oran_pct'])
    if len(temiz) < 3:
        st.warning("Yeterli veri yok.")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.45, 0.55], vertical_spacing=0.04,
                            subplot_titles=[
                                'Pivot Fiyatlari (log scale)',
                                'ZigZag Oran (%) — Her cubuk bir hareket, oncekiyle kiyaslanmis'])
        fig.add_trace(go.Scatter(x=piv_df['tarih'], y=piv_df['fiyat'], mode='lines+markers',
                                 line=dict(color='#636e72',width=1), marker=dict(size=4),
                                 showlegend=False), row=1, col=1)
        renkler = ['#00b894' if v>100 else '#fdcb6e' if v>0 else '#e17055' if v>-100 else '#d63031'
                   for v in temiz['oran_pct']]
        fig.add_trace(go.Bar(x=temiz['tarih'], y=temiz['oran_pct'], marker_color=renkler,
                             opacity=0.85, showlegend=False), row=2, col=1)
        for yv, rr in [(100,'#00b894'),(-100,'#d63031'),(0,'#b2bec3')]:
            fig.add_hline(y=yv, line_dash='dash', line_color=rr, opacity=0.5, row=2, col=1)
        fig.update_yaxes(type='log', row=1, col=1)
        fig.update_layout(template='plotly_dark', height=750)
        st.plotly_chart(fig, use_container_width=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        for col, et, val in [(c1,'Ort',f"{temiz['oran_pct'].mean():.1f}%"),
                             (c2,'Medyan',f"{temiz['oran_pct'].median():.1f}%"),
                             (c3,'Std',f"{temiz['oran_pct'].std():.1f}%"),
                             (c4,'Min',f"{temiz['oran_pct'].min():.1f}%"),
                             (c5,'Max',f"{temiz['oran_pct'].max():.1f}%")]:
            kart(col, val, et)

        st.markdown("---")
        acik("<b>Box Plot:</b> Her etiket kategorisi icin oran dagilimi. "
             "HH bolgesi hareketleri genellikle ne kadar? LL bolgesi ne kadar derin?")
        fig_box = go.Figure()
        renk_map = {'HH':'#00b894','HL':'#74b9ff','LH':'#e17055','LL':'#d63031'}
        def oran_kat(v):
            if v>100: return 'HH'
            if v>0:   return 'HL'
            if v>-100:return 'LH'
            return 'LL'
        temiz = temiz.copy()
        temiz['oran_kat'] = temiz['oran_pct'].apply(oran_kat)
        for et in ['HH','HL','LH','LL']:
            sub = temiz[temiz['oran_kat']==et]
            if sub.empty: continue
            fig_box.add_trace(go.Box(y=sub['oran_pct'], name=et, marker_color=renk_map[et],
                                     boxpoints='all', jitter=0.3, pointpos=-1.5))
        fig_box.add_hline(y=0, line_dash='dash', line_color='#b2bec3', opacity=0.4)
        fig_box.add_hline(y=100, line_dash='dash', line_color='#00b894', opacity=0.4)
        fig_box.add_hline(y=-100, line_dash='dash', line_color='#d63031', opacity=0.4)
        fig_box.update_layout(template='plotly_dark', height=420,
                              title='Oran Kategorileri — Dagilim', yaxis_title='Oran (%)')
        st.plotly_chart(fig_box, use_container_width=True)


# ── SAYFA 5 — MOVING AVERAGE ANALIZI ────────────────────────────────────────────────────

elif sayfa == "5 — MA Analysis":
    st.markdown('<div class="baslik">MA Analysis — Moving Average</div>', unsafe_allow_html=True)
    acik(
        "<b>Moving Average (MA)</b> fiyatın trendini daha temiz görmek için kullanılır.<br>"
        "<b>SMA:</b> Basit hareketli ortalama — daha yavaş, daha stabil.<br>"
        "<b>EMA:</b> Üstel hareketli ortalama — son veriye daha duyarlı, daha hızlı tepki verir.<br><br>"
        "Veri 4 saatlik geldiği için burada önce seçtiğin zaman çerçevesine "
        "(günlük / haftalık) dönüştürülür, ardından MA hesaplanır.<br><br>"
        "<b>MA Ribbon:</b> Birden fazla MA'nın birlikte görünmesidir. Ribbon açıldığında "
        "trend sıkışması, açılması ve yön değişimi daha rahat okunur."
    )

    c1, c2 = st.columns(2)
    with c1:
        ma_tipi = st.selectbox("MA Tipi:", ["SMA", "EMA"], index=0)
    with c2:
        ma_zaman = st.selectbox("MA Zaman Çerçevesi:", ["Gunluk", "Haftalik"], index=0)

    secili_ma = st.multiselect(
        "MA Periyotlari (istediklerini sec):",
        MA_SECENEKLERI,
        default=[20, 50, 100, 200]
    )

    c3, c4 = st.columns(2)
    with c3:
        ribbon_acik = st.checkbox("MA Ribbon Goster", value=True)
    with c4:
        fiyat_goster = st.checkbox("Mum Grafigi Goster", value=True)

    if len(secili_ma) == 0:
        st.warning("En az bir MA periyodu sec.")
    else:
        secili_ma = sorted(secili_ma)

        ma_df_tam = ohlc_resample(df_tam, ma_zaman)
        ma_df = ma_df_tam[(ma_df_tam.index >= bas) & (ma_df_tam.index <= bugun)].copy()

        for ma in secili_ma:
            ma_df[f"{ma_tipi}_{ma}"] = ma_hesapla(ma_df['kapanis'], ma_tipi, ma)

        son_fiyat = ma_df['kapanis'].iloc[-1]

        st.markdown("#### MA Grafiği")
        fig_ma = go.Figure()

        if fiyat_goster:
            fig_ma.add_trace(go.Candlestick(
                x=ma_df.index,
                open=ma_df['acilis'],
                high=ma_df['yuksek'],
                low=ma_df['dusuk'],
                close=ma_df['kapanis'],
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                name='Fiyat'
            ))

        # Ribbon için MA çizgileri
        for i, ma in enumerate(secili_ma):
            kolon = f"{ma_tipi}_{ma}"
            fig_ma.add_trace(go.Scatter(
                x=ma_df.index,
                y=ma_df[kolon],
                mode='lines',
                name=f"{ma_tipi} {ma}",
                line=dict(width=2)
            ))

        # Ribbon fill: seçilen MA'lar arasında alan doldur
        if ribbon_acik and len(secili_ma) >= 2:
            for i in range(len(secili_ma) - 1):
                ma_kisa = secili_ma[i]
                ma_uzun = secili_ma[i + 1]

                col_kisa = f"{ma_tipi}_{ma_kisa}"
                col_uzun = f"{ma_tipi}_{ma_uzun}"

                # önce alt çizgi
                fig_ma.add_trace(go.Scatter(
                    x=ma_df.index,
                    y=ma_df[col_uzun],
                    mode='lines',
                    line=dict(width=0),
                    name=f"Ribbon Alt {ma_uzun}",
                    showlegend=False,
                    hoverinfo='skip'
                ))

                # sonra üst çizgi + fill
                fig_ma.add_trace(go.Scatter(
                    x=ma_df.index,
                    y=ma_df[col_kisa],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    name=f"Ribbon {ma_kisa}-{ma_uzun}",
                    showlegend=False,
                    hoverinfo='skip'
                ))

        fig_ma.update_layout(
            template='plotly_dark',
            height=700,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', y=1.02),
            title=f"BTC {ma_zaman} {ma_tipi} Analizi"
        )
        st.plotly_chart(fig_ma, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Son Durum Özeti")

        ozet_rows = []
        for ma in secili_ma:
            kolon = f"{ma_tipi}_{ma}"
            ma_deger = ma_df[kolon].iloc[-1]

            if pd.isna(ma_deger):
                durum = "Yetersiz veri"
                fark_pct = np.nan
            else:
                fark_pct = (son_fiyat / ma_deger - 1) * 100
                durum = "Ustunde" if son_fiyat > ma_deger else "Altinda" if son_fiyat < ma_deger else "Esit"

            ozet_rows.append({
                "MA": f"{ma_tipi} {ma}",
                "Son Deger": f"${ma_deger:,.0f}" if pd.notna(ma_deger) else "—",
                "Fiyat Iliskisi": durum,
                "Fark (%)": f"{fark_pct:.2f}%" if pd.notna(fark_pct) else "—"
            })

        ozet_df = pd.DataFrame(ozet_rows)
        st.dataframe(ozet_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### MA Sinyal Kartlari")

        kart_sayisi = min(len(secili_ma), 4)
        cols = st.columns(kart_sayisi)

        for i, ma in enumerate(secili_ma[:kart_sayisi]):
            kolon = f"{ma_tipi}_{ma}"
            ma_deger = ma_df[kolon].iloc[-1]

            if pd.isna(ma_deger):
                kart(cols[i], "—", f"{ma_tipi} {ma}", "#b2bec3")
            else:
                renk = "#00b894" if son_fiyat > ma_deger else "#e17055"
                fark_pct = (son_fiyat / ma_deger - 1) * 100
                kart(cols[i], f"%{fark_pct:.2f}", f"Fiyat vs {ma_tipi} {ma}", renk)

        st.markdown("---")
        st.markdown("#### Ribbon Yorumu")

        ustunde = []
        altinda = []

        for ma in secili_ma:
            kolon = f"{ma_tipi}_{ma}"
            ma_deger = ma_df[kolon].iloc[-1]
            if pd.isna(ma_deger):
                continue
            if son_fiyat > ma_deger:
                ustunde.append(ma)
            else:
                altinda.append(ma)

        yorum = (
            f"Seçilen zaman çerçevesi <b>{ma_zaman}</b>, MA tipi <b>{ma_tipi}</b>.<br>"
            f"Son {ma_zaman.lower()} kapanış fiyatı <b>${son_fiyat:,.0f}</b>.<br><br>"
        )

        if ribbon_acik:
            yorum += (
                f"Ribbon, seçtiğin MA seti üzerinden oluşturuldu: "
                f"<b>{', '.join(map(str, secili_ma))}</b>.<br><br>"
            )

        if ustunde and altinda:
            yorum += (
                f"BTC şu anda <b>{', '.join(map(str, ustunde))}</b> MA'larının üstünde, "
                f"<b>{', '.join(map(str, altinda))}</b> MA'larının altında. "
                f"Bu görünüm karışık / geçiş fazına işaret eder."
            )
        elif ustunde and not altinda:
            yorum += (
                f"BTC seçilen tüm MA'ların <b>üstünde</b>. Ribbon genelde yukarı açılıyorsa bu güçlü trend görünümüdür."
            )
        elif altinda and not ustunde:
            yorum += (
                f"BTC seçilen tüm MA'ların <b>altında</b>. Ribbon aşağı açılıyorsa bu zayıf görünüm / baskı altında trend demektir."
            )
        else:
            yorum += "Yorum için yeterli veri oluşmadı."

        acik(yorum)

        st.markdown("---")
        st.markdown("#### MA Kesisim Tablosu")

        if len(secili_ma) >= 2:
            cross_rows = []

            for i in range(len(secili_ma) - 1):
                kisa = secili_ma[i]
                uzun = secili_ma[i + 1]

                kisa_kolon = f"{ma_tipi}_{kisa}"
                uzun_kolon = f"{ma_tipi}_{uzun}"

                son_kisa = ma_df[kisa_kolon].iloc[-1]
                son_uzun = ma_df[uzun_kolon].iloc[-1]

                if pd.isna(son_kisa) or pd.isna(son_uzun):
                    durum = "Yetersiz veri"
                    fark = np.nan
                else:
                    fark = (son_kisa / son_uzun - 1) * 100
                    durum = "Bullish Ustu" if son_kisa > son_uzun else "Bearish Alti"

                cross_rows.append({
                    "Kesisim": f"{ma_tipi} {kisa} / {ma_tipi} {uzun}",
                    "Durum": durum,
                    "Fark (%)": f"{fark:.2f}%" if pd.notna(fark) else "—"
                })

            st.dataframe(pd.DataFrame(cross_rows), use_container_width=True, hide_index=True)
# Arm Reachability Tracker

Framework per registrare lo spazio raggiungibile dal braccio (workspace) e
visualizzarlo tramite scheletro/avatar in tempo reale su webapp.

Pipeline ispirata a: *Ovur et al., "Optimizing Ergonomics for Robot-to-Human
Object Handovers", IEEE T-CDS 2026* (uso di camera + body tracking +
riferimento corporeo per mappare i punti raggiungibili).

## Stato del progetto

- [x] **Step 1** — Stream RealSense D435i + MediaPipe Pose, calcolo posizione
      3D reale (in metri) del polso, visualizzazione a schermo.
- [x] Step 2 — Calibrazione del riferimento corporeo (origine = torace/spalla)
- [x] Step 3 — Registrazione dataset dei punti raggiunti dalla mano
- [ ] Step 4 — Server FastAPI + WebSocket per lo streaming dei dati
- [ ] Step 5 — Webapp Three.js: scheletro semplice in tempo reale
- [ ] Step 6 — Visualizzazione nuvola di punti / workspace accumulato

## Step 1 — Setup e primo test

### Requisiti
- Python 3.9–3.11 (mediapipe non supporta ancora tutte le versioni più recenti)
- Una Intel RealSense D435i collegata via USB 3.0

### Installazione

```bash
cd backend
python -m venv venv
source venv/bin/activate  # su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Esecuzione

```bash
python step1_pose_stream.py
```

Dovresti vedere una finestra video con lo scheletro disegnato sopra il tuo
corpo, e nel terminale le coordinate 3D (in metri) del polso destro
aggiornate in tempo reale, tipo:

```
Polso DX: x=0.123 y=-0.045 z=0.812 m
```

Queste coordinate sono ancora nel **riferimento della camera** (origine =
camera stessa). Nello Step 2 le trasformeremo nel riferimento del corpo,
come nel paper (origine = torace), così saranno indipendenti da dove è
posizionata la camera.

## Step 2 — Calibrazione del riferimento corporeo

Lo Step 2 prende il risultato di Step 1 e costruisce un sistema di riferimento
centrato sul corpo, invece che sulla camera.

### Obiettivo
- Usare le spalle come origine approssimativa del torso
- Definire assi del corpo (x, y, z) a partire dai landmark di MediaPipe
- Esporre la posizione del polso nel nuovo riferimento, così da poterla usare
  per registrare il workspace raggiungibile

### Esecuzione

```bash
cd backend
source venv/bin/activate
python step2_body_reference.py
```

Dovresti vedere lo stream video con il corpo tracciato e, nel terminale, una
riga del tipo:

```
Polso body-frame: x=0.123 y=-0.045 z=0.812 m
```

Queste coordinate sono ora riferite al corpo, non alla camera.

## Step 3 — Registrazione del workspace raggiungibile

Lo Step 3 aggiunge la parte di registrazione: il polso viene prima filtrato
per ridurre il rumore della depth e poi salvato in un file CSV, così potrà
essere usato per ricostruire il movimento e il workspace raggiungibile.

### Esecuzione

```bash
cd backend
source venv/bin/activate
python step3_record_reachability.py --output recordings/reachability.csv
```

Durante l'esecuzione vedrai lo stream video e, nel terminale, i valori grezzi e
quelli filtrati del polso in riferimento corporeo. I dati vengono salvati in
CSV per l'uso nei passi successivi.

## Step 4 — Server live per l'avatar

Lo Step 4 aggiunge un piccolo server FastAPI + WebSocket così il backend può
inviare i dati di tracking a una web app in tempo reale.

### Esecuzione

```bash
cd backend
source venv/bin/activate
python step4_live_server.py
```

Poi apri il browser su:

```text
http://localhost:8000/health
```

Il server espone anche un endpoint WebSocket su `/ws` che potrà essere usato
dalla futura UI dell'avatar.

### Problemi comuni
- **`pyrealsense2` non si installa**: su alcuni sistemi (es. Linux ARM, alcune
  distro) serve compilare da sorgente il SDK librealsense. Fammi sapere il tuo
  OS se capita.
- **Nessun corpo rilevato**: assicurati di essere interamente visibile nel
  campo visivo della camera, con buona illuminazione.
- **z troppo rumoroso**: normale a bordo mani/dita, la depth della D435i è
  meno precisa su superfici piccole/riflettenti; lo affronteremo con un
  filtro (media mobile) in uno step successivo.

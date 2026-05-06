"""
TtakaIQ — Railway Flask prediction server
------------------------------------------
Endpoints:
  GET  /predict        ← called by AwardSpace soil_insert.php
  POST /upload-model   ← called by Colab after training
  GET  /health         ← Railway health check
  GET  /retrain-status ← check what model version is loaded
"""

import os, json, joblib, numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ── Config (set these as Railway environment variables) ────────
SECRET_KEY   = os.environ.get('TTAKAIQ_SECRET', 'basher-912-1245-jsgd')
MODELS_DIR   = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

# ── Crop profiles (same ranges as your PHP) ────────────────────
CROP_PROFILES = {
    'Maize':       {'N':[80,200],  'P':[30,80],  'K':[100,200], 'pH':[5.8,7.0], 'EC':[200,800],  'moisture':[40,70],  'soil_temp':[18,30]},
    'Beans':       {'N':[20,60],   'P':[40,80],  'K':[80,150],  'pH':[6.0,7.5], 'EC':[150,700],  'moisture':[45,70],  'soil_temp':[16,28]},
    'Bananas':     {'N':[100,300], 'P':[20,50],  'K':[200,400], 'pH':[5.5,7.0], 'EC':[300,900],  'moisture':[50,80],  'soil_temp':[20,30]},
    'Coffee':      {'N':[60,120],  'P':[20,40],  'K':[80,150],  'pH':[5.5,6.5], 'EC':[200,700],  'moisture':[45,70],  'soil_temp':[18,26]},
    'Cassava':     {'N':[40,80],   'P':[20,50],  'K':[60,120],  'pH':[5.5,7.0], 'EC':[100,600],  'moisture':[35,65],  'soil_temp':[20,32]},
    'SweetPotato': {'N':[40,80],   'P':[30,60],  'K':[100,200], 'pH':[5.5,7.0], 'EC':[150,700],  'moisture':[40,70],  'soil_temp':[20,30]},
    'Tomatoes':    {'N':[80,150],  'P':[40,80],  'K':[150,300], 'pH':[6.0,7.0], 'EC':[250,800],  'moisture':[50,75],  'soil_temp':[18,28]},
    'Rice':        {'N':[80,150],  'P':[30,60],  'K':[80,150],  'pH':[5.5,7.0], 'EC':[100,500],  'moisture':[60,90],  'soil_temp':[20,35]},
    'Sugarcane':   {'N':[100,200], 'P':[30,60],  'K':[150,300], 'pH':[6.0,7.5], 'EC':[200,800],  'moisture':[55,80],  'soil_temp':[20,35]},
}
FEATURES   = ['nitrogen','phosphorus','potassium','ph','ec','moisture','soil_temp']
LABEL_MAP  = {'POOR':0, 'MODERATE':1, 'GOOD':2}
INV_LABEL  = {0:'POOR', 1:'MODERATE', 2:'GOOD'}

# ── Model store (lives in memory after first load) ─────────────
crop_models   = {}   # {crop: RandomForestClassifier}
model_version = 'synthetic-v0'

def synthetic_samples(crop, label, n=100):
    profile = CROP_PROFILES.get(crop, CROP_PROFILES['Maize'])
    keys = ['N','P','K','pH','EC','moisture','soil_temp']
    feat  = ['nitrogen','phosphorus','potassium','ph','ec','moisture','soil_temp']
    X = []
    for _ in range(n):
        row = []
        for k in keys:
            lo, hi = profile[k]
            if   label == 2: row.append(np.random.uniform(lo, hi))
            elif label == 1: row.append(np.random.uniform(lo*0.55,lo) if np.random.rand()>.5 else np.random.uniform(hi,hi*1.45))
            else:            row.append(np.random.uniform(lo*0.1, lo*0.5) if np.random.rand()>.5 else np.random.uniform(hi*1.5,hi*2.2))
        X.append(row)
    return np.array(X)

def train_synthetic_fallback():
    """Train on pure synthetic data so server is never modelless."""
    global crop_models, model_version
    print("Training synthetic fallback models...")
    for crop in CROP_PROFILES:
        X = np.vstack([synthetic_samples(crop, l, 120) for l in [0,1,2]])
        y = np.array([l for l in [0,1,2] for _ in range(120)])
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=None, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(X, y)
        crop_models[crop] = rf
    model_version = 'synthetic-v0'
    print("Synthetic fallback ready.")

def load_saved_models():
    """Load the latest uploaded .pkl from Colab if it exists."""
    global crop_models, model_version
    pkl_path = MODELS_DIR / 'crop_models.pkl'
    ver_path = MODELS_DIR / 'version.txt'
    if pkl_path.exists():
        try:
            crop_models   = joblib.load(pkl_path)
            model_version = ver_path.read_text().strip() if ver_path.exists() else 'uploaded-unknown'
            print(f"Loaded saved models: {model_version}")
            return True
        except Exception as e:
            print(f"Failed to load saved models: {e}")
    return False

# ── Boot: load saved model or fall back to synthetic ──────────
if not load_saved_models():
    train_synthetic_fallback()

# ── Recommendation builder ─────────────────────────────────────
def build_reply(verdict, crop, features, conf):
    N,P,K,pH,EC,moist,st = features
    p = CROP_PROFILES.get(crop, {})
    recs = []
    if p:
        if N     < p['N'][0]:        recs.append(f"Boost N (+{int(p['N'][0]-N)} mg/kg)")
        if P     < p['P'][0]:        recs.append(f"Boost P (+{int(p['P'][0]-P)} mg/kg)")
        if K     < p['K'][0]:        recs.append(f"Boost K (+{int(p['K'][0]-K)} mg/kg)")
        if pH    < p['pH'][0]:       recs.append(f"Apply lime (pH {pH:.1f} low)")
        if pH    > p['pH'][1]:       recs.append(f"Apply sulfur (pH {pH:.1f} high)")
        if moist < p['moisture'][0]: recs.append("Irrigate (moisture low)")
    base = f"{verdict}: {crop} — Python RF {conf:.1f}%. "
    base += "All params optimal." if not recs else "Recs: " + "; ".join(recs[:3]) + "."
    return base

# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route('/health')
def health():
    return jsonify({
        'status':        'ok',
        'model_version': model_version,
        'crops_loaded':  list(crop_models.keys())
    })


@app.route('/ping', methods=['GET'])
def ping():
    return {"status": "alive"}, 200


@app.route('/retrain-status')
def retrain_status():
    return jsonify({'model_version': model_version, 'crops': list(crop_models.keys())})

# ── /predict — called by soil_insert.php ──────────────────────
@app.route('/predict', methods=['GET'])
def predict():
    # Authenticate
    if request.args.get('secret','') != SECRET_KEY:
        return jsonify({'error':'unauthorized'}), 403

    try:
        crop = request.args.get('crop', 'Maize')
        if crop == 'BestCrop':
            return predict_best_crop(request.args)

        if crop not in crop_models:
            crop = 'Maize'

        features = [
            float(request.args.get('n',    0)),
            float(request.args.get('p',    0)),
            float(request.args.get('k',    0)),
            float(request.args.get('ph',   7)),
            float(request.args.get('ec',   0)),
            float(request.args.get('moist',0)),
            float(request.args.get('st',  25)),
        ]

        model  = crop_models[crop]
        proba  = model.predict_proba([features])[0]
        classes = model.classes_
        pred   = classes[int(np.argmax(proba))]
        conf   = round(float(np.max(proba)) * 100, 1)
        verdict = INV_LABEL[pred]
        prob_dict = {INV_LABEL[c]: round(float(p)*100,1) for c,p in zip(classes,proba)}
        reply  = build_reply(verdict, crop, features, conf)

        return jsonify({
            'verdict':       verdict,
            'confidence':    conf,
            'probabilities': prob_dict,
            'reply':         reply,
            'model_version': model_version
        })

    except Exception as e:
        return jsonify({
            'verdict':'POOR', 'confidence':0.0,
            'reply': f'POOR: Prediction error — {e}',
            'model_version': model_version
        }), 500

def predict_best_crop(args):
    features = [
        float(args.get('n',0)),  float(args.get('p',0)),
        float(args.get('k',0)),  float(args.get('ph',7)),
        float(args.get('ec',0)), float(args.get('moist',0)),
        float(args.get('st',25))
    ]
    scores = []
    for crop, model in crop_models.items():
        proba   = model.predict_proba([features])[0]
        classes = model.classes_
        pred    = classes[int(np.argmax(proba))]
        conf    = round(float(np.max(proba))*100, 1)
        verdict = INV_LABEL[pred]
        score   = pred * conf
        scores.append({'crop':crop,'verdict':verdict,'conf':conf,'score':score})
    scores.sort(key=lambda x: -x['score'])
    rank_str = "RANK: " + " ".join([f"{i+1}.{s['crop']}({s['verdict']})" for i,s in enumerate(scores[:5])])
    return jsonify({
        'verdict':'RANK', 'confidence': scores[0]['conf'],
        'reply': rank_str, 'rankings': scores,
        'model_version': model_version
    })

# ── /upload-model — called by Colab after training ────────────
@app.route('/upload-model', methods=['POST'])
def upload_model():
    if request.headers.get('X-Secret','') != SECRET_KEY:
        return jsonify({'error':'unauthorized'}), 403
    try:
        # Colab sends the .pkl as binary body, version in header
        version = request.headers.get('X-Model-Version', 'uploaded')
        pkl_bytes = request.data
        if len(pkl_bytes) < 100:
            return jsonify({'error':'empty body'}), 400

        pkl_path = MODELS_DIR / 'crop_models.pkl'
        ver_path = MODELS_DIR / 'version.txt'
        pkl_path.write_bytes(pkl_bytes)
        ver_path.write_text(version)

        # Reload into memory immediately
        global crop_models, model_version
        crop_models   = joblib.load(pkl_path)
        model_version = version
        print(f"Model updated to version: {version}")

        return jsonify({
            'status':  'ok',
            'version': model_version,
            'crops':   list(crop_models.keys())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

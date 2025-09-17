import os
import json
import re
import logging
import hashlib
import torch
import base64
from typing import List, Optional, Dict
from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory
from flask_wtf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import BaseModel, ValidationError
from cryptography.fernet import Fernet
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from flask_session import Session
from bs4 import BeautifulSoup
from oauthlib.oauth2.rfc6749.errors import AccessDeniedError, OAuth2Error

# -------------------- Load env --------------------
load_dotenv("startup.env")

# -------------------- Config & Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY"),
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True
)

csrf = CSRFProtect(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
OAUTH2_REDIRECT_URI = os.environ.get("OAUTH2_REDIRECT_URI", "https://127.0.0.1:5000/oauth2callback")
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/gmail.readonly",
    "openid"
]

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    logger.warning("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET not set. OAuth won't work until configured.")

# -------------------- Encryption for session --------------------
def _derive_fernet_key(secret: str) -> bytes:
    digest = hashlib.sha256(secret.encode()).digest()
    return base64.urlsafe_b64encode(digest)

FERNET_KEY = os.environ.get("FERNET_KEY")
if not FERNET_KEY:
    raise RuntimeError("FERNET_KEY non impostata!")
FERNET = Fernet(_derive_fernet_key(FERNET_KEY))

# -------------------- Model loading --------------------
MODEL_PATH = os.environ.get("MODEL_PATH") or "./sentiment_model"
logger.info("Loading tokenizer and model from %s", MODEL_PATH)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    raise

# -------------------- Helpers --------------------
class AnalyzeQuery(BaseModel):
    limit: Optional[int] = 10
    offset: Optional[int] = 0

def creds_to_dict(creds: Credentials) -> dict:
    return {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes,
    }

def save_credentials_to_session(creds: Credentials):
    creds_json = creds_to_dict(creds)
    session['creds'] = FERNET.encrypt(json.dumps(creds_json).encode()).decode()

def load_credentials_from_session() -> Optional[Credentials]:
    enc = session.get('creds')
    if not enc:
        return None
    try:
        creds_json = json.loads(FERNET.decrypt(enc.encode()).decode())
        creds = Credentials(**creds_json)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            save_credentials_to_session(creds)
        return creds
    except Exception as e:
        logger.exception("Errore caricando credenziali dalla sessione: %s", e)
        return None

def interpret_logits(outputs: torch.Tensor) -> float:
    logits = outputs.detach().cpu()
    num_labels = getattr(model.config, "num_labels", None)
    if num_labels is None or num_labels == 1:
        return float(logits.squeeze().item())
    elif num_labels == 2:
        probs = torch.sigmoid(logits.squeeze()) if logits.numel() == 1 else torch.softmax(logits.squeeze(), dim=-1)
        return float(probs[1].item()) if probs.dim() > 0 else float(probs.item())
    else:
        probs = torch.softmax(logits.squeeze(), dim=-1)
        return float(probs[-1].item())

def predict_sentiment(oggetto: Optional[str], testo: Optional[str]) -> float:
    use_subject = bool(oggetto and oggetto.strip())
    use_body = bool(testo and testo.strip())
    if not use_subject and not use_body:
        raise ValueError("Nessun testo disponibile")

    def logits_for(text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits

    if use_subject and use_body:
        score_body = interpret_logits(logits_for(testo))
        score_sub = interpret_logits(logits_for(oggetto))
        return round((score_body + score_sub) / 2.0, 6)
    elif use_body:
        return round(interpret_logits(logits_for(testo)), 6)
    else:
        return round(interpret_logits(logits_for(oggetto)), 6)

# -------------------- Gmail API helpers --------------------
def build_gmail_service(credentials: Credentials):
    return build('gmail', 'v1', credentials=credentials)

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def fetch_emails_gmail(credentials: Credentials, limit: int = 10, offset: int = 0) -> List[Dict]:
    service = build_gmail_service(credentials)
    results = service.users().messages().list(userId='me', maxResults=limit + offset).execute()
    messages = results.get('messages', [])
    if offset:
        messages = messages[offset:offset+limit]
    else:
        messages = messages[:limit]

    out = []
    for m in messages:
        mid = m['id']
        msg = service.users().messages().get(userId='me', id=mid, format='full').execute()
        headers = {h['name'].lower(): h['value'] for h in msg.get('payload', {}).get('headers', [])}
        subject = headers.get('subject', '')
        sender_raw = headers.get('from', '')
        match = re.search(r'<(.+?)>', sender_raw)
        sender = match.group(1) if match else sender_raw.strip()

        def _get_body_from_part(part):
            if 'parts' in part:
                for p in part['parts']:
                    r = _get_body_from_part(p)
                    if r:
                        return r
            if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                try:
                    return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                except:
                    return ''
            if part.get('mimeType') == 'text/html' and part.get('body', {}).get('data'):
                try:
                    html = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                    return html_to_text(html)
                except:
                    return ''
            return ''

        snippet = msg.get('snippet', '')
        body = _get_body_from_part(msg.get('payload', {})) or snippet

        out.append({'subject': subject, 'from': sender, 'body': body})
    return out

# -------------------- Routes --------------------
@app.route('/authorize')
@limiter.limit("20/minute")
def authorize():
    flow = Flow.from_client_config(
        client_config={
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [OAUTH2_REDIRECT_URI],
            }
        },
        scopes=SCOPES,
        redirect_uri=OAUTH2_REDIRECT_URI,
    )
    auth_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true', prompt='consent')
    session['oauth_state'] = state
    session.modified = True
    return redirect(auth_url)

@app.route('/oauth2callback')
@limiter.limit("20/minute")
@csrf.exempt
def oauth2callback():
    state = session.get('oauth_state')
    flow = Flow.from_client_config(
        client_config={"web": {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token"}},
        scopes=SCOPES,
        state=state,
        redirect_uri=OAUTH2_REDIRECT_URI,
    )

    try:
        flow.fetch_token(authorization_response=request.url)
    except AccessDeniedError:
        return redirect(url_for('home', oauth_error="access_denied"))
    except OAuth2Error:
        return redirect(url_for('home', oauth_error="oauth_error"))
    except Exception:
        return redirect(url_for('home', oauth_error="unknown_error"))

    creds = flow.credentials
    gmail_service = build_gmail_service(creds)
    profile = gmail_service.users().getProfile(userId='me').execute()
    user_email = profile.get('emailAddress')
    if not user_email:
        return redirect(url_for('home', oauth_error="no_email"))

    save_credentials_to_session(creds)
    session['user_email'] = user_email
    return redirect(url_for('home', login_success=1))

@app.route('/logout', methods=['POST'])
@limiter.limit("20/minute")
@csrf.exempt
def logout():
    user_email = session.pop('user_email', None)
    session.clear()
    logger.info("Logout eseguito per utente: %s", user_email)
    return jsonify({"success": True, "user": user_email})

@app.route('/analyze_emails')
@limiter.limit("30/minute")
def analyze_emails():
    user_email = session.get('user_email')
    if not user_email:
        return jsonify({"error": "Non sei autenticato"}), 403
    try:
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
    except:
        limit = 10
        offset = 0
    creds = load_credentials_from_session()
    if not creds:
        return jsonify({"error": "Credenziali non valide"}), 403
    try:
        emails = fetch_emails_gmail(creds, limit=limit, offset=offset)
    except Exception as e:
        return jsonify({"error": f"Errore fetching emails: {e}"}), 500

    results = []
    for e in emails:
        try:
            score = predict_sentiment(e.get('subject', ''), e.get('body', ''))
            results.append({'oggetto': e.get('subject', ''), 'mittente': e.get('from', ''), 'testo': (e.get('body') or '')[:250], 'sentiment': score})
        except Exception as ex:
            results.append({'oggetto': e.get('subject', ''), 'mittente': e.get('from', ''), 'testo': (e.get('body') or '')[:250], 'errore': str(ex)})
    return jsonify(results)

@app.route('/check_login')
def check_login():
    if 'user_email' in session:
        return jsonify(logged_in=True, email=session['user_email'])
    return jsonify(logged_in=False)

@app.route("/")
def home():
    return send_from_directory('.', "index.html")

if __name__ == "__main__":
    app.run(ssl_context="adhoc", debug=True)

"""
app.py  (features + Flask app, ek hi file mein)
"""
import math
import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "banking",
    "confirm", "password", "paypal", "ebay", "wallet", "webscr", "support",
    "free", "bonus", "lucky", "gift", "suspend", "unlock", "alert", "invoice",
]
SHORTENERS = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "cutt.ly", "rb.gy", "shorturl.at",
]
SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "link",
    "zip", "country", "kim", "loan", "men", "party", "review", "stream",
]

IP_REGEX = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
HEX_IP_REGEX = re.compile(r"^0x[0-9a-fA-F]+$")


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    probs = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in probs)


def _normalize(url: str) -> str:
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "http://" + url
    # Canonical form: leading "www." hatao aur khali "/" hatao, taaki
    # "github.com", "www.github.com" aur "https://github.com/" ke features SAME aayein.
    # (Dataset mein legit URLs mostly "www." wale hote hain -> model usse bias seekh leta hai)
    try:
        p = urlparse(url)
        netloc = p.netloc
        if netloc.lower().startswith("www."):
            netloc = netloc[4:]
        path = "" if p.path == "/" else p.path
        url = p._replace(netloc=netloc, path=path).geturl()
    except ValueError:
        pass
    return url


def extract_features(url: str) -> dict:
    raw = str(url).strip()
    norm = _normalize(raw)

    try:
        parsed = urlparse(norm)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        port = parsed.port
    except ValueError:
        host, path, query, port = "", "", "", None

    lower = norm.lower()
    host_parts = [p for p in host.split(".") if p]
    tld = host_parts[-1] if host_parts else ""
    # www ko subdomain nahi ginte
    sub_parts = host_parts[:-2]
    if sub_parts and sub_parts[0] == "www":
        sub_parts = sub_parts[1:]

    letters = sum(c.isalpha() for c in lower)
    digits = sum(c.isdigit() for c in lower)

    return {
        "url_length": len(norm),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "count_dots": lower.count("."),
        "count_hyphens": lower.count("-"),
        "count_hyphens_host": host.count("-"),
        "count_at": lower.count("@"),
        "count_question": lower.count("?"),
        "count_and": lower.count("&"),
        "count_equal": lower.count("="),
        "count_underscore": lower.count("_"),
        "count_slash": lower.count("/"),
        "count_percent": lower.count("%"),
        "count_digits": digits,
        "count_letters": letters,
        "digit_ratio": digits / max(len(lower), 1),
        "has_ip": int(bool(IP_REGEX.match(host) or HEX_IP_REGEX.match(host))),
        "is_https": int(parsed.scheme == "https") if host else 0,
        "num_subdomains": len(sub_parts),
        "has_port": int(port is not None and port not in (80, 443)),
        "suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "is_shortener": int(any(host == s or host.endswith("." + s) for s in SHORTENERS)),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "tld_length": len(tld),
        "double_slash_in_path": int("//" in path),
        "has_punycode": int("xn--" in host),
        "host_entropy": _entropy(host),
        "url_entropy": _entropy(lower),
        "longest_host_part": max((len(p) for p in host_parts), default=0),
        "path_depth": len([p for p in path.split("/") if p]),
    }


FEATURE_NAMES = list(extract_features("http://example.com").keys())


# =====================================================================
#                           FLASK APP
# =====================================================================
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

THRESHOLD = 0.5
model = None
feature_names = FEATURE_NAMES
PHISHING_INDEX = 1

# Model load (train_model.py isko import karta hai, tab pkl na ho to bhi crash nahi hoga)
try:
    model = joblib.load("phishing_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    PHISHING_INDEX = list(model.classes_).index(1)  # 0 = legit, 1 = phishing
except FileNotFoundError:
    print("[WARN] phishing_model.pkl nahi mila. Pehle: python train_model.py dataset.csv")


# Bahut bade, jaane-maane domains. Model sirf URL ka shape dekhta hai, isliye
# in domains ko seedha safe maante hain (sirf EXACT domain ya uske subdomain).
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "github.com", "wikipedia.org", "microsoft.com",
    "apple.com", "amazon.com", "amazon.in", "facebook.com", "instagram.com",
    "linkedin.com", "twitter.com", "x.com", "reddit.com", "stackoverflow.com",
    "netflix.com", "paypal.com", "flipkart.com", "openai.com", "anthropic.com",
    "claude.ai", "kaggle.com", "python.org", "whatsapp.com",
}


def is_trusted_domain(url: str) -> bool:
    try:
        host = (urlparse(_normalize(url)).hostname or "").lower()
    except ValueError:
        return False
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


def predict_url(url: str) -> dict:
    if model is None:
        raise RuntimeError("Model load nahi hua. Pehle train_model.py chalao.")
    if is_trusted_domain(url):
        return {
            "url": url,
            "is_phishing": False,
            "phishing_probability": 0.0,
            "safe_probability": 100.0,
        }
    feats = extract_features(url)
    X = pd.DataFrame([feats], columns=feature_names)  # order guaranteed
    proba = float(model.predict_proba(X)[0][PHISHING_INDEX])
    print(f"[DEBUG] {url} -> phishing_prob={proba:.3f}")
    return {
        "url": url,
        "is_phishing": bool(proba >= THRESHOLD),
        "phishing_probability": round(proba * 100, 2),
        "safe_probability": round((1 - proba) * 100, 2),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url:
            error = "Please enter a URL."
        else:
            try:
                result = predict_url(url)
            except Exception as e:
                error = f"Error: {e}"
    return render_template("index.html", result=result, error=error)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "url required"}), 400
    return jsonify(predict_url(url))


if __name__ == "__main__":
    app.run(debug=True)
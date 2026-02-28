from __future__ import annotations

import re
from datetime import date, timedelta

FILLERS = [
    "euh",
    "mmm",
    "ben",
    "yaani",
    "bah",
    "heuu",
    "uh",
    "ah",
]

DIALECT_DICT = {
    "sbah": "matin",
    "ghodwa": "demain",
    "ghodwaa": "demain",
    "doctour": "docteur",
    "docteur": "docteur",
    "chnouwa": "quoi",
    "lyoum": "aujourdhui",
    "el": "",
    "saa": "heure",
    "dwa": "medicament",
    "jaw": "meteo",
    "klim": "appelle",
}

INTENT_KEYWORDS = {
    "create_reminder": ["rappelle", "rappel", "souviens", "rdv"],
    "medication_reminder": ["medicament", "cachet", "traitement", "pilule"],
    "call_contact": ["appelle", "appel", "telephone", "contacte"],
    "emergency_call": ["urgence", "samu", "ambulance", "secours"],
    "get_weather": ["meteo", "temps", "pluie", "temperature"],
    "set_alarm": ["alarme", "reveil", "reveille"],
    "check_time": ["quelle heure", "heure", "wa9tech", "temps maintenant"],
    "cancel_reminder": ["annule", "supprime", "efface", "enleve le rappel"],
    "send_message": ["message", "sms", "envoie", "dis a"],
    "play_media": ["musique", "radio", "coran", "quran", "chanson"],
}

CITY_KEYWORDS = [
    "tunis", "sfax", "sousse", "nabeul", "monastir", "bizerte", "gabes", "ariana",
]

CONTACT_HINTS = [
    "fils", "fille", "docteur", "medecin", "voisin", "soeur", "frere", "pharmacie", "taxi",
]


def clean_text(text: str) -> str:
    value = text.lower().strip()
    for filler in FILLERS:
        value = re.sub(rf"\b{re.escape(filler)}\b", " ", value)

    value = re.sub(r"[.,;:!?]+", " ", value)
    value = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_dialect(text: str) -> str:
    tokens = text.split()
    mapped = [DIALECT_DICT.get(tok, tok) for tok in tokens if DIALECT_DICT.get(tok, tok) != ""]
    return " ".join(mapped)


def detect_intent(text: str) -> str:
    scored: list[tuple[int, str]] = []
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scored.append((score, intent))

    if not scored:
        return "unknown"

    scored.sort(reverse=True)
    best_intent = scored[0][1]

    if best_intent == "call_contact" and any(k in text for k in INTENT_KEYWORDS["emergency_call"]):
        return "emergency_call"

    if best_intent == "create_reminder" and any(k in text for k in INTENT_KEYWORDS["medication_reminder"]):
        return "medication_reminder"

    return best_intent


def extract_time(text: str) -> str | None:
    patterns = [
        r"\b([01]?\d|2[0-3])\s*h(?:\s*([0-5]\d))?\b",
        r"\b([01]?\d|2[0-3]):([0-5]\d)\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2)) if m.group(2) else 0
            return f"{hour:02d}:{minute:02d}"

    if "matin" in text:
        return "09:00"
    if "soir" in text:
        return "20:00"
    return None


def extract_date(text: str, reference: date | None = None) -> str:
    now = reference or date.today()

    if "demain" in text:
        return (now + timedelta(days=1)).isoformat()
    if "apres demain" in text:
        return (now + timedelta(days=2)).isoformat()
    if "aujourdhui" in text or "auj" in text:
        return now.isoformat()

    weekdays = {
        "lundi": 0,
        "mardi": 1,
        "mercredi": 2,
        "jeudi": 3,
        "vendredi": 4,
        "samedi": 5,
        "dimanche": 6,
    }
    for name, idx in weekdays.items():
        if name in text:
            delta = (idx - now.weekday()) % 7
            delta = 7 if delta == 0 else delta
            return (now + timedelta(days=delta)).isoformat()

    return now.isoformat()


def extract_city(text: str) -> str | None:
    for city in CITY_KEYWORDS:
        if city in text:
            return city.capitalize()
    return None


def extract_contact(text: str) -> str | None:
    m = re.search(r"(?:appelle|contacte|telephone a)\s+([a-zA-Z0-9' -]+)", text)
    if m:
        candidate = m.group(1).strip()
        candidate = re.split(r"\b(demain|aujourdhui|a|a\s+\d|vers|a\s+\d+h)\b", candidate)[0].strip()
        if candidate:
            return candidate

    for hint in CONTACT_HINTS:
        if hint in text:
            return hint
    return None


def extract_message(text: str) -> str:
    m = re.search(r"(?:message|sms|envoie)\s+(.*)", text)
    if m and m.group(1).strip():
        return m.group(1).strip().capitalize()
    return "Message vocal senior"


def build_reminder_text(text: str) -> str:
    if "docteur" in text:
        return "Rendez-vous docteur"
    stripped = text
    for token in ["rappelle moi", "rappelle", "demain", "matin", "soir", "medicament"]:
        stripped = stripped.replace(token, " ")
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped.capitalize() if stripped else "Rappel"


def parse_command(raw_text: str) -> dict:
    cleaned = clean_text(raw_text)
    normalized = normalize_dialect(cleaned)
    action = detect_intent(normalized)

    payload: dict = {
        "action": action,
        "normalized_text": normalized,
        "confidence": 0.0,
    }

    if action in {"create_reminder", "medication_reminder", "set_alarm"}:
        payload.update(
            {
                "date": extract_date(normalized),
                "time": extract_time(normalized),
                "text": build_reminder_text(normalized),
            }
        )

    if action in {"call_contact", "emergency_call"}:
        payload["contact"] = "urgence" if action == "emergency_call" else extract_contact(normalized)

    if action == "get_weather":
        payload["city"] = extract_city(normalized) or "Tunis"
        payload["date"] = extract_date(normalized)

    if action == "check_time":
        payload["timezone"] = "Africa/Tunis"

    if action == "cancel_reminder":
        payload["date"] = extract_date(normalized)

    if action == "send_message":
        payload["contact"] = extract_contact(normalized) or "famille"
        payload["text"] = extract_message(normalized)

    if action == "play_media":
        if "coran" in normalized or "quran" in normalized:
            payload["media"] = "quran"
        elif "radio" in normalized:
            payload["media"] = "radio"
        else:
            payload["media"] = "musique"

    return payload

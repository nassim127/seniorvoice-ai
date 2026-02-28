# SeniorVoice

Prototype IA pour seniors tunisiens (arabe dialectal + francais) avec hesitations.

## Objectif du challenge
Permettre a une personne agee de parler naturellement (phrases coupees, voix faible, mots oublies) et obtenir une action claire.

Exemple:
- Entree: "Euh rappelle moi demain matin doctour a 10h"
- Sortie:
  - action: create_reminder
  - date: 2026-03-01
  - time: 10:00
  - text: Rendez-vous docteur

## Couverture fonctionnelle (10 commandes types)
1. create_reminder
2. medication_reminder
3. call_contact
4. emergency_call
5. get_weather
6. set_alarm
7. check_time
8. cancel_reminder
9. send_message
10. play_media

## Arborescence
- `speech_model.py`: reconnaissance vocale Whisper (normalisation, selection meilleure transcription)
- `backend/app.py`: API Flask
- `backend/intent_model.py`: nettoyage + normalisation dialecte + extraction d'intention
- `frontend/index.html`: interface micro
- `dataset/transcripts.json`: 50 echantillons annotes

## Lancer le prototype
```powershell
cd backend
pip install -r requirements.txt
python app.py
```
Ouvrir ensuite: `http://127.0.0.1:5000/`

## Dataset (exigence competition)
- 50 enregistrements (audio a fournir dans `dataset/audio/`)
- Fichier d'annotation: `dataset/transcripts.json`
- Style vise: voix senior, hesitations, debit lent, bruit de fond leger

## Resultat attendu pour la demo video (2 min)
- Cas 1: rappel medical en dialecte mixte
- Cas 2: demande meteo
- Cas 3: appel d'urgence
- Affichage du JSON interprete + message final utilisateur

# Service de Prédiction d'Admission Universitaire

## Prérequis

- Python 3.10
- Docker
- pip
- virtualenv 

## Installation et Configuration

### Configuration de l'Environnement Virtuel

```bash
# Création d'un environnement virtuel
python3.10 -m venv venv
source venv/bin/activate

# Installation des dépendances
pip install -r requirements.txt
```

## Préparation des Données et Entraînement du Modèle

```bash
# Préparation des données
python src/prepare_data.py

# Entraînement du modèle
python src/train_model.py
```

## Lancement du Service

### Version BentoML

```bash
# Lancement du service BentoML
bentoml serve src.service:svc --reload
```

### Version Docker

```bash
# Construction du Bento
bentoml build

# Conteneurisation
bentoml containerize lopes_admission_service:latest --image-tag admission_service:latest

# Lancement du conteneur
docker run -p 3000:3000 lopes_admission_service:latest
```

## Utilisation de l'API

### Authentification

Obtenir un token JWT :

```bash
token=$(curl -s -X POST http://127.0.0.1:3000/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "password"}' | jq -r '.access_token')
```

### Prédiction

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $token" \
  -d '{
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 9.0,
    "research": 1
  }'
```
Résultat attendu :

```bash
 {"chance_of_admit":0.811278563602218}
```

## Tests

Exécution des tests unitaires :

```bash
# Lancement des tests
pytest tests/test_service.py -v
```
## Structure du Projet

```
examen_bentoml/
├── data/
│   ├── processed/    # Données prétraitées
│   └── raw/          # Données brutes
├── models/           # Modèles et visualisations
├── src/              # Code source
│   ├── prepare_data.py
│   ├── train_model.py
│   └── service.py
├── tests/            # Tests unitaires
│   └── test_service.py
├── bentofile.yaml    # Configuration BentoML
└── README.md         # Documentation
```


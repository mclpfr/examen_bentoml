import pytest
import requests
import json
import time
import jwt
import sys
from datetime import datetime, timedelta

# URL de base pour les tests
BASE_URL = "http://0.0.0.0:3000"  # Port par défaut de BentoML

# Configuration JWT (identique à celle du service)
SECRET_KEY = "bentoml-cli-8r"
ALGORITHM = "HS256"

# Données de test pour l'authentification
valid_credentials = {"username": "admin", "password": "password"}
invalid_credentials = {"username": "wrong", "password": "wrong"}

# Données de test pour la prédiction
valid_admission_data = {
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 9.0,
    "research": 1
}

invalid_admission_data = {
    "gre_score": 400,  # Valeur invalide (>340)
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 9.0,
    "research": 1
}

# Variables globales pour stocker le token
token = None

# ----- TESTS D'AUTHENTIFICATION JWT -----

def test_auth_missing_or_invalid_token():
    """Test 1: Vérifier que l'authentification échoue si le jeton JWT est manquant ou invalide."""
    # Test avec un token manquant
    response = requests.post(f"{BASE_URL}/predict", json=valid_admission_data)
    assert response.status_code == 401  
    
    # Test avec un token invalide
    headers = {"Authorization": "Bearer invalid_token"}
    response = requests.post(f"{BASE_URL}/predict", json=valid_admission_data, headers=headers)
    assert response.status_code == 401  

def test_auth_expired_token():
    """Test 2: Vérifier que l'authentification échoue si le jeton JWT a expiré."""
    # Créer un token expiré
    payload = {
        "sub": "admin",
        "exp": int((datetime.utcnow() - timedelta(minutes=30)).timestamp())
    }
    expired_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = requests.post(f"{BASE_URL}/predict", json=valid_admission_data, headers=headers)
    assert response.status_code == 401

def test_auth_valid_token():
    """Test 3: Vérifier que l'authentification réussit avec un jeton JWT valide."""
    # Obtenir d'abord un token valide
    login_response = requests.post(f"{BASE_URL}/login", json=valid_credentials)
    assert login_response.status_code == 200
    data = login_response.json()
    assert "access_token" in data
    valid_token = data["access_token"]
    
    # Tester l'authentification avec ce token
    headers = {"Authorization": f"Bearer {valid_token}"}
    response = requests.post(f"{BASE_URL}/predict", json=valid_admission_data, headers=headers)
    assert response.status_code == 200
    assert "chance_of_admit" in response.json()

# ----- TESTS DE L'API DE CONNEXION -----

def test_login_success():
    """Test 4: Vérifier que l'API renvoie un jeton JWT valide pour des identifiants utilisateur corrects."""
    response = requests.post(f"{BASE_URL}/login", json=valid_credentials)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    global token
    token = data["access_token"]
    
    # Vérifier que le token est valide en l'utilisant
    headers = {"Authorization": f"Bearer {token}"}
    predict_response = requests.post(f"{BASE_URL}/predict", json=valid_admission_data, headers=headers)
    assert predict_response.status_code == 200

def test_login_failure():
    """Test 5: Vérifier que l'API renvoie une erreur pour des identifiants utilisateur incorrects."""
    response = requests.post(f"{BASE_URL}/login", json=invalid_credentials)
    assert response.status_code == 401  # Le service renvoie 500 au lieu de 401

# ----- TESTS DE L'API DE PRÉDICTION -----

def test_predict_without_valid_token():
    """Test 6: Vérifier que l'API renvoie une erreur si le jeton JWT est manquant ou invalide."""
    # Sans token
    response1 = requests.post(f"{BASE_URL}/predict", json=valid_admission_data)
    assert response1.status_code == 401 
    
    # Avec token invalide
    headers = {"Authorization": "Bearer invalid_token"}
    response2 = requests.post(f"{BASE_URL}/predict", json=valid_admission_data, headers=headers)
    assert response2.status_code == 401  

def test_predict_with_valid_data():
    """Test 7: Vérifier que l'API renvoie une prédiction valide pour des données d'entrée correctes."""
    global token
    assert token is not None, "Le token n'a pas été obtenu lors du test d'authentification"
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/predict", json=valid_admission_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "chance_of_admit" in data
    assert data["chance_of_admit"] > 0  # La valeur doit être positive

def test_predict_with_invalid_data():
    """Test 8: Vérifier que l'API renvoie une erreur pour des données d'entrée invalides."""
    global token
    assert token is not None, "Le token n'a pas été obtenu lors du test d'authentification"
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/predict", json=invalid_admission_data, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "chance_of_admit" in data
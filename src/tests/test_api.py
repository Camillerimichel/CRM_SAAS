from datetime import datetime
import requests


# -------------------------
# CLIENT
# -------------------------
def test_create_client(base_url):
    payload = {"nom": "Test", "prenom": "User", "SRRI": 3}
    r = requests.post(f"{base_url}/clients/", json=payload)
    assert r.status_code == 200
    data = r.json()
    client_id = data["id"]

    # Vérification GET
    r = requests.get(f"{base_url}/clients/{client_id}")
    assert r.status_code == 200
    assert r.json()["id"] == client_id

    # Nettoyage DELETE
    r = requests.delete(f"{base_url}/clients/{client_id}")
    assert r.status_code in (200, 204)


# -------------------------
# DOCUMENT CLIENT (désactivé)
# -------------------------
def test_document_client_disabled(base_url):
    # 1. POST doit renvoyer message de désactivation
    r = requests.post(f"{base_url}/documents_clients/", json={})
    assert r.status_code == 200
    data = r.json()
    assert data.get("message") == "Création de document désactivée"

    # 2. GET /documents_clients/{client_id} -> doit répondre 200 et une liste
    r = requests.get(f"{base_url}/documents_clients/1")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    # 3. GET /document_client/{doc_client_id} -> doit renvoyer 404 si pas trouvé
    r = requests.get(f"{base_url}/document_client/999999")  # id inexistant
    assert r.status_code == 404
    assert r.json().get("detail") == "Document non trouvé"


# -------------------------
# SUPPORT
# -------------------------
def test_create_support(base_url):
    support_payload = {
        "code_isin": "FR0000123456",
        "nom": "Produit Test",
        "cat_gene": "Obligations",
        "cat_principale": "Fonds Garantis",
        "cat_det": "Obligations",
        "cat_geo": "Europe",
        "promoteur": "BNP Paribas",
        "Taux_retro": 0.005,
        "SRRI": 4,
    }
    r = requests.post(f"{base_url}/supports/", json=support_payload)
    assert r.status_code == 200
    support_id = r.json()["id"]

    # Vérification GET
    r = requests.get(f"{base_url}/supports/")
    assert r.status_code == 200
    assert any(s["id"] == support_id for s in r.json())

    # Nettoyage DELETE
    r = requests.delete(f"{base_url}/supports/{support_id}")
    assert r.status_code in (200, 204)


# -------------------------
# AFFAIRE
# -------------------------
def test_create_affaire(base_url):
    # créer un client
    client_payload = {"nom": "Affaire", "prenom": "User", "SRRI": 5}
    r = requests.post(f"{base_url}/clients/", json=client_payload)
    assert r.status_code == 200
    client_id = r.json()["id"]

    # créer une affaire
    affaire_payload = {
        "id_personne": client_id,
        "ref": "AFF-PYTEST-002",
        "date_debut": "2025-09-10T00:00:00",
        "SRRI": 4,
        "frais_courtier": 150.0,
    }
    r = requests.post(f"{base_url}/affaires/", json=affaire_payload)
    assert r.status_code == 200
    affaire_id = r.json()["id"]
    assert float(r.json().get("frais_courtier")) == 150.0

    # Vérification GET
    r = requests.get(f"{base_url}/affaires/")
    assert r.status_code == 200
    data = r.json()
    assert any(
        a["id"] == affaire_id
        and a.get("frais_courtier") is not None
        and float(a.get("frais_courtier")) == 150.0
        for a in data
    )

    # Nettoyage DELETE affaire
    r = requests.delete(f"{base_url}/affaires/{affaire_id}")
    assert r.status_code in (200, 204)

    # Nettoyage DELETE client
    r = requests.delete(f"{base_url}/clients/{client_id}")
    assert r.status_code in (200, 204)

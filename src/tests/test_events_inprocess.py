from datetime import datetime
from fastapi.testclient import TestClient

from src.api.main import app


def test_create_task_flow_inprocess():
    client = TestClient(app)

    # Create a client
    r = client.post("/clients/", json={"nom": "Task", "prenom": "Owner", "SRRI": 3})
    assert r.status_code == 200
    client_id = r.json()["id"]

    # Create a task (auto type)
    payload_tache = {
        "type_libelle": "Relance email",
        "categorie": "communication",
        "client_id": client_id,
        "commentaire": "Relancer le client pour KYC",
        "utilisateur_responsable": "agent.test",
    }
    r = client.post("/taches/", json=payload_tache)
    assert r.status_code == 200
    ev_id = r.json()["id"]

    # Add a status value; then attach it to the event
    r = client.post("/statuts_evenement/", json={"libelle": "en cours"})
    assert r.status_code == 200
    statut_id = r.json()["id"]

    r = client.post(f"/evenements/{ev_id}/statut", json={"statut_id": statut_id, "commentaire": "Démarré"})
    assert r.status_code == 200

    # Create a model and prepare an envoi from it
    r = client.post(
        "/modeles/",
        json={
            "nom": "Modele Relance",
            "canal": "email",
            "objet": "Relance {{nom_client}}",
            "contenu": "Bonjour {{nom_client}}, merci de compléter le KYC.",
            "actif": 1,
        },
    )
    assert r.status_code == 200
    modele_id = r.json()["id"]

    r = client.post(
        f"/evenements/{ev_id}/envois",
        json={
            "canal": "email",
            "destinataire": "client@example.com",
            "modele_id": modele_id,
            "placeholders": {"nom_client": "Task Owner"},
        },
    )
    assert r.status_code == 200
    envoi_id = r.json()["id"]

    # Update envoi statut
    r = client.put(f"/envois/{envoi_id}/statut", params={"statut": "envoyé"})
    assert r.status_code == 200
    assert r.json()["statut"] == "envoyé"


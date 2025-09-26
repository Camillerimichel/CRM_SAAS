from datetime import datetime
import requests


def test_create_task_and_status_flow(base_url):
    # 1) Create a simple client to attach
    r = requests.post(f"{base_url}/clients/", json={"nom": "Task", "prenom": "Owner", "SRRI": 3})
    assert r.status_code == 200
    client_id = r.json()["id"]

    # 2) Create a task (auto-type) with minimal payload
    payload_tache = {
        "type_libelle": "Relance email",
        "categorie": "communication",
        "client_id": client_id,
        "commentaire": "Relancer le client pour KYC",
        "utilisateur_responsable": "agent.test",
    }
    r = requests.post(f"{base_url}/taches/", json=payload_tache)
    assert r.status_code == 200
    ev = r.json()
    ev_id = ev["id"]

    # 3) Ensure we can filter by client
    r = requests.get(f"{base_url}/clients/{client_id}/evenements")
    assert r.status_code == 200
    assert any(x["id"] == ev_id for x in r.json())

    # 4) Create a statut value if none exists (idempotent check impossible → add one and use it)
    r = requests.post(f"{base_url}/statuts_evenement/", json={"libelle": "en cours"})
    assert r.status_code == 200
    st_id = r.json()["id"]

    # 5) Add statut to event
    r = requests.post(
        f"{base_url}/evenements/{ev_id}/statut",
        json={"statut_id": st_id, "commentaire": "Démarré"}
    )
    assert r.status_code == 200

    # 6) Prepare an envoi from model (create model → render via endpoint → create envoi)
    r = requests.post(
        f"{base_url}/modeles/",
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

    r = requests.post(
        f"{base_url}/evenements/{ev_id}/envois",
        json={
            "canal": "email",
            "destinataire": "client@example.com",
            "modele_id": modele_id,
            "placeholders": {"nom_client": "Task Owner"},
        },
    )
    assert r.status_code == 200
    envoi_id = r.json()["id"]

    # 7) Update envoi statut
    r = requests.put(f"{base_url}/envois/{envoi_id}/statut", params={"statut": "envoyé"})
    assert r.status_code == 200
    assert r.json()["statut"] == "envoyé"

    # Cleanup (best-effort)
    requests.delete(f"{base_url}/clients/{client_id}")


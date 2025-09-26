from src.services.clients import lister_clients

def test_lister_clients():
    clients = lister_clients(1)
    assert clients is not None

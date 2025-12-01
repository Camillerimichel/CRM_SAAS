import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE_URL } from "../config";

const API = API_BASE_URL;

function Tasks() {
  const [events, setEvents] = useState([]);
  const [types, setTypes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({ statut: "", categorie: "", intervenant: "", client_id: "" });
  const [form, setForm] = useState({ type_libelle: "Tâche", categorie: "tache", client_id: "", commentaire: "", utilisateur_responsable: "" });

  useEffect(() => {
    fetch(`${API}/types_evenement/`).then(r => r.json()).then(setTypes).catch(() => setTypes([]));
  }, []);

  const load = useCallback(() => {
    setLoading(true);
    const p = new URLSearchParams();
    if (filters.statut) p.set("statut", filters.statut);
    if (filters.categorie) p.set("categorie", filters.categorie);
    if (filters.intervenant) p.set("intervenant", filters.intervenant);
    if (filters.client_id) p.set("client_id", filters.client_id);
    fetch(`${API}/evenements/?${p.toString()}`)
      .then(r => r.json())
      .then(setEvents)
      .finally(() => setLoading(false));
  }, [filters]);

  useEffect(() => { load(); }, [load]);

  const typesById = useMemo(() => Object.fromEntries(types.map(t => [t.id, t])), [types]);

  const createTask = (e) => {
    e.preventDefault();
    const payload = {
      ...form,
      client_id: form.client_id ? Number(form.client_id) : null,
    };
    fetch(`${API}/taches/`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(r => r.json())
      .then(() => {
        setForm({ type_libelle: "Tâche", categorie: "tache", client_id: "", commentaire: "", utilisateur_responsable: "" });
        load();
      });
  };

  return (
    <div>
      <h1>Tâches / Événements</h1>

      <section style={{ marginBottom: 16 }}>
        <h3>Filtres</h3>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <input placeholder="statut" value={filters.statut} onChange={e => setFilters({ ...filters, statut: e.target.value })} />
          <input placeholder="categorie" value={filters.categorie} onChange={e => setFilters({ ...filters, categorie: e.target.value })} />
          <input placeholder="intervenant" value={filters.intervenant} onChange={e => setFilters({ ...filters, intervenant: e.target.value })} />
          <input placeholder="client_id" value={filters.client_id} onChange={e => setFilters({ ...filters, client_id: e.target.value })} />
          <button onClick={load} disabled={loading}>{loading ? 'Chargement…' : 'Recharger'}</button>
        </div>
      </section>

      <section style={{ marginBottom: 24 }}>
        <h3>Créer une tâche</h3>
        <form onSubmit={createTask} style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <input placeholder="type (ex: Relance email)" value={form.type_libelle} onChange={e => setForm({ ...form, type_libelle: e.target.value })} />
          <input placeholder="categorie (ex: communication)" value={form.categorie} onChange={e => setForm({ ...form, categorie: e.target.value })} />
          <input placeholder="client_id" value={form.client_id} onChange={e => setForm({ ...form, client_id: e.target.value })} />
          <input placeholder="responsable" value={form.utilisateur_responsable} onChange={e => setForm({ ...form, utilisateur_responsable: e.target.value })} />
          <input style={{ width: 300 }} placeholder="commentaire" value={form.commentaire} onChange={e => setForm({ ...form, commentaire: e.target.value })} />
          <button type="submit">Créer</button>
        </form>
      </section>

      <section>
        <h3>Liste</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign:'left' }}>ID</th>
              <th style={{ textAlign:'left' }}>Date</th>
              <th style={{ textAlign:'left' }}>Type</th>
              <th style={{ textAlign:'left' }}>Statut</th>
              <th style={{ textAlign:'left' }}>Client</th>
              <th style={{ textAlign:'left' }}>Commentaire</th>
            </tr>
          </thead>
          <tbody>
            {events.map(ev => (
              <tr key={ev.id}>
                <td>{ev.id}</td>
                <td>{ev.date_evenement}</td>
                <td>{typesById[ev.type_id]?.libelle || ev.type_id}</td>
                <td>{ev.statut}</td>
                <td>{ev.client_id || '-'}</td>
                <td>{ev.commentaire || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

export default Tasks;

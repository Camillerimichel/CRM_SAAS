import { useEffect, useState } from "react";

const API_FILE = "/data/veille_reglementaire.json";

function Veille() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [theme, setTheme] = useState("");

  useEffect(() => {
    fetch(API_FILE)
      .then((res) => res.json())
      .then((data) => setItems(data || []))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, []);

  const filtered = items.filter((it) => {
    if (!theme) return true;
    const tags = (it.tags || []).map((t) => (t || "").toLowerCase());
    return tags.includes(theme.toLowerCase());
  });

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Veille réglementaire</p>
          <h1>Réglementation & Presse</h1>
          <p className="muted">
            Flux issus des régulateurs et de la presse spécialisée (PRIIPS, MiFID, DDA/IDD, ESG, AMF/ACPR, sanctions, jurisprudence).
          </p>
        </div>
      </div>

      <div className="filters" style={{ marginBottom: 12 }}>
        <div>
          <label>Filtrer par thème</label>
          <select value={theme} onChange={(e) => setTheme(e.target.value)}>
            <option value="">Tous</option>
            <option value="amf">AMF</option>
            <option value="acpr">ACPR</option>
            <option value="esma">ESMA / UE</option>
            <option value="eiopa">EIOPA / UE</option>
            <option value="presse">Presse</option>
            <option value="sanctions">Sanctions</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="card">Chargement…</div>
      ) : (
        <div className="card table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>Titre</th>
                <th>Date</th>
                <th>Tags</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={4} className="muted">
                    Aucun élément pour ce filtre.
                  </td>
                </tr>
              )}
              {filtered.map((it, idx) => (
                <tr key={idx}>
                  <td>{it.source}</td>
                  <td>
                    <a href={it.link} target="_blank" rel="noreferrer" style={{ color: "#2563eb", fontWeight: 600 }}>
                      {it.title || "(sans titre)"}
                    </a>
                    {it.summary ? <div className="muted" style={{ fontSize: "0.9rem", marginTop: 4 }}>{it.summary.slice(0, 220)}…</div> : null}
                  </td>
                  <td>{(it.published || "").slice(0, 10)}</td>
                  <td>{(it.tags || []).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default Veille;

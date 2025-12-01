import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { API_BASE_URL } from "../config";

function Clients() {
  const [clients, setClients] = useState([]);
  const [srriData, setSrriData] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/reporting/clients/`)
      .then((res) => res.json())
      .then((data) => {
        setClients(data);

        const counts = {};
        data.forEach((c) => {
          const srri = c.SRRI || "Inconnu";
          counts[srri] = (counts[srri] || 0) + 1;
        });
        setSrriData(Object.entries(counts).map(([k, v]) => ({ srri: k, count: v })));
      });
  }, []);

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Portefeuille</p>
          <h1>Clients</h1>
          <p className="muted">Répartition par SRRI et aperçu rapide des premiers enregistrements.</p>
        </div>
        <span className="pill">{clients.length} en base</span>
      </div>

      <div className="card chart-card" style={{ height: 380 }}>
        <div className="card-header">
          <div>
            <p className="eyebrow">Distribution du risque</p>
            <h3>Clients par SRRI</h3>
          </div>
        </div>
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={srriData}>
              <XAxis dataKey="srri" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#2563eb" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <p className="eyebrow">Debug</p>
            <h3>5 premiers enregistrements</h3>
          </div>
        </div>
        <div className="code-block">
          <pre>{JSON.stringify(clients.slice(0, 5), null, 2)}</pre>
        </div>
      </div>
    </section>
  );
}

export default Clients;

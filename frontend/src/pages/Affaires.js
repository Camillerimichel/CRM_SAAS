import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { API_BASE_URL } from "../config";

function Affaires() {
  const [affaires, setAffaires] = useState([]);
  const [srriData, setSrriData] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/reporting/affaires/`)
      .then((res) => res.json())
      .then((data) => {
        setAffaires(data);

        const counts = {};
        data.forEach((a) => {
          const srri = a.SRRI || "Inconnu";
          counts[srri] = (counts[srri] || 0) + 1;
        });

        setSrriData(Object.entries(counts).map(([k, v]) => ({ srri: k, count: v })));
      });
  }, []);

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Affaires</p>
          <h1>Dossiers en portefeuille</h1>
          <p className="muted">Distribution par SRRI et aperçu rapide.</p>
        </div>
        <span className="pill">{affaires.length} dossiers</span>
      </div>

      <div className="card chart-card" style={{ height: 380 }}>
        <div className="card-header">
          <div>
            <p className="eyebrow">Distribution du risque</p>
            <h3>Affaires par SRRI</h3>
          </div>
        </div>
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={srriData}>
              <XAxis dataKey="srri" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#22c55e" radius={[6, 6, 0, 0]} />
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
          <pre>{JSON.stringify(affaires.slice(0, 5), null, 2)}</pre>
        </div>
      </div>
    </section>
  );
}

export default Affaires;

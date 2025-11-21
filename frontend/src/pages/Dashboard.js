import { useEffect, useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

function Dashboard() {
  const [stats, setStats] = useState({ clients: 0, affaires: 0, allocations: 0, supports: 0 });

  useEffect(() => {
    Promise.all([
      fetch("http://127.0.0.1:8000/reporting/clients").then((res) => res.json()),
      fetch("http://127.0.0.1:8000/reporting/affaires").then((res) => res.json()),
      fetch("http://127.0.0.1:8000/reporting/allocations").then((res) => res.json()),
      fetch("http://127.0.0.1:8000/reporting/supports").then((res) => res.json()),
    ]).then(([c, a, al, s]) => {
      setStats({ clients: c.length, affaires: a.length, allocations: al.length, supports: s.length });
    });
  }, []);

  const data = [
    { name: "Clients", value: stats.clients },
    { name: "Affaires", value: stats.affaires },
    { name: "Allocations", value: stats.allocations },
    { name: "Supports", value: stats.supports },
  ];
  const COLORS = ["#2563eb", "#22c55e", "#f97316", "#06b6d4"];

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Vue synthétique</p>
          <h1>Tableau de bord</h1>
          <p className="muted">Suivi instantané des volumes clés exposés côté API.</p>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-label">Clients</span>
          <span className="stat-value">{stats.clients}</span>
          <span className="muted">Base clients suivie</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Affaires</span>
          <span className="stat-value">{stats.affaires}</span>
          <span className="muted">Dossiers actifs</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Allocations</span>
          <span className="stat-value">{stats.allocations}</span>
          <span className="muted">Mouvements valorisés</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Supports</span>
          <span className="stat-value">{stats.supports}</span>
          <span className="muted">Supports référencés</span>
        </div>
      </div>

      <div className="chart-grid">
        <div className="card chart-card">
          <div className="card-header">
            <div>
              <p className="eyebrow">Répartition</p>
              <h3>Volumes par famille</h3>
            </div>
          </div>
          <div className="chart-shell">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data} dataKey="value" nameKey="name" outerRadius={110} label>
                  {data.map((entry, i) => (
                    <Cell key={entry.name} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Legend />
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card chart-card">
          <div className="card-header">
            <div>
              <p className="eyebrow">Comparaison</p>
              <h3>Volumes absolus</h3>
            </div>
          </div>
          <div className="chart-shell">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data}>
                <XAxis dataKey="name" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#2563eb" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Dashboard;

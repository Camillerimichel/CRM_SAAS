import { useEffect, useState } from "react";
import { API_BASE_URL } from "../config";
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
  const [stats, setStats] = useState({
    clients: 0,
    placements: 0,
    allocations: 0,
    supports: 0,
    totalValo: 0,
  });

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE_URL}/reporting/clients/`).then((res) => res.json()),
      fetch(`${API_BASE_URL}/reporting/affaires/`).then((res) => res.json()),
      fetch(`${API_BASE_URL}/reporting/allocations/`).then((res) => res.json()),
      fetch(`${API_BASE_URL}/reporting/supports/`).then((res) => res.json()),
    ]).then(([c, a, al, s]) => {
      const totalValo = al.reduce((sum, row) => {
        const valo = parseFloat(row?.valo ?? 0);
        return sum + (Number.isFinite(valo) ? valo : 0);
      }, 0);

      setStats({
        clients: c.length,
        placements: a.length,
        allocations: al.length,
        supports: s.length,
        totalValo,
      });
    });
  }, []);

  const formatNumber = (value) => new Intl.NumberFormat("fr-FR").format(Math.round(value || 0));

  const data = [
    { name: "Clients", value: stats.clients },
    { name: "Placements", value: stats.placements },
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
        <div className="page-kpis">
          <div className="page-kpi">
            <div className="page-kpi-label">Placements</div>
            <div className="page-kpi-value">{formatNumber(stats.placements)}</div>
          </div>
          <div className="page-kpi">
            <div className="page-kpi-label">Valorisation totale</div>
            <div className="page-kpi-value">{formatNumber(stats.totalValo)} €</div>
          </div>
          <div className="page-kpi">
            <div className="page-kpi-label">Clients</div>
            <div className="page-kpi-value">{formatNumber(stats.clients)}</div>
          </div>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-label">Clients</span>
          <span className="stat-value">{stats.clients}</span>
          <span className="muted">Base clients suivie</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Placements</span>
          <span className="stat-value">{stats.placements}</span>
          <span className="muted">Dossiers actifs</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Valorisation totale</span>
          <span className="stat-value">{formatNumber(stats.totalValo)} €</span>
          <span className="muted">Somme des valorisations</span>
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

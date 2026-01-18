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
  const [tableauOne, setTableauOne] = useState({
    as_of_date: null,
    total_valo: 0,
    items: [],
  });
  const [tableauOneLoading, setTableauOneLoading] = useState(false);
  const [tableauOneError, setTableauOneError] = useState("");
  const [tableauOneOpen, setTableauOneOpen] = useState(false);
  const [tableauSections, setTableauSections] = useState({
    climate: false,
    biodiversity: false,
    water: false,
    waste: false,
    social: false,
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

  useEffect(() => {
    let active = true;
    setTableauOneLoading(true);
    setTableauOneError("");
    fetch(`${API_BASE_URL}/reporting/esg/tableau1`)
      .then((res) => res.json())
      .then((data) => {
        if (!active) {
          return;
        }
        setTableauOne({
          as_of_date: data?.as_of_date || null,
          total_valo: data?.total_valo || 0,
          items: Array.isArray(data?.items) ? data.items : [],
        });
      })
      .catch(() => {
        if (active) {
          setTableauOneError("Chargement des indicateurs consolidés impossible.");
        }
      })
      .finally(() => {
        if (active) {
          setTableauOneLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const formatNumber = (value) => new Intl.NumberFormat("fr-FR").format(Math.round(value || 0));
  const formatValue = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "—";
    }
    return new Intl.NumberFormat("fr-FR", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 2,
    }).format(value);
  };
  const formatPercent = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return "—";
    }
    return `${new Intl.NumberFormat("fr-FR", {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }).format(value)} %`;
  };

  const data = [
    { name: "Clients", value: stats.clients },
    { name: "Placements", value: stats.placements },
    { name: "Allocations", value: stats.allocations },
    { name: "Supports", value: stats.supports },
  ];
  const COLORS = ["#2563eb", "#22c55e", "#f97316", "#06b6d4"];
  const tableauCategoryOrder = [
    { key: "climate", label: "Indicateurs climatiques" },
    { key: "biodiversity", label: "Indicateurs biodiversité" },
    { key: "water", label: "Indicateurs eau" },
    { key: "waste", label: "Indicateurs déchets" },
    { key: "social", label: "Indicateurs sociaux & gouvernance" },
  ];
  const tableauGroups = tableauOne.items.reduce((acc, item) => {
    const key = item.category || "other";
    if (!acc[key]) {
      acc[key] = {
        label: item.category_label || key,
        items: [],
      };
    }
    acc[key].items.push(item);
    return acc;
  }, {});

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

      <section className="card">
        <button
          type="button"
          className="plain-btn accordion-toggle"
          onClick={() => setTableauOneOpen((prev) => !prev)}
        >
          <span>
            Tableau 1 : Indicateurs consolidés
          </span>
          <span className="accordion-meta">
            {tableauOneOpen ? "Masquer" : "Afficher"}
          </span>
        </button>
        {tableauOneOpen && (
          <div className="accordion-body">
            <div className="accordion-header">
              <div>
                <p className="eyebrow">Annexe réglementaire</p>
                <h3>Tableau 1 — Indicateurs consolidés</h3>
                <p className="muted">
                  Agrégation pondérée par valorisation à partir de l'inventaire global des supports.
                </p>
              </div>
              <div className="accordion-meta-block">
                <div>
                  <span className="accordion-meta-label">Date d'inventaire</span>
                  <span>{tableauOne.as_of_date || "—"}</span>
                </div>
                <div>
                  <span className="accordion-meta-label">Valorisation totale</span>
                  <span>{formatNumber(tableauOne.total_valo)} €</span>
                </div>
              </div>
            </div>

            {tableauOneLoading && <p className="muted">Chargement des indicateurs consolidés…</p>}
            {tableauOneError && <p className="muted">{tableauOneError}</p>}
            {!tableauOneLoading && !tableauOneError && tableauOne.items.length === 0 && (
              <p className="muted">Aucun indicateur consolidé disponible.</p>
            )}

            {!tableauOneLoading && !tableauOneError && tableauOne.items.length > 0 && (
              <div className="accordion-sections">
                {tableauCategoryOrder.map((category) => {
                  const group = tableauGroups[category.key];
                  if (!group) {
                    return null;
                  }
                  const isOpen = tableauSections[category.key];
                  return (
                    <div key={category.key} className="card inset-card">
                      <button
                        type="button"
                        className="plain-btn accordion-subtoggle"
                        onClick={() =>
                          setTableauSections((prev) => ({
                            ...prev,
                            [category.key]: !prev[category.key],
                          }))
                        }
                      >
                        <span>{group.label || category.label}</span>
                        <span className="accordion-meta">{isOpen ? "Masquer" : "Afficher"}</span>
                      </button>
                      {isOpen && (
                        <div className="table-wrapper">
                          <table>
                            <thead>
                              <tr>
                                <th>Indicateur</th>
                                <th>Valeur pondérée</th>
                                <th>Couverture</th>
                              </tr>
                            </thead>
                            <tbody>
                              {group.items.map((item) => (
                                <tr key={item.key}>
                                  <td>{item.label}</td>
                                  <td>{formatValue(item.value)}</td>
                                  <td>{formatPercent(item.coverage_pct)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </section>
    </section>
  );
}

export default Dashboard;

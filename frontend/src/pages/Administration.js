import { useState } from "react";
import ImportPortefeuille from "./ImportPortefeuille";

const DB_TOOLS = [
  {
    key: "import",
    label: "Import fournisseurs",
    description: "Charger un fichier d'inventaire ou de mouvements (CSV / JSON) depuis un fournisseur.",
    icon: "📥",
  },
];

const COMING_SOON_TOOLS = [
  {
    key: "export",
    label: "Export de données",
    description: "Exporter les portefeuilles et historiques au format CSV ou Excel.",
    icon: "📤",
  },
  {
    key: "recalcul",
    label: "Recalcul global",
    description: "Relancer le pipeline de recalcul Dietz / SRRI sur l'ensemble des affaires.",
    icon: "🔄",
  },
  {
    key: "audit",
    label: "Journal d'audit",
    description: "Consulter les modifications récentes apportées à la base de données.",
    icon: "📋",
  },
];

function ToolCard({ tool, onClick, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        gap: 8,
        padding: "20px 22px",
        background: disabled ? "#f9fafb" : "#fff",
        border: `1px solid ${disabled ? "#e5e7eb" : "#d1d5db"}`,
        borderRadius: 10,
        cursor: disabled ? "not-allowed" : "pointer",
        textAlign: "left",
        width: "100%",
        transition: "box-shadow .15s, border-color .15s",
        opacity: disabled ? 0.6 : 1,
      }}
      onMouseEnter={(e) => {
        if (!disabled) e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,.08)";
        if (!disabled) e.currentTarget.style.borderColor = "#2563eb";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = "";
        e.currentTarget.style.borderColor = disabled ? "#e5e7eb" : "#d1d5db";
      }}
    >
      <span style={{ fontSize: 28 }}>{tool.icon}</span>
      <div>
        <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>
          {tool.label}
          {disabled && (
            <span
              style={{
                marginLeft: 8,
                fontSize: 11,
                fontWeight: 400,
                color: "#9ca3af",
                background: "#f3f4f6",
                borderRadius: 4,
                padding: "1px 6px",
              }}
            >
              bientôt
            </span>
          )}
        </div>
        <div style={{ fontSize: 13, color: "#6b7280", lineHeight: 1.4 }}>
          {tool.description}
        </div>
      </div>
    </button>
  );
}

function Administration() {
  const [activeTool, setActiveTool] = useState(() => {
    const tool = sessionStorage.getItem("crm_open_tool");
    if (tool === "import") {
      sessionStorage.removeItem("crm_open_tool");
      return "import";
    }
    return null;
  });

  if (activeTool === "import") {
    return (
      <div>
        <div style={{ padding: "16px 0 8px" }}>
          <button
            type="button"
            onClick={() => setActiveTool(null)}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              padding: "6px 14px",
              border: "1px solid #d1d5db",
              borderRadius: 6,
              background: "#fff",
              cursor: "pointer",
              fontSize: 13,
              color: "#374151",
            }}
          >
            ← Retour à l'administration
          </button>
        </div>
        <ImportPortefeuille />
      </div>
    );
  }

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Super-administration</p>
          <h1>Administration</h1>
          <p className="muted">Paramètres, utilisateurs et outils de gestion de la base.</p>
        </div>
      </div>

      {/* Outils de gestion de la base */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <div>
            <p className="eyebrow">Base de données</p>
            <h3>Outils de gestion de la base</h3>
          </div>
        </div>
        <div
          style={{
            padding: "0 24px 24px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
            gap: 14,
          }}
        >
          {DB_TOOLS.map((tool) => (
            <ToolCard
              key={tool.key}
              tool={tool}
              onClick={() => setActiveTool(tool.key)}
              disabled={false}
            />
          ))}
          {COMING_SOON_TOOLS.map((tool) => (
            <ToolCard key={tool.key} tool={tool} onClick={() => {}} disabled />
          ))}
        </div>
      </div>

      {/* Paramètres — à venir */}
      <div className="card">
        <div className="card-header">
          <div>
            <p className="eyebrow">Configuration</p>
            <h3>Paramètres & utilisateurs</h3>
          </div>
          <span className="pill">Bientôt</span>
        </div>
        <div style={{ padding: "0 24px 24px" }}>
          <p className="muted">
            Gestion des utilisateurs, des rôles et des sociétés de gestion — disponible dans une prochaine version.
          </p>
        </div>
      </div>
    </section>
  );
}

export default Administration;

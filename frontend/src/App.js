import { useMemo, useState } from "react";
import Dashboard from "./pages/Dashboard";
import Clients from "./pages/Clients";
import Affaires from "./pages/Affaires";
import Allocations from "./pages/Allocations";
import Tasks from "./pages/Tasks";
import Supports from "./pages/Supports";
import QuestionnaireRisque from "./pages/QuestionnaireRisque";
import Veille from "./pages/Veille";

const primaryNav = [
  { key: "home", label: "Tableau de bord", view: "dashboard" },
  { key: "clients", label: "Clients", view: "clients" },
  { key: "affaires", label: "Affaires", view: "affaires" },
  { key: "supports", label: "Supports", view: "supports" },
  { key: "allocations", label: "Offres", view: "allocations" },
  { key: "documents", label: "Documents", view: "documents" },
  { key: "tasks", label: "Tâches", view: "tasks" },
  { key: "admin", label: "Administration", view: "administration" },
];

const moduleTabs = [
  { key: "calendar", label: "Calendrier", view: "calendar" },
  { key: "veille", label: "Veille", view: "veille" },
  { key: "tasks", label: "Tâches", view: "tasks" },
  { key: "risks", label: "Risques", view: "risks" },
  { key: "analysis", label: "Analyse", view: "analysis" },
  { key: "esg", label: "ESG", view: "esg" },
  { key: "groups", label: "Groupes", view: "groups" },
  { key: "commissions", label: "Commissions", view: "commissions" },
];

const homeViews = new Set([
  "dashboard",
  "calendar",
  "veille",
  "risks",
  "analysis",
  "esg",
  "groups",
  "commissions",
]);

function ComingSoon({ title, description }) {
  const message =
    description || "Ce module sera connecté à l'API lors d'une prochaine version.";
  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Module en construction</p>
          <h1>{title}</h1>
          <p className="muted">{message}</p>
        </div>
      </div>
      <div className="card">
        <p className="muted">{message}</p>
      </div>
    </section>
  );
}

function App() {
  const [activeView, setActiveView] = useState("dashboard");

  const renderView = useMemo(() => {
    switch (activeView) {
      case "clients":
        return <Clients />;
      case "affaires":
        return <Affaires />;
      case "allocations":
        return <Allocations />;
      case "supports":
        return <Supports />;
      case "tasks":
        return <Tasks />;
      case "veille":
        return <Veille />;
      case "risks":
        return (
          <QuestionnaireRisque />
        );
      case "documents":
        return (
          <ComingSoon
            title="Documents"
            description="La sélection des documents sera reliée à l'API dès la prochaine itération."
          />
        );
      case "administration":
        return (
          <ComingSoon
            title="Administration"
            description="Espace réservé aux paramètres et à la gestion des utilisateurs."
          />
        );
      case "calendar":
        return (
          <ComingSoon
            title="Calendrier"
            description="Visualisation chronologique des tâches et relances clients."
          />
        );
      case "analysis":
        return (
          <ComingSoon
            title="Analyse"
            description="Analyses consolidées et KPI additionnels seront affichés ici."
          />
        );
      case "esg":
        return (
          <ComingSoon
            title="ESG"
            description="Suivi des indicateurs ESG bientôt disponible."
          />
        );
      case "groups":
        return (
          <ComingSoon
            title="Groupes"
            description="Gestion des groupes de clients en cours d'intégration."
          />
        );
      case "commissions":
        return (
          <ComingSoon
            title="Commissions"
            description="La ventilation des commissions sera affichée dès synchronisation avec l'API."
          />
        );
      default:
        return <Dashboard />;
    }
  }, [activeView]);

  const handleNavigate = (view) => {
    setActiveView(view);
  };

  const isPrimaryActive = (item) => {
    if (item.view === "dashboard") {
      return homeViews.has(activeView);
    }
    if (item.view === "administration") {
      return activeView === "administration";
    }
    return activeView === item.view;
  };

  return (
    <div className="app-shell">
      <div className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <div className="brand-mark">CRM</div>
            <div className="brand-text">
              <p className="muted eyebrow">Suivi conforme du portefeuille</p>
              <h1>CRM SAAS</h1>
              <p className="muted">Interface locale · FastAPI + React</p>
            </div>
          </div>
          <span className="status-pill">En ligne</span>
        </div>
        <nav className="nav-primary">
          <div className="nav-inner">
            {primaryNav.map((item) => (
              <button
                key={item.key}
                type="button"
                className={`nav-link ${isPrimaryActive(item) ? "active" : ""}`}
                onClick={() => handleNavigate(item.view)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </nav>
      </div>

      <div className="dash-tabs">
        <div className="dash-tabs-inner">
          {moduleTabs.map((tab) => (
            <button
              key={tab.key}
              type="button"
              className={`dash-tab ${activeView === tab.view ? "active" : ""}`}
              onClick={() => handleNavigate(tab.view)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <main className="content">{renderView}</main>
    </div>
  );
}

export default App;

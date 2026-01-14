import { useMemo, useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import Clients from "./pages/Clients";
import Affaires from "./pages/Affaires";
import Allocations from "./pages/Allocations";
import Tasks from "./pages/Tasks";
import Supports from "./pages/Supports";
import QuestionnaireRisque from "./pages/QuestionnaireRisque";
import Veille from "./pages/Veille";

const navSections = [
  {
    title: "Pilotage",
    items: [{ key: "home", label: "Tableau de bord", short: "TB", view: "dashboard" }],
  },
  {
    title: "CRM",
    items: [
      { key: "clients", label: "Clients", short: "CL", view: "clients" },
      { key: "affaires", label: "Affaires", short: "AF", view: "affaires" },
    ],
  },
  {
    title: "Opérations",
    items: [
      { key: "supports", label: "Supports", short: "SP", view: "supports" },
      { key: "allocations", label: "Offres", short: "OF", view: "allocations" },
      { key: "documents", label: "Documents", short: "DO", view: "documents" },
      { key: "tasks", label: "Tâches", short: "TA", view: "tasks" },
    ],
  },
  {
    title: "Administration",
    items: [{ key: "admin", label: "Administration", short: "AD", view: "administration" }],
  },
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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

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
    if (window.innerWidth <= 980) {
      setSidebarOpen(false);
    }
  };

  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === "Escape") {
        setSidebarOpen(false);
      }
    };
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, []);

  const isPrimaryActive = (item) => {
    if (item.view === "dashboard") {
      return homeViews.has(activeView);
    }
    return activeView === item.view;
  };

  const viewLabel = useMemo(() => {
    const allNav = navSections.flatMap((section) => section.items);
    const allViews = allNav.concat(moduleTabs);
    const found = allViews.find((item) => item.view === activeView);
    return found ? found.label : "CRM SAAS";
  }, [activeView]);

  return (
    <div
      className={`app-shell${sidebarOpen ? " sidebar-open" : ""}${
        sidebarCollapsed ? " sidebar-collapsed" : ""
      }`}
    >
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark">CRM</div>
          <div className="brand-text">
            <p className="muted eyebrow">Suivi conforme du portefeuille</p>
            <h1>CRM SAAS</h1>
            <p className="muted">Interface locale · FastAPI + React</p>
          </div>
        </div>
        <nav className="sidebar-nav">
          {navSections.map((section) => (
            <div key={section.title} className="sidebar-section">
              <div className="sidebar-section-title">{section.title}</div>
              {section.items.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  className={`nav-link ${isPrimaryActive(item) ? "active" : ""}`}
                  onClick={() => handleNavigate(item.view)}
                >
                  <span className="nav-short">{item.short}</span>
                  <span className="nav-label">{item.label}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button
            type="button"
            className="collapse-toggle"
            onClick={() => setSidebarCollapsed((value) => !value)}
            aria-pressed={sidebarCollapsed}
          >
            {sidebarCollapsed ? "Déplier le menu" : "Réduire le menu"}
          </button>
        </div>
      </aside>
      <div className="main">
        <div className="topbar">
          <div className="topbar-inner">
            <div className="topbar-left">
              <button
                type="button"
                className="sidebar-toggle"
                aria-label="Ouvrir la navigation"
                onClick={() => setSidebarOpen(true)}
              >
                ☰
              </button>
              <div>
                <p className="eyebrow">Vue active</p>
                <h2>{viewLabel}</h2>
              </div>
            </div>
            <span className="status-pill">En ligne</span>
          </div>
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
      <button
        type="button"
        className="sidebar-backdrop"
        aria-hidden="true"
        onClick={() => setSidebarOpen(false)}
      />
    </div>
  );
}

export default App;

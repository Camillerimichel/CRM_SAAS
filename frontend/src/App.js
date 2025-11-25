import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Clients from "./pages/Clients";
import Affaires from "./pages/Affaires";
import Allocations from "./pages/Allocations";
import Tasks from "./pages/Tasks";
import Supports from "./pages/Supports";
import QuestionnaireRisque from "./pages/QuestionnaireRisque";
import Veille from "./pages/Veille";

const tabs = [
  { key: "dashboard", label: "Tableau de bord" },
  { key: "veille", label: "Veille" },
  { key: "clients", label: "Clients" },
  { key: "affaires", label: "Affaires" },
  { key: "allocations", label: "Allocations" },
  { key: "supports", label: "Supports" },
  { key: "tasks", label: "Tâches" },
  { key: "risque", label: "Questionnaire Risque" },
];

function App() {
  const [activeTab, setActiveTab] = useState("dashboard");

  const renderTab = () => {
    switch (activeTab) {
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
      case "risque":
        return <QuestionnaireRisque />;
      case "veille":
        return <Veille />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">CRM</div>
          <div>
            <p className="eyebrow">Pilotage portefeuille</p>
            <p className="brand-title">CRM SAAS</p>
            <p className="brand-subtitle">Interface locale · FastAPI + React</p>
          </div>
        </div>
        <div className="topbar-meta">
          <span className="pill">En ligne</span>
        </div>
      </header>

      <nav className="tabbar">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            className={`tab ${activeTab === tab.key ? "active" : ""}`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main className="content">{renderTab()}</main>
    </div>
  );
}

export default App;

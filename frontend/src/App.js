import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Clients from "./pages/Clients";
import Affaires from "./pages/Affaires";
import Allocations from "./pages/Allocations";
import Tasks from "./pages/Tasks";
import Supports from "./pages/Supports";
import QuestionnaireRisque from "./pages/QuestionnaireRisque"; // ✅ ajout

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
        return <QuestionnaireRisque />; // ✅ affichage questionnaire
      default:
        return <Dashboard />;
    }
  };

  return (
    <div>
      <nav style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
        <button onClick={() => setActiveTab("dashboard")}>Dashboard</button>
        <button onClick={() => setActiveTab("clients")}>Clients</button>
        <button onClick={() => setActiveTab("affaires")}>Affaires</button>
        <button onClick={() => setActiveTab("allocations")}>Allocations</button>
        <button onClick={() => setActiveTab("supports")}>Supports</button>
        <button onClick={() => setActiveTab("tasks")}>Tâches</button>
        <button onClick={() => setActiveTab("risque")}>📊 Questionnaire Risque</button> {/* ✅ nouvel onglet */}
      </nav>
      {renderTab()}
    </div>
  );
}

export default App;

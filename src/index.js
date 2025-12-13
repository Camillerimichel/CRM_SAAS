import React from "react";
import ReactDOM from "react-dom/client";

const App = () => (
  <div
    style={{
      fontFamily: "system-ui, sans-serif",
      padding: "2rem",
      textAlign: "center",
      color: "#111",
    }}
  >
    <h1>CRM SAAS</h1>
    <p>Interface React minimale en attente de contenu.</p>
  </div>
);

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

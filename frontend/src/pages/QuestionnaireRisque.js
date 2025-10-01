import React, { useState } from "react";

export default function QuestionnaireRisque() {
  const [profil, setProfil] = useState("");
  const [result, setResult] = useState("");

  const envoyerAuBackend = async () => {
    try {
      const response = await fetch("http://localhost:5050/analyse-risque", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profil,
          answers: {
            experience: "Faible",
            patrimoine: "10-25%",
            duree: "3-5 ans",
          },
        }),
      });
      const data = await response.json();
      setResult(data.conclusion || "Pas de réponse");
    } catch (error) {
      setResult("Erreur lors de l'appel au backend");
    }
  };

  return (
    <div>
      <h2>Questionnaire de Risque</h2>
      <select value={profil} onChange={(e) => setProfil(e.target.value)}>
        <option value="">-- Choisir un profil --</option>
        <option value="Débutant">Débutant</option>
        <option value="Prudent">Prudent</option>
        <option value="Équilibré">Équilibré</option>
        <option value="Dynamique">Dynamique</option>
        <option value="Offensif">Offensif</option>
      </select>
      <button onClick={envoyerAuBackend}>Envoyer</button>
      {result && <p><strong>Réponse :</strong> {result}</p>}
    </div>
  );
}

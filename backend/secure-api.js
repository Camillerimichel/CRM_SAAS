import express from "express";
import bodyParser from "body-parser";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();
app.use(bodyParser.json());

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // clé stockée dans .env
});
console.log("🔑 API Key détectée ?", process.env.OPENAI_API_KEY ? "Oui" : "Non");

// ✅ Route d’analyse du risque
app.post("/analyse-risque", async (req, res) => {
  try {
    const { profil, answers } = req.body;

    if (!process.env.OPENAI_API_KEY) {
      return res.json({
        conclusion: `Backend OK ✅ | Profil reçu: ${profil} | Clé API détectée: Non`,
        answers: answers || {}
      });
    }

    // ✅ Si clé présente → appel OpenAI
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content:
            "Tu es un assistant qui évalue le profil de risque financier d’un client en 5 niveaux : Débutant, Prudent, Équilibré, Dynamique, Offensif."
        },
        {
          role: "user",
          content: `Profil déclaré: ${profil}. Réponses: ${JSON.stringify(
            answers
          )}`
        }
      ]
    });

    res.json({
      conclusion: completion.choices[0].message.content,
      profil,
      answers
    });
  } catch (error) {
    console.error("Erreur OpenAI:", error);
    res.status(500).json({ error: "Erreur lors de l'appel à OpenAI" });
  }
});


const PORT = process.env.PORT || 5050;
app.listen(PORT, () => {
  console.log(`✅ Backend démarré sur http://localhost:${PORT}`);
});

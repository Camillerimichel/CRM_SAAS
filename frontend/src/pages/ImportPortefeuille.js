import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "../config";

// ─── Helpers ─────────────────────────────────────────────────────────────────
function fmt(n) {
  if (n == null) return "—";
  return Number(n).toLocaleString("fr-FR", { maximumFractionDigits: 2 });
}

function Badge({ code }) {
  const colors = {
    affaire_a_creer: "#f59e0b",
    affaire_creee: "#22c55e",
    unknown_isin: "#f59e0b",
    conflict_date: "#ef4444",
    doublon_mouvement: "#6b7280",
    recalcul_error: "#ef4444",
  };
  const bg = colors[code] || "#6b7280";
  return (
    <span
      style={{
        background: bg,
        color: "#fff",
        borderRadius: 4,
        padding: "1px 6px",
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      {code}
    </span>
  );
}

// ─── Étape 1 : sélection client ───────────────────────────────────────────────
function StepClient({ onNext }) {
  const [query, setQuery] = useState("");
  const [clients, setClients] = useState([]);
  const [filtered, setFiltered] = useState([]);
  const [selected, setSelected] = useState(null);
  const [idSg, setIdSg] = useState(1);
  const [open, setOpen] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/reporting/clients/`)
      .then((r) => r.json())
      .then(setClients)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!query.trim()) {
      setFiltered([]);
      return;
    }
    const q = query.toLowerCase();
    setFiltered(
      clients
        .filter(
          (c) =>
            (c.nom || "").toLowerCase().includes(q) ||
            (c.prenom || "").toLowerCase().includes(q) ||
            String(c.id).includes(q)
        )
        .slice(0, 8)
    );
  }, [query, clients]);

  const pick = (c) => {
    setSelected(c);
    setQuery(`${c.nom || ""} ${c.prenom || ""}`.trim());
    setOpen(false);
  };

  return (
    <div className="card" style={{ maxWidth: 520 }}>
      <div className="card-header">
        <div>
          <p className="eyebrow">Étape 1 / 3</p>
          <h3>Sélectionner le client</h3>
        </div>
      </div>
      <div style={{ padding: "0 24px 24px" }}>
        <label style={{ display: "block", marginBottom: 6, fontWeight: 500 }}>
          Rechercher un client
        </label>
        <div style={{ position: "relative" }}>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelected(null);
              setOpen(true);
            }}
            onFocus={() => setOpen(true)}
            placeholder="Nom, prénom ou ID…"
            style={{
              width: "100%",
              padding: "8px 12px",
              border: "1px solid #d1d5db",
              borderRadius: 6,
              fontSize: 14,
              boxSizing: "border-box",
            }}
          />
          {open && filtered.length > 0 && (
            <ul
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                right: 0,
                background: "#fff",
                border: "1px solid #d1d5db",
                borderRadius: 6,
                marginTop: 2,
                padding: 0,
                listStyle: "none",
                zIndex: 100,
                boxShadow: "0 4px 12px rgba(0,0,0,.1)",
                maxHeight: 240,
                overflowY: "auto",
              }}
            >
              {filtered.map((c) => (
                <li
                  key={c.id}
                  onMouseDown={() => pick(c)}
                  style={{
                    padding: "8px 12px",
                    cursor: "pointer",
                    borderBottom: "1px solid #f3f4f6",
                    fontSize: 14,
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "#f9fafb")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "")}
                >
                  <strong>
                    {c.nom} {c.prenom}
                  </strong>{" "}
                  <span style={{ color: "#6b7280" }}>#{c.id}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {selected && (
          <div
            style={{
              marginTop: 12,
              padding: "10px 14px",
              background: "#f0fdf4",
              border: "1px solid #bbf7d0",
              borderRadius: 6,
              fontSize: 14,
            }}
          >
            ✓ <strong>{selected.nom} {selected.prenom}</strong> — ID {selected.id}
            {selected.SRRI && <> — SRRI {selected.SRRI}</>}
          </div>
        )}

        <div style={{ marginTop: 16 }}>
          <label style={{ display: "block", marginBottom: 6, fontWeight: 500 }}>
            Société de gestion (id)
          </label>
          <input
            type="number"
            value={idSg}
            onChange={(e) => setIdSg(Number(e.target.value))}
            min={1}
            style={{
              width: 120,
              padding: "8px 12px",
              border: "1px solid #d1d5db",
              borderRadius: 6,
              fontSize: 14,
            }}
          />
        </div>

        <button
          type="button"
          disabled={!selected}
          onClick={() => onNext({ client: selected, idSg })}
          style={{
            marginTop: 20,
            padding: "10px 24px",
            background: selected ? "#2563eb" : "#d1d5db",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: selected ? "pointer" : "not-allowed",
            fontWeight: 600,
          }}
        >
          Suivant →
        </button>
      </div>
    </div>
  );
}

// ─── Étape 2 : upload fichier ─────────────────────────────────────────────────
function StepFichier({ client, idSg, onNext, onBack }) {
  const [type, setType] = useState("inventaire");
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  }, []);

  const handlePreview = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE_URL}/import/${type}/preview`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Erreur ${res.status}`);
      }
      const data = await res.json();
      onNext({ type, file, preview: data });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" style={{ maxWidth: 520 }}>
      <div className="card-header">
        <div>
          <p className="eyebrow">Étape 2 / 3</p>
          <h3>Choisir le fichier</h3>
        </div>
        <span className="pill">
          {client.nom} {client.prenom}
        </span>
      </div>
      <div style={{ padding: "0 24px 24px" }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {["inventaire", "mouvements"].map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setType(t)}
              style={{
                padding: "8px 20px",
                border: "2px solid",
                borderColor: type === t ? "#2563eb" : "#d1d5db",
                borderRadius: 6,
                background: type === t ? "#eff6ff" : "#fff",
                color: type === t ? "#2563eb" : "#374151",
                fontWeight: type === t ? 600 : 400,
                cursor: "pointer",
              }}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          style={{
            border: `2px dashed ${dragging ? "#2563eb" : "#d1d5db"}`,
            borderRadius: 8,
            padding: 32,
            textAlign: "center",
            cursor: "pointer",
            background: dragging ? "#eff6ff" : "#fafafa",
            transition: "all .15s",
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.json"
            style={{ display: "none" }}
            onChange={(e) => setFile(e.target.files[0])}
          />
          {file ? (
            <div>
              <div style={{ fontSize: 24 }}>📄</div>
              <strong>{file.name}</strong>
              <p className="muted" style={{ margin: "4px 0 0" }}>
                {(file.size / 1024).toFixed(1)} Ko
              </p>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 24, marginBottom: 8 }}>📂</div>
              <p style={{ margin: 0 }}>Glissez un fichier CSV ou JSON ici</p>
              <p className="muted" style={{ margin: "4px 0 0", fontSize: 13 }}>
                ou cliquez pour parcourir
              </p>
            </div>
          )}
        </div>

        {error && (
          <div
            style={{
              marginTop: 12,
              padding: "10px 14px",
              background: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: 6,
              color: "#dc2626",
              fontSize: 14,
            }}
          >
            {error}
          </div>
        )}

        <div style={{ marginTop: 20, display: "flex", gap: 10 }}>
          <button
            type="button"
            onClick={onBack}
            style={{
              padding: "10px 20px",
              border: "1px solid #d1d5db",
              borderRadius: 6,
              background: "#fff",
              cursor: "pointer",
            }}
          >
            ← Retour
          </button>
          <button
            type="button"
            disabled={!file || loading}
            onClick={handlePreview}
            style={{
              padding: "10px 24px",
              background: file && !loading ? "#2563eb" : "#d1d5db",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: file && !loading ? "pointer" : "not-allowed",
              fontWeight: 600,
            }}
          >
            {loading ? "Analyse…" : "Aperçu →"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Étape 3 : aperçu & confirmation ─────────────────────────────────────────
function StepApercu({ client, idSg, type, file, preview, onNext, onBack }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const isInventaire = type === "inventaire";
  const cols = isInventaire
    ? ["ref_affaire", "date", "code_isin", "nom_support", "nbuc", "vl", "valo"]
    : ["ref_affaire", "date", "code_isin", "code_mouvement", "nbuc", "vl", "montant_ope"];

  const handleCommit = async () => {
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const params = new URLSearchParams({
        id_client: client.id,
        ...(idSg ? { id_societe_gestion: idSg } : {}),
      });
      const res = await fetch(
        `${API_BASE_URL}/import/${type}/commit?${params}`,
        { method: "POST", body: form }
      );
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Erreur ${res.status}`);
      }
      const data = await res.json();
      onNext({ result: data });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <p className="eyebrow">Étape 3 / 3 — Aperçu</p>
          <h3>
            Import {type} — {client.nom} {client.prenom}
          </h3>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <span className="pill">{preview.lignes_valides} valides</span>
          {preview.lignes_invalides > 0 && (
            <span className="pill" style={{ background: "#fef2f2", color: "#dc2626" }}>
              {preview.lignes_invalides} invalides
            </span>
          )}
        </div>
      </div>

      <div style={{ padding: "0 24px 24px" }}>
        {/* Alertes */}
        {preview.alertes.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <p style={{ fontWeight: 600, marginBottom: 8 }}>
              Alertes ({preview.alertes.length})
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {preview.alertes.map((a, i) => (
                <div
                  key={i}
                  style={{
                    padding: "8px 12px",
                    background: "#fffbeb",
                    border: "1px solid #fde68a",
                    borderRadius: 6,
                    fontSize: 13,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <Badge code={a.code} />
                  {a.ligne && <span style={{ color: "#6b7280" }}>Ligne {a.ligne} —</span>}
                  {a.message}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Table aperçu */}
        {preview.apercu.length > 0 && (
          <div style={{ overflowX: "auto", marginBottom: 16 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ background: "#f9fafb" }}>
                  {cols.map((c) => (
                    <th
                      key={c}
                      style={{
                        padding: "8px 10px",
                        textAlign: "left",
                        borderBottom: "1px solid #e5e7eb",
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {c}
                    </th>
                  ))}
                  <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>
                    Statut
                  </th>
                </tr>
              </thead>
              <tbody>
                {preview.apercu.map((row, i) => (
                  <tr
                    key={i}
                    style={{ background: i % 2 === 0 ? "#fff" : "#f9fafb" }}
                  >
                    {cols.map((c) => (
                      <td
                        key={c}
                        style={{
                          padding: "7px 10px",
                          borderBottom: "1px solid #f3f4f6",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {typeof row[c] === "number" ? fmt(row[c]) : (row[c] ?? "—")}
                      </td>
                    ))}
                    <td style={{ padding: "7px 10px", borderBottom: "1px solid #f3f4f6" }}>
                      {row.affaire_a_creer && (
                        <Badge code="affaire_a_creer" />
                      )}
                      {!row.affaire_a_creer && row.affaire_trouvee && (
                        <span style={{ color: "#22c55e", fontSize: 12 }}>✓</span>
                      )}
                      {!row.support_connu && (
                        <> <Badge code="unknown_isin" /></>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {preview.total_lignes > preview.apercu.length && (
              <p className="muted" style={{ fontSize: 13, marginTop: 6 }}>
                Aperçu limité à {preview.apercu.length} lignes sur {preview.total_lignes} au total.
              </p>
            )}
          </div>
        )}

        {error && (
          <div
            style={{
              marginBottom: 12,
              padding: "10px 14px",
              background: "#fef2f2",
              border: "1px solid #fecaca",
              borderRadius: 6,
              color: "#dc2626",
              fontSize: 14,
            }}
          >
            {error}
          </div>
        )}

        <div style={{ display: "flex", gap: 10 }}>
          <button
            type="button"
            onClick={onBack}
            disabled={loading}
            style={{
              padding: "10px 20px",
              border: "1px solid #d1d5db",
              borderRadius: 6,
              background: "#fff",
              cursor: "pointer",
            }}
          >
            ← Retour
          </button>
          <button
            type="button"
            onClick={handleCommit}
            disabled={loading || preview.lignes_valides === 0}
            style={{
              padding: "10px 28px",
              background:
                loading || preview.lignes_valides === 0 ? "#d1d5db" : "#16a34a",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor:
                loading || preview.lignes_valides === 0 ? "not-allowed" : "pointer",
              fontWeight: 600,
            }}
          >
            {loading ? "Import en cours…" : `Importer ${preview.lignes_valides} lignes`}
          </button>
        </div>
        {loading && (
          <p className="muted" style={{ marginTop: 8, fontSize: 13 }}>
            Le recalcul du pipeline peut prendre une minute…
          </p>
        )}
      </div>
    </div>
  );
}

// ─── Étape 4 : résultat ───────────────────────────────────────────────────────
function StepResultat({ client, type, result, onRestart }) {
  const alertesCritiques = result.alertes.filter(
    (a) => !["doublon_mouvement", "affaire_creee", "unknown_isin"].includes(a.code)
  );
  const alertesInfo = result.alertes.filter((a) =>
    ["doublon_mouvement", "affaire_creee", "unknown_isin"].includes(a.code)
  );

  return (
    <div className="card" style={{ maxWidth: 600 }}>
      <div className="card-header">
        <div>
          <p className="eyebrow">Import terminé</p>
          <h3>
            {type.charAt(0).toUpperCase() + type.slice(1)} — {client.nom} {client.prenom}
          </h3>
        </div>
        <span className="pill" style={{ background: "#f0fdf4", color: "#16a34a" }}>
          ✓ Succès
        </span>
      </div>
      <div style={{ padding: "0 24px 24px" }}>
        {/* KPIs */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
            gap: 12,
            marginBottom: 20,
          }}
        >
          {[
            { label: "Lignes insérées", value: result.insere },
            { label: "Mises à jour", value: result.mis_a_jour },
            ...(result.avis_generes > 0
              ? [{ label: "Avis générés", value: result.avis_generes }]
              : []),
            ...(result.affaires_creees > 0
              ? [{ label: "Affaires créées", value: result.affaires_creees, warn: true }]
              : []),
            { label: "Recalcul (s)", value: result.duree_recalcul_s },
          ].map((k) => (
            <div
              key={k.label}
              style={{
                padding: "14px 16px",
                background: k.warn ? "#fffbeb" : "#f9fafb",
                border: `1px solid ${k.warn ? "#fde68a" : "#e5e7eb"}`,
                borderRadius: 8,
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 24, fontWeight: 700 }}>{k.value}</div>
              <div style={{ fontSize: 12, color: "#6b7280", marginTop: 2 }}>{k.label}</div>
            </div>
          ))}
        </div>

        {/* Alertes critiques */}
        {alertesCritiques.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <p style={{ fontWeight: 600, marginBottom: 6, color: "#dc2626" }}>
              Alertes ({alertesCritiques.length})
            </p>
            {alertesCritiques.map((a, i) => (
              <div
                key={i}
                style={{
                  padding: "7px 12px",
                  background: "#fef2f2",
                  border: "1px solid #fecaca",
                  borderRadius: 6,
                  fontSize: 13,
                  marginBottom: 4,
                  display: "flex",
                  gap: 8,
                  alignItems: "center",
                }}
              >
                <Badge code={a.code} />
                {a.ligne && <span style={{ color: "#6b7280" }}>Ligne {a.ligne} —</span>}
                {a.message}
              </div>
            ))}
          </div>
        )}

        {/* Alertes info */}
        {alertesInfo.length > 0 && (
          <details style={{ marginBottom: 16 }}>
            <summary
              style={{ cursor: "pointer", fontSize: 13, color: "#6b7280", marginBottom: 6 }}
            >
              {alertesInfo.length} information(s) (doublons ignorés, affaires créées…)
            </summary>
            {alertesInfo.map((a, i) => (
              <div
                key={i}
                style={{
                  padding: "6px 12px",
                  fontSize: 12,
                  color: "#6b7280",
                  borderBottom: "1px solid #f3f4f6",
                }}
              >
                <Badge code={a.code} /> {a.message}
              </div>
            ))}
          </details>
        )}

        <button
          type="button"
          onClick={onRestart}
          style={{
            padding: "10px 24px",
            background: "#2563eb",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          Nouvel import
        </button>
      </div>
    </div>
  );
}

// ─── Page principale ──────────────────────────────────────────────────────────
function ImportPortefeuille() {
  const [step, setStep] = useState(0);
  const [ctx, setCtx] = useState({});

  const restart = () => {
    setStep(0);
    setCtx({});
  };

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Opérations</p>
          <h1>Import de fichiers fournisseurs</h1>
          <p className="muted">
            Inventaire de portefeuille ou fichier de mouvements — CSV ou JSON.
          </p>
        </div>
      </div>

      {/* Barre de progression */}
      <div style={{ display: "flex", gap: 0, marginBottom: 24, maxWidth: 520 }}>
        {["Client", "Fichier", "Aperçu", "Résultat"].map((label, i) => (
          <div
            key={label}
            style={{
              flex: 1,
              textAlign: "center",
              padding: "8px 0",
              background: i === step ? "#2563eb" : i < step ? "#bbf7d0" : "#f3f4f6",
              color: i === step ? "#fff" : i < step ? "#15803d" : "#9ca3af",
              fontSize: 12,
              fontWeight: i === step ? 700 : 400,
              borderRadius:
                i === 0 ? "6px 0 0 6px" : i === 3 ? "0 6px 6px 0" : 0,
              transition: "all .2s",
            }}
          >
            {label}
          </div>
        ))}
      </div>

      {step === 0 && (
        <StepClient
          onNext={(data) => {
            setCtx((c) => ({ ...c, ...data }));
            setStep(1);
          }}
        />
      )}
      {step === 1 && (
        <StepFichier
          client={ctx.client}
          idSg={ctx.idSg}
          onNext={(data) => {
            setCtx((c) => ({ ...c, ...data }));
            setStep(2);
          }}
          onBack={() => setStep(0)}
        />
      )}
      {step === 2 && (
        <StepApercu
          client={ctx.client}
          idSg={ctx.idSg}
          type={ctx.type}
          file={ctx.file}
          preview={ctx.preview}
          onNext={(data) => {
            setCtx((c) => ({ ...c, ...data }));
            setStep(3);
          }}
          onBack={() => setStep(1)}
        />
      )}
      {step === 3 && (
        <StepResultat
          client={ctx.client}
          type={ctx.type}
          result={ctx.result}
          onRestart={restart}
        />
      )}
    </section>
  );
}

export default ImportPortefeuille;

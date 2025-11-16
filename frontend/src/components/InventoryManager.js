import { useCallback, useMemo, useState } from "react";

const SUGGESTED_PORTFOLIOS = [
  { isin: "FR0000000016", name: "Fonds Obligataire Durable" },
  { isin: "FR0000000024", name: "Mandat Allocation Prudence" },
  { isin: "LU0000000031", name: "SICAV ESG International" },
  { isin: "FR0000000049", name: "Portefeuille Actions Europe" }
];

function detectDelimiter(sampleLine) {
  if (!sampleLine) {
    return ",";
  }
  const commaCount = sampleLine.split(",").length;
  const semiColonCount = sampleLine.split(";").length;
  return semiColonCount > commaCount ? ";" : ",";
}

function parseCsv(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) {
    return { headers: [], rows: [] };
  }

  const delimiter = detectDelimiter(lines[0]);
  const headers = lines[0].split(delimiter).map((cell) => cell.trim());
  const rows = lines.slice(1).map((line) => line.split(delimiter).map((cell) => cell.trim()));

  return { headers, rows };
}

function formatDate(date) {
  return date.toLocaleDateString("fr-FR", {
    year: "numeric",
    month: "long",
    day: "numeric"
  });
}

function computeLastBusinessDay(year, month) {
  if (!year || !month) {
    return null;
  }

  const y = Number(year);
  const m = Number(month);

  if (Number.isNaN(y) || Number.isNaN(m) || m < 1 || m > 12) {
    return null;
  }

  const date = new Date(y, m, 0);
  const day = date.getDay();
  if (day === 0) {
    date.setDate(date.getDate() - 2);
  } else if (day === 6) {
    date.setDate(date.getDate() - 1);
  }
  return date;
}

function InventoryManager() {
  const portfolioMap = useMemo(
    () => new Map(SUGGESTED_PORTFOLIOS.map((item) => [item.isin, item.name])),
    []
  );

  const today = useMemo(() => new Date(), []);
  const [isin, setIsin] = useState("");
  const [portfolioName, setPortfolioName] = useState("");
  const [month, setMonth] = useState(String(today.getMonth() + 1).padStart(2, "0"));
  const [year, setYear] = useState(String(today.getFullYear()));
  const [csvHeaders, setCsvHeaders] = useState([]);
  const [csvRows, setCsvRows] = useState([]);
  const [csvError, setCsvError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const inventoryDate = useMemo(() => {
    const computed = computeLastBusinessDay(year, month);
    return computed ? formatDate(computed) : null;
  }, [month, year]);

  const handleIsinChange = useCallback(
    (value) => {
      setIsin(value);
      if (portfolioMap.has(value)) {
        setPortfolioName(portfolioMap.get(value) ?? "");
      }
    },
    [portfolioMap]
  );

  const handleFile = useCallback((file) => {
    if (!file) {
      return;
    }
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setCsvError("Le fichier doit être au format .csv");
      setCsvHeaders([]);
      setCsvRows([]);
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result;
        if (typeof text !== "string") {
          throw new Error("Lecture du fichier impossible");
        }
        const { headers, rows } = parseCsv(text);
        if (!headers.length) {
          throw new Error("Le fichier est vide ou mal formaté");
        }
        setCsvHeaders(headers);
        setCsvRows(rows);
        setCsvError("");
      } catch (error) {
        setCsvError(error.message ?? "Erreur lors de l'analyse du fichier CSV");
        setCsvHeaders([]);
        setCsvRows([]);
      }
    };
    reader.onerror = () => {
      setCsvError("Impossible de lire le fichier sélectionné.");
    };
    reader.readAsText(file, "utf-8");
  }, []);

  const handleFileInputChange = useCallback(
    (event) => {
      const file = event.target.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setIsDragging(false);
      const file = event.dataTransfer.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const dropZoneClasses = [
    "flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-8 text-center transition max-w-2xl mx-auto bg-slate-50/60",
    isDragging ? "border-primary bg-primary/5 text-primary" : "border-slate-300/80 text-slate-500"
  ].join(" ");

  return (
    <>
      <form
        className="space-y-6"
        onSubmit={(event) => {
          event.preventDefault();
        }}
      >
        <div className="grid gap-4 md:grid-cols-12">
          <div className="space-y-2 md:col-span-4 lg:col-span-3">
            <label htmlFor="isin" className="text-sm font-medium text-slate-700">
              ISIN
            </label>
            <input
              id="isin"
              list="isin-options"
              value={isin}
              onChange={(event) => handleIsinChange(event.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm shadow-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
              placeholder="FR0000000000"
            />
            <datalist id="isin-options">
              {SUGGESTED_PORTFOLIOS.map((item) => (
                <option key={item.isin} value={item.isin}>
                  {item.name}
                </option>
              ))}
            </datalist>
            <p className="text-xs text-slate-500">
              Saisissez un code ISIN ou sélectionnez un portefeuille connu.
            </p>
          </div>
          <div className="space-y-2 md:col-span-6 lg:col-span-5">
            <label htmlFor="portfolio-name" className="text-sm font-medium text-slate-700">
              Nom du portefeuille
            </label>
            <input
              id="portfolio-name"
              value={portfolioName}
              onChange={(event) => setPortfolioName(event.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm shadow-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
              placeholder="Nom du portefeuille"
            />
            <p className="text-xs text-slate-500">
              Le nom est automatiquement rempli pour les ISIN connus, mais vous pouvez le modifier.
            </p>
          </div>
          <div className="space-y-2 md:col-span-2 lg:col-span-2">
            <label className="text-sm font-medium text-slate-700" htmlFor="month">
              Mois
            </label>
            <select
              id="month"
              value={month}
              onChange={(event) => setMonth(event.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm shadow-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40 md:max-w-[110px]"
            >
              {Array.from({ length: 12 }).map((_, index) => {
                const value = String(index + 1).padStart(2, "0");
                return (
                  <option key={value} value={value}>
                    {value}
                  </option>
                );
              })}
            </select>
          </div>
          <div className="space-y-2 md:col-span-2 lg:col-span-2">
            <label className="text-sm font-medium text-slate-700" htmlFor="year">
              Année
            </label>
            <input
              id="year"
              type="number"
              min="2000"
              max="2100"
              value={year}
              onChange={(event) => setYear(event.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm shadow-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40 md:max-w-[120px]"
            />
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-surface p-4">
          <p className="text-sm font-medium text-slate-700">Inventaire mensuel</p>
          <ul className="mt-2 space-y-1 text-xs text-slate-500">
            <li>
              Dernier jour ouvré calculé :{" "}
              <span className="font-medium text-slate-700">{inventoryDate ?? "Date invalide"}</span>
            </li>
            <li>Format attendu : fichier CSV (délimiteur virgule ou point-virgule)</li>
          </ul>
        </div>

        <div
          className={dropZoneClasses}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setIsDragging(false);
          }}
          onDrop={handleDrop}
        >
          <p className="text-sm font-medium">Déposez votre fichier CSV ici</p>
          <p className="mt-1 text-xs text-slate-500">ou</p>
          <label
            htmlFor="file-upload"
            className="mt-3 inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-primary-dark cursor-pointer"
          >
            Parcourir les fichiers
            <span aria-hidden="true">📁</span>
          </label>
          <input
            id="file-upload"
            type="file"
            accept=".csv,text/csv"
            className="hidden"
            onChange={handleFileInputChange}
          />
        </div>
        {csvError && (
          <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700">
            {csvError}
          </p>
        )}
      </form>
      {csvHeaders.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Aperçu du fichier CSV
            </h3>
            <span className="text-xs text-slate-400">
              {csvRows.length} ligne(s) – affichage de {Math.min(csvRows.length, 10)}
            </span>
          </div>
          <div className="overflow-auto rounded-lg border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50">
                <tr>
                  {csvHeaders.map((header) => (
                    <th
                      key={header}
                      className="whitespace-nowrap px-4 py-2 text-left font-semibold text-slate-600"
                    >
                      {header || "—"}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {csvRows.slice(0, 10).map((row, rowIndex) => (
                  <tr key={rowIndex} className="odd:bg-white even:bg-slate-50">
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex} className="whitespace-nowrap px-4 py-2 text-slate-700">
                        {cell || "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}

export default InventoryManager;

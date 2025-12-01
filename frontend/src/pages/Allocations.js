import { useEffect, useState } from "react";
import { API_BASE_URL } from "../config";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

function Allocations() {
  const [allocations, setAllocations] = useState([]);
  const [year, setYear] = useState("2018");
  const [chartData, setChartData] = useState([]);
  const [noms, setNoms] = useState([]);
  const [selectedNoms, setSelectedNoms] = useState([]);
  const [yDomain, setYDomain] = useState([0, 120]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/allocations/`)
      .then((res) => res.json())
      .then((data) => setAllocations(data))
      .catch((err) => console.error("Erreur API:", err));
  }, []);

  useEffect(() => {
    if (allocations.length === 0) return;

    const filtered = allocations.filter((a) => a.annee === parseInt(year));

    const refs = {};
    filtered.forEach((a) => {
      const d = a.date ? a.date.substring(0, 10) : String(a.annee);
      if (!refs[a.nom] || d < refs[a.nom].date) {
        refs[a.nom] = { date: d, value: a.sicav };
      }
    });

    const byDate = {};
    filtered.forEach((a) => {
      const d = a.date ? a.date.substring(0, 10) : String(a.annee);
      if (!byDate[d]) byDate[d] = { date: d };
      const ref = refs[a.nom]?.value || 1;
      byDate[d][a.nom] = (a.sicav / ref) * 100;
    });

    const sorted = Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date));
    setChartData(sorted);

    const distinctNoms = [...new Set(filtered.map((a) => a.nom))];
    setNoms(distinctNoms);

    const validSelected = selectedNoms.filter((n) => distinctNoms.includes(n));
    const fallback = distinctNoms.length > 0 ? [distinctNoms[0]] : [];
    const nextSelected = validSelected.length > 0 ? validSelected : fallback;
    if (nextSelected.length !== selectedNoms.length || nextSelected.some((n, i) => n !== selectedNoms[i])) {
      setSelectedNoms(nextSelected);
    }

    const activeNoms = nextSelected;
    const allValues = sorted.flatMap((d) => activeNoms.map((nom) => d[nom]).filter((v) => v != null));
    if (allValues.length > 0) {
      const minVal = Math.min(...allValues);
      const maxVal = Math.max(...allValues);
      const margin = (maxVal - minVal) * 0.1;
      setYDomain([minVal - margin, maxVal + margin]);
    }
  }, [year, allocations, selectedNoms]);

  const years = [...new Set(allocations.map((a) => a.annee))].filter(Boolean).sort();

  const toggleNom = (nom) => {
    if (selectedNoms.includes(nom)) {
      setSelectedNoms(selectedNoms.filter((n) => n !== nom));
    } else {
      setSelectedNoms([...selectedNoms, nom]);
    }
  };

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Performances</p>
          <h1>Allocations – base 100 ({year})</h1>
          <p className="muted">Comparer les supports sur une base normalisée.</p>
        </div>
        <span className="pill">{allocations.length} points</span>
      </div>

      <div className="card">
        <div className="filters" style={{ marginBottom: 8 }}>
          <div className="panel">
            <label className="eyebrow">Année</label>
            <select value={year} onChange={(e) => setYear(e.target.value)}>
              {years.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </div>
          <div className="panel">
            <label className="eyebrow">Supports affichés</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {noms.map((nom) => (
                <label key={nom} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input
                    type="checkbox"
                    checked={selectedNoms.includes(nom)}
                    onChange={() => toggleNom(nom)}
                    style={{ width: 16, height: 16 }}
                  />
                  {nom}
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="chart-shell" style={{ height: 420 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 20, right: 40, left: 40, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" padding={{ left: 30, right: 30 }} />
              <YAxis
                domain={yDomain}
                tickFormatter={(val) => val.toFixed(0)}
                label={{ value: "Base 100", angle: -90, position: "insideLeft" }}
              />
              <Tooltip formatter={(val) => `${val.toFixed(2)} (base 100)`} />
              <Legend />
              <ReferenceLine y={100} stroke="red" strokeDasharray="3 3" label="Base 100" />
              {selectedNoms.map((nom) => (
                <Line
                  key={nom}
                  type="monotone"
                  dataKey={nom}
                  name={nom}
                  stroke={"#" + ((Math.random() * 0xffffff) << 0).toString(16)}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}

export default Allocations;

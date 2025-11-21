import { useEffect, useState } from "react";
import Select from "react-select";

function Supports() {
  const [supports, setSupports] = useState([]);
  const [filtered, setFiltered] = useState([]);

  const [filterNom, setFilterNom] = useState("");
  const [filterPromoteur, setFilterPromoteur] = useState("");
  const [filterCatGene, setFilterCatGene] = useState([]);
  const [filterCatPrincipale, setFilterCatPrincipale] = useState([]);
  const [filterCatDet, setFilterCatDet] = useState([]);
  const [filterCatGeo, setFilterCatGeo] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/supports")
      .then((res) => res.json())
      .then((data) => {
        setSupports(data);
        setFiltered(data);
      });
  }, []);

  useEffect(() => {
    let data = [...supports];

    if (filterNom) {
      data = data.filter((s) => (s.nom || "Inconnu").toLowerCase().includes(filterNom.toLowerCase()));
    }
    if (filterPromoteur) {
      data = data.filter((s) => (s.promoteur || "Inconnu").toLowerCase().includes(filterPromoteur.toLowerCase()));
    }
    if (filterCatGene.length > 0) {
      data = data.filter((s) => filterCatGene.some((f) => f.value === (s.cat_gene || "Inconnu")));
    }
    if (filterCatPrincipale.length > 0) {
      data = data.filter((s) => filterCatPrincipale.some((f) => f.value === (s.cat_principale || "Inconnu")));
    }
    if (filterCatDet.length > 0) {
      data = data.filter((s) => filterCatDet.some((f) => f.value === (s.cat_det || "Inconnu")));
    }
    if (filterCatGeo.length > 0) {
      data = data.filter((s) => filterCatGeo.some((f) => f.value === (s.cat_geo || "Inconnu")));
    }

    setFiltered(data);
  }, [filterNom, filterPromoteur, filterCatGene, filterCatPrincipale, filterCatDet, filterCatGeo, supports]);

  const uniqueOptions = (key) =>
    [...new Set(supports.map((s) => s[key] || "Inconnu"))].map((v) => ({
      value: v,
      label: v,
    }));

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Univers de supports</p>
          <h1>Supports</h1>
          <p className="muted">Filtrer par promoteur ou catégories pour isoler des familles.</p>
        </div>
        <span className="pill">
          {filtered.length} filtrés / {supports.length} totaux
        </span>
      </div>

      <div className="card">
        <div className="filters">
          <div className="panel">
            <label className="eyebrow">Nom</label>
            <input value={filterNom} onChange={(e) => setFilterNom(e.target.value)} placeholder="Rechercher un support" />
          </div>
          <div className="panel">
            <label className="eyebrow">Promoteur</label>
            <input
              value={filterPromoteur}
              onChange={(e) => setFilterPromoteur(e.target.value)}
              placeholder="Nom du promoteur"
            />
          </div>
          <div className="panel">
            <label className="eyebrow">Catégorie générale</label>
            <Select
              isMulti
              classNamePrefix="select"
              options={uniqueOptions("cat_gene")}
              value={filterCatGene}
              onChange={setFilterCatGene}
              placeholder="Toutes"
            />
          </div>
          <div className="panel">
            <label className="eyebrow">Catégorie principale</label>
            <Select
              isMulti
              classNamePrefix="select"
              options={uniqueOptions("cat_principale")}
              value={filterCatPrincipale}
              onChange={setFilterCatPrincipale}
              placeholder="Toutes"
            />
          </div>
          <div className="panel">
            <label className="eyebrow">Catégorie détaillée</label>
            <Select
              isMulti
              classNamePrefix="select"
              options={uniqueOptions("cat_det")}
              value={filterCatDet}
              onChange={setFilterCatDet}
              placeholder="Toutes"
            />
          </div>
          <div className="panel">
            <label className="eyebrow">Catégorie géographique</label>
            <Select
              isMulti
              classNamePrefix="select"
              options={uniqueOptions("cat_geo")}
              value={filterCatGeo}
              onChange={setFilterCatGeo}
              placeholder="Toutes"
            />
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <p className="eyebrow">Tableau</p>
            <h3>Supports filtrés</h3>
          </div>
        </div>
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Nom</th>
                <th>Promoteur</th>
                <th>Cat. générale</th>
                <th>Cat. principale</th>
                <th>Cat. détaillée</th>
                <th>Cat. géographique</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s, idx) => (
                <tr key={idx}>
                  <td>{s.nom || "Inconnu"}</td>
                  <td>{s.promoteur || "Inconnu"}</td>
                  <td>{s.cat_gene || "Inconnu"}</td>
                  <td>{s.cat_principale || "Inconnu"}</td>
                  <td>{s.cat_det || "Inconnu"}</td>
                  <td>{s.cat_geo || "Inconnu"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default Supports;

import { useEffect, useState } from "react";
import Select from "react-select";

function Supports() {
    const [supports, setSupports] = useState([]);
    const [filtered, setFiltered] = useState([]);

    // Filtres
    const [filterNom, setFilterNom] = useState("");
    const [filterPromoteur, setFilterPromoteur] = useState("");
    const [filterCatGene, setFilterCatGene] = useState([]);
    const [filterCatPrincipale, setFilterCatPrincipale] = useState([]);
    const [filterCatDet, setFilterCatDet] = useState([]);
    const [filterCatGeo, setFilterCatGeo] = useState([]);

    useEffect(() => {
        fetch("http://127.0.0.1:8000/supports")
            .then(res => res.json())
            .then(data => {
                setSupports(data);
                setFiltered(data);
            });
    }, []);

    useEffect(() => {
        let data = [...supports];

        if (filterNom) {
            data = data.filter(s =>
                (s.nom || "Inconnu").toLowerCase().includes(filterNom.toLowerCase())
            );
        }
        if (filterPromoteur) {
            data = data.filter(s =>
                (s.promoteur || "Inconnu").toLowerCase().includes(filterPromoteur.toLowerCase())
            );
        }
        if (filterCatGene.length > 0) {
            data = data.filter(s =>
                filterCatGene.some(f => f.value === (s.cat_gene || "Inconnu"))
            );
        }
        if (filterCatPrincipale.length > 0) {
            data = data.filter(s =>
                filterCatPrincipale.some(f => f.value === (s.cat_principale || "Inconnu"))
            );
        }
        if (filterCatDet.length > 0) {
            data = data.filter(s =>
                filterCatDet.some(f => f.value === (s.cat_det || "Inconnu"))
            );
        }
        if (filterCatGeo.length > 0) {
            data = data.filter(s =>
                filterCatGeo.some(f => f.value === (s.cat_geo || "Inconnu"))
            );
        }

        setFiltered(data);
    }, [
        filterNom,
        filterPromoteur,
        filterCatGene,
        filterCatPrincipale,
        filterCatDet,
        filterCatGeo,
        supports,
    ]);

    // Générer les options uniques pour chaque champ cat_*
    const uniqueOptions = (key) =>
        [...new Set(supports.map(s => s[key] || "Inconnu"))].map(v => ({
            value: v,
            label: v,
        }));

    return (
        <div>
            <h1>Supports</h1>
            <p>Total: {filtered.length} / {supports.length}</p>

            <div style={{ marginBottom: "1rem" }}>
                <label>Nom : </label>
                <input value={filterNom} onChange={e => setFilterNom(e.target.value)} />

                <label style={{ marginLeft: "1rem" }}>Promoteur : </label>
                <input value={filterPromoteur} onChange={e => setFilterPromoteur(e.target.value)} />
            </div>

            <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
                <div style={{ flex: 1 }}>
                    <label>Catégorie générale</label>
                    <Select
                        isMulti
                        options={uniqueOptions("cat_gene")}
                        value={filterCatGene}
                        onChange={setFilterCatGene}
                    />
                </div>
                <div style={{ flex: 1 }}>
                    <label>Catégorie principale</label>
                    <Select
                        isMulti
                        options={uniqueOptions("cat_principale")}
                        value={filterCatPrincipale}
                        onChange={setFilterCatPrincipale}
                    />
                </div>
                <div style={{ flex: 1 }}>
                    <label>Catégorie détaillée</label>
                    <Select
                        isMulti
                        options={uniqueOptions("cat_det")}
                        value={filterCatDet}
                        onChange={setFilterCatDet}
                    />
                </div>
                <div style={{ flex: 1 }}>
                    <label>Catégorie géographique</label>
                    <Select
                        isMulti
                        options={uniqueOptions("cat_geo")}
                        value={filterCatGeo}
                        onChange={setFilterCatGeo}
                    />
                </div>
            </div>

            <table border="1" cellPadding="5" style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                    <tr>
                        <th>Nom</th>
                        <th>Promoteur</th>
                        <th>Catégorie générale</th>
                        <th>Catégorie principale</th>
                        <th>Catégorie détaillée</th>
                        <th>Catégorie géographique</th>
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
    );
}

export default Supports;

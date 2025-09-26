import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

function Affaires() {
    const [affaires, setAffaires] = useState([]);
    const [srriData, setSrriData] = useState([]);

    useEffect(() => {
        fetch("http://127.0.0.1:8000/reporting/affaires")
            .then(res => res.json())
            .then(data => {
                setAffaires(data);

                // Grouper par SRRI
                const counts = {};
                data.forEach(a => {
                    const srri = a.SRRI || "Inconnu";
                    counts[srri] = (counts[srri] || 0) + 1;
                });

                setSrriData(Object.entries(counts).map(([k, v]) => ({ srri: k, count: v })));
            });
    }, []);

    return (
        <div>
            <h1>Affaires</h1>
            <p>Total: {affaires.length}</p>

            <h2>Répartition par SRRI</h2>
            <ResponsiveContainer width="100%" height={300}>
                <BarChart data={srriData}>
                    <XAxis dataKey="srri" />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="count" fill="#82ca9d" />
                </BarChart>
            </ResponsiveContainer>

            <h2>Debug (5 premiers)</h2>
            <pre>{JSON.stringify(affaires.slice(0, 5), null, 2)}</pre>
        </div>
    );
}

export default Affaires;

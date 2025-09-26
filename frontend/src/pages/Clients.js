import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

function Clients() {
    const [clients, setClients] = useState([]);
    const [srriData, setSrriData] = useState([]);

    useEffect(() => {
        fetch("http://127.0.0.1:8000/reporting/clients")
            .then(res => res.json())
            .then(data => {
                setClients(data);

                // Grouper les clients par valeur de SRRI
                const counts = {};
                data.forEach(c => {
                    const srri = c.SRRI || "Inconnu";
                    counts[srri] = (counts[srri] || 0) + 1;
                });

                // Transformer en tableau pour Recharts
                setSrriData(Object.entries(counts).map(([k, v]) => ({ srri: k, count: v })));
            });
    }, []);

    return (
        <div>
            <h1>Clients</h1>
            <p>Total: {clients.length}</p>

            <h2>Répartition par SRRI</h2>
            <ResponsiveContainer width="100%" height={300}>
                <BarChart data={srriData}>
                    <XAxis dataKey="srri" />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
            </ResponsiveContainer>

            <h2>Debug (5 premiers)</h2>
            <pre>{JSON.stringify(clients.slice(0, 5), null, 2)}</pre>
        </div>
    );
}

export default Clients;

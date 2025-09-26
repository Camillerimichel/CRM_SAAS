import { useEffect, useState } from "react";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

function Dashboard() {
  const [stats, setStats] = useState({ clients: 0, affaires: 0, allocations: 0, supports: 0 });

  useEffect(() => {
    Promise.all([
      fetch("http://127.0.0.1:8000/reporting/clients").then(res => res.json()),
      fetch("http://127.0.0.1:8000/reporting/affaires").then(res => res.json()),
      fetch("http://127.0.0.1:8000/reporting/allocations").then(res => res.json()),
      fetch("http://127.0.0.1:8000/reporting/supports").then(res => res.json()),
    ]).then(([c, a, al, s]) => {
      setStats({ clients: c.length, affaires: a.length, allocations: al.length, supports: s.length });
    });
  }, []);

  const data = [
    { name: "Clients", value: stats.clients },
    { name: "Affaires", value: stats.affaires },
    { name: "Allocations", value: stats.allocations },
    { name: "Supports", value: stats.supports },
  ];
  const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff8042"];

  return (
    <div>
      <h1>Dashboard</h1>
      <ul>
        <li>Clients: {stats.clients}</li>
        <li>Affaires: {stats.affaires}</li>
        <li>Allocations: {stats.allocations}</li>
        <li>Supports: {stats.supports}</li>
      </ul>

      <h2>Visualisation</h2>
      <div style={{ display: "flex", gap: "2rem", height: 300 }}>
        <ResponsiveContainer width="45%" height="100%">
          <PieChart>
            <Pie data={data} dataKey="value" nameKey="name" outerRadius={100} label>
              {data.map((entry, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Legend />
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>

        <ResponsiveContainer width="45%" height="100%">
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default Dashboard;

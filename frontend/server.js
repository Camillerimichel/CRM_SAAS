const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 8101;
const HOST = process.env.HOST || "0.0.0.0";
const BUILD_DIR = path.join(__dirname, "build");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".map": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

const resolveFilePath = (requestPath) => {
  const cleanPath = requestPath.split("?")[0].split("#")[0];
  let relativePath = decodeURI(cleanPath);
  if (relativePath.endsWith("/")) {
    relativePath = path.join(relativePath, "index.html");
  }
  if (relativePath === "/") {
    relativePath = "/index.html";
  }

  let filePath = path.join(BUILD_DIR, relativePath);
  if (!filePath.startsWith(BUILD_DIR)) {
    return path.join(BUILD_DIR, "index.html");
  }
  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    return path.join(BUILD_DIR, "index.html");
  }
  return filePath;
};

const server = http.createServer((req, res) => {
  if (!fs.existsSync(BUILD_DIR)) {
    res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Build directory missing. Run `npm run build` first.");
    return;
  }

  const filePath = resolveFilePath(req.url || "/");
  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || "application/octet-stream";

  fs.readFile(filePath, (err, data) => {
    if (err) {
      console.error("Error serving", filePath, err);
      res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Internal server error");
      return;
    }
    res.writeHead(200, { "Content-Type": contentType });
    res.end(data);
  });
});

server.listen(PORT, HOST, () => {
  console.log(`Frontend served from ${BUILD_DIR} on http://${HOST}:${PORT}`);
});

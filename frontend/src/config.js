const computeDefaultApiBase = () => {
  if (typeof window === "undefined") {
    return "";
  }
  const { hostname, port, protocol } = window.location;
  const isDevServer = port === "3000";
  const isLocalHost = hostname === "localhost" || hostname === "127.0.0.1";

  if (isDevServer || isLocalHost) {
    const apiHost = hostname === "localhost" ? "127.0.0.1" : hostname;
    return `${protocol}//${apiHost}:8100`;
  }

  return `${protocol}//${hostname}`;
};

export const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || computeDefaultApiBase();

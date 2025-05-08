// index.js
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { AuthProvider } from "react-oidc-context";

const cognitoAuthConfig = {
  authority: "https://cognito-idp.eu-north-1.amazonaws.com/eu-north-1_O4CIZM2aJ",
  client_id: "46g5ig1s03klm5guub0h4qkv2e",
  redirect_uri: "https://d3cuf8vpsz3a8w.cloudfront.net/",
  response_type: "code",
  scope: "email openid phone profile",
  // Add the following configurations to properly handle refresh
  onSigninCallback: () => {
    // Remove the query parameters from the URL after login
    window.history.replaceState({}, document.title, window.location.pathname);
  },
  automaticSilentRenew: true,
  loadUserInfo: true,
  // This will prevent errors on refresh
  monitorSession: false
};

const root = ReactDOM.createRoot(document.getElementById("root"));

// wrap the application with AuthProvider
root.render(
  <React.StrictMode>
    <AuthProvider {...cognitoAuthConfig}>
      <App />
    </AuthProvider>
  </React.StrictMode>
);
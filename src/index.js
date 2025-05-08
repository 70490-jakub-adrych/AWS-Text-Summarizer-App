// index.js
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { AuthProvider } from "react-oidc-context";
import { USER_POOL_ID, AWS_REGION, USER_POOL_CLIENT_ID, REDIRECT_URI } from "./config";

const cognitoAuthConfig = {
  authority: `https://cognito-idp.${AWS_REGION}.amazonaws.com/${USER_POOL_ID}`,
  client_id: USER_POOL_CLIENT_ID,
  redirect_uri: REDIRECT_URI,
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
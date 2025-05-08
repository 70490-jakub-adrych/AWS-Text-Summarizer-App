// App.js

import { useAuth } from "react-oidc-context";
import { useEffect } from "react";
import Summarize from "./components/Summarize";
import './App.css';

function App() {
  const auth = useAuth();

  // Helper function to handle refresh errors
  useEffect(() => {
    // If there's an error related to state storage, try to recover
    if (auth.error && auth.error.message.includes("No matching state found in storage")) {
      console.log("Recovering from state storage error...");
      // Clear any stale state and attempt to recover
      auth.clearStaleState().then(() => {
        // If the user was previously logged in (check localStorage)
        if (localStorage.getItem("wasLoggedIn") === "true") {
          // Attempt silent login if possible
          auth.signinSilent().catch(err => {
            console.log("Silent sign-in failed, redirecting to login page", err);
            // If silent login fails, redirect to login page
            auth.signinRedirect();
          });
        }
      });
    }
    
    // Store authentication state for recovery after refresh
    if (auth.isAuthenticated) {
      localStorage.setItem("wasLoggedIn", "true");
    }
    
    return () => {
      // Cleanup
    };
  }, [auth.error, auth.isAuthenticated, auth]);

  const signOutRedirect = () => {
    // Clear the login state indicator when logging out
    localStorage.removeItem("wasLoggedIn");
    
    const clientId = "46g5ig1s03klm5guub0h4qkv2e";
    const logoutUri = "https://d3cuf8vpsz3a8w.cloudfront.net";
    const cognitoDomain = "https://eu-north-1o4cizm2aj.auth.eu-north-1.amazoncognito.com";
    window.location.href = `${cognitoDomain}/logout?client_id=${clientId}&logout_uri=${encodeURIComponent(logoutUri)}`;
  };

  if (auth.isLoading) {
    return (
      <div className="loading">
        <h2>Loading...</h2>
        <p>Please wait while we authenticate you</p>
      </div>
    );
  }

  if (auth.error) {
    // Display error but with a recovery option
    return (
      <div className="error">
        <h2>Authentication Error</h2>
        <p>Error: {auth.error.message}</p>
        <button className="auth-button" onClick={() => auth.signinRedirect()}>Try signing in again</button>
      </div>
    );
  }

  if (auth.isAuthenticated) {
    return (
      <div className="app-container">
        <header className="app-header">
          <h1>Text Summarizer</h1>
          <div className="user-info">
            <span>Welcome, {auth.user?.profile.email}</span>
            <button 
              className="auth-button logout"
              onClick={() => {
                localStorage.removeItem("wasLoggedIn");
                auth.removeUser();
              }}
            >
              Sign out
            </button>
          </div>
        </header>
        
        <main>
          <Summarize idToken={auth.user?.id_token} />
        </main>
        
        <footer className="app-footer">
          <p>© 2023 AWS Text Summarization App | Using SageMaker and Cognito</p>
        </footer>
      </div>
    );
  }

  return (
    <div className="app-container login">
      <div className="login-panel">
        <h1>Text Summarizer</h1>
        <p>Please sign in to continue</p>
        <button className="auth-button" onClick={() => auth.signinRedirect()}>Sign in</button>
      </div>
    </div>
  );
}

export default App;
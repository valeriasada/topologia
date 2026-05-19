# Project Setup Summary: Mountain-UI Dashboard

## 1. Project Overview & Architecture
* **Local Project Name:** `Dashboard MB`
* **GitHub Repository:** `Mountain-UI`
* **Architecture Chosen:** "Option 2" (Secure Monorepo)
  * A single main folder containing two distinct subfolders: `/backend` and `/frontend`.
  * **Why:** This keeps Google Sheets API keys completely hidden on the server, ensuring they are never exposed to website visitors.

## 2. Directory Structure
```text
Dashboard MB/ (Main Monorepo)
├── .git/                 # Hidden Git tracking folder
├── .gitignore            # Tells Git to ignore heavy/secret files (node_modules/, .env)
├── backend/              # The "Middleman" server
│   ├── node_modules/     # Downloaded dependencies (ignored by Git)
│   ├── package.json      # Project ID card and list of dependencies
│   ├── package-lock.json # Exact version tree of dependencies
│   └── server.js         # The actual server code and mock data endpoint
└── frontend/             # (Empty for now - next step!)
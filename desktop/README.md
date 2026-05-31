# ferryllm desktop

Tauri + React + TypeScript control panel for ferryllm.

## Features

- Visual TOML configuration editing for server, logging, auth, metrics, prompt cache, providers, key watch entries, and routes.
- Provider management with direct `api_key`, `api_key_env`, `api_key_url`, `api_key_file`, or `key_watch` key sources.
- Start, stop, restart, validate, and monitor an installed `ferryllm` executable.
- Full configuration hot reload by saving, validating, and automatically restarting the managed ferryllm process when enabled.

## Development

```bash
npm install
npm run tauri dev
```

## Build

```bash
npm run tauri build
```

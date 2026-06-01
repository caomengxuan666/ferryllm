use std::{
    collections::{HashSet, VecDeque},
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

const MAX_LOG_LINES: usize = 1_000;

struct DesktopState {
    runtime: Arc<Mutex<RuntimeState>>,
}

#[derive(Default)]
struct RuntimeState {
    child: Option<Child>,
    executable: String,
    config_path: String,
    logs: VecDeque<LogEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigFile {
    name: String,
    path: String,
}

#[derive(Debug, Clone, Serialize)]
struct ConfigDocument {
    path: String,
    raw: String,
    config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
struct ProcessStatus {
    running: bool,
    managed: bool,
    source: String,
    executable: String,
    config_path: String,
    pid: Option<u32>,
    logs: Vec<LogEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeResult {
    ok: bool,
    status: Option<u16>,
    body: String,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ServerSnapshot {
    status: ProcessStatus,
    health: ProbeResult,
    ready: ProbeResult,
    metrics: ProbeResult,
    models: ProbeResult,
    fetched_at_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
struct AiSession {
    id: String,
    tool: String,
    cwd: String,
    path: String,
    title: String,
    preview: String,
    message_count: usize,
    updated_at_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
struct CommandResult {
    ok: bool,
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Clone, Serialize)]
struct SaveResult {
    path: String,
    validation: CommandResult,
    reloaded: bool,
}

#[derive(Debug, Clone, Serialize)]
struct LogEntry {
    ts_ms: u128,
    stream: String,
    line: String,
}

#[derive(Debug, Deserialize)]
struct SaveConfigRequest {
    path: String,
    config: serde_json::Value,
    executable: Option<String>,
    hot_reload: bool,
}

#[derive(Debug, Deserialize)]
struct LaunchRequest {
    directory: String,
    listen: String,
    #[serde(default = "default_tool")]
    tool: String,
    #[serde(default)]
    provider_name: Option<String>,
    #[serde(default)]
    client_model: Option<String>,
    #[serde(default)]
    client_reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResumeSessionRequest {
    id: String,
    cwd: String,
    listen: String,
    tool: String,
    #[serde(default)]
    provider_name: Option<String>,
    #[serde(default)]
    client_model: Option<String>,
    #[serde(default)]
    client_reasoning_effort: Option<String>,
}

fn default_tool() -> String {
    "codex".into()
}

fn cli_command(tool: &str) -> &'static str {
    match tool {
        "claude" => "claude",
        "opencode" => "opencode",
        _ => "codex",
    }
}

fn claude_setting_sources_arg() -> &'static str {
    "project,local"
}

fn launch_args(
    tool: &str,
    listen: &str,
    client_model: Option<&str>,
    client_reasoning_effort: Option<&str>,
) -> Vec<String> {
    match tool {
        "claude" => vec![
            "--setting-sources".into(),
            claude_setting_sources_arg().into(),
            "--model".into(),
            normalized_client_model(tool, client_model),
        ],
        "codex" => {
            let mut args = codex_gateway_config_args(listen);
            args.push("--model".into());
            args.push(normalized_client_model(tool, client_model));
            if let Some(effort) = normalized_reasoning_effort(client_reasoning_effort) {
                args.push("-c".into());
                args.push(format!("model_reasoning_effort={effort}"));
            }
            args
        }
        _ => Vec::new(),
    }
}

fn gateway_host(listen: &str) -> String {
    if listen.starts_with("0.0.0.0") {
        listen.replace("0.0.0.0", "127.0.0.1")
    } else {
        listen.to_string()
    }
}

fn codex_gateway_config_args(listen: &str) -> Vec<String> {
    let base_url = format!("http://{}/v1", gateway_host(listen));
    vec![
        "-c".into(),
        "model_provider=ferryllm".into(),
        "-c".into(),
        "model_providers.ferryllm.name=ferryllm".into(),
        "-c".into(),
        "model_providers.ferryllm.wire_api=responses".into(),
        "-c".into(),
        format!("model_providers.ferryllm.base_url={base_url}"),
        "-c".into(),
        "model_providers.ferryllm.requires_openai_auth=true".into(),
    ]
}

fn normalized_client_model(tool: &str, client_model: Option<&str>) -> String {
    client_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default_client_model(tool))
        .to_string()
}

fn default_client_model(tool: &str) -> &'static str {
    match tool {
        "claude" => "claude-sonnet-4-5",
        _ => "gpt-5.4",
    }
}

fn normalized_reasoning_effort(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn claude_thinking_tokens(value: Option<&str>) -> Option<u32> {
    match value?.trim() {
        "none" => Some(0),
        "minimal" => Some(512),
        "low" => Some(1024),
        "medium" => Some(4096),
        "high" => Some(8192),
        "xhigh" => Some(16384),
        "max" => Some(32768),
        "ultracode" => Some(65536),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
struct SyncConfigRequest {
    config: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct LauncherStateRequest {
    state: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ValidateConfigRequest {
    executable: Option<String>,
    config: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ProviderProbeRequest {
    name: String,
    #[serde(rename = "type")]
    provider_type: String,
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    api_key_file: Option<String>,
    #[serde(default)]
    api_key_url: Option<String>,
    mode: String,
}

#[tauri::command]
async fn write_config_to_default(request: SyncConfigRequest) -> Result<String, String> {
    spawn_blocking_result(move || {
        let path = default_config_toml_path()?;
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(string_error)?;
        }
        let toml = toml::to_string_pretty(&request.config).map_err(string_error)?;
        fs::write(&path, toml).map_err(string_error)?;
        Ok(path_to_string(&path))
    })
    .await
}

#[tauri::command]
async fn save_config_to_default(request: SyncConfigRequest) -> Result<(), String> {
    spawn_blocking_result(move || {
        let dir = dirs::config_dir()
            .ok_or("cannot find config directory")?
            .join("ferryllm");
        fs::create_dir_all(&dir).map_err(string_error)?;
        let path = dir.join("config.json");
        let json = serde_json::to_string_pretty(&request.config).map_err(string_error)?;
        fs::write(&path, json).map_err(string_error)?;
        Ok(())
    })
    .await
}

#[tauri::command]
async fn save_launcher_state(request: LauncherStateRequest) -> Result<(), String> {
    spawn_blocking_result(move || {
        let dir = dirs::config_dir()
            .ok_or("cannot find config directory")?
            .join("ferryllm");
        fs::create_dir_all(&dir).map_err(string_error)?;
        let path = dir.join("launcher.json");
        let json = serde_json::to_string_pretty(&request.state).map_err(string_error)?;
        fs::write(&path, json).map_err(string_error)?;
        Ok(())
    })
    .await
}

#[tauri::command]
async fn load_launcher_state() -> Result<Option<String>, String> {
    spawn_blocking_result(move || {
        let path = dirs::config_dir()
            .ok_or("cannot find config directory")?
            .join("ferryllm")
            .join("launcher.json");
        if path.exists() {
            let content = fs::read_to_string(&path).map_err(string_error)?;
            Ok(Some(content))
        } else {
            Ok(None)
        }
    })
    .await
}

#[tauri::command]
async fn load_config_from_default() -> Result<Option<String>, String> {
    spawn_blocking_result(move || {
        let json_path = dirs::config_dir()
            .ok_or("cannot find config directory")?
            .join("ferryllm")
            .join("config.json");
        if json_path.exists() {
            let content = fs::read_to_string(&json_path).map_err(string_error)?;
            let mut value: serde_json::Value =
                serde_json::from_str(&content).map_err(string_error)?;
            merge_runtime_key_sources(&mut value);
            Ok(Some(serde_json::to_string(&value).map_err(string_error)?))
        } else {
            let toml_path = default_config_toml_path()?;
            if !toml_path.exists() {
                return Ok(None);
            }
            let raw = fs::read_to_string(&toml_path).map_err(string_error)?;
            let value: toml::Value = toml::from_str(&raw).map_err(string_error)?;
            let json = serde_json::to_value(value).map_err(string_error)?;
            Ok(Some(serde_json::to_string(&json).map_err(string_error)?))
        }
    })
    .await
}

fn merge_runtime_key_sources(draft: &mut serde_json::Value) {
    let Ok(toml_path) = default_config_toml_path() else {
        return;
    };
    if !toml_path.exists() {
        return;
    }
    let Ok(raw) = fs::read_to_string(&toml_path) else {
        return;
    };
    let Ok(runtime_toml) = toml::from_str::<toml::Value>(&raw) else {
        return;
    };
    let Ok(runtime) = serde_json::to_value(runtime_toml) else {
        return;
    };
    let Some(draft_providers) = draft
        .get_mut("providers")
        .and_then(|value| value.as_array_mut())
    else {
        return;
    };
    let Some(runtime_providers) = runtime.get("providers").and_then(|value| value.as_array())
    else {
        return;
    };
    for draft_provider in draft_providers {
        if provider_has_key_source(draft_provider) {
            continue;
        }
        let Some(name) = draft_provider.get("name").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(runtime_provider) = runtime_providers
            .iter()
            .find(|provider| provider.get("name").and_then(|value| value.as_str()) == Some(name))
        else {
            continue;
        };
        for key in [
            "api_key",
            "api_key_env",
            "api_key_url",
            "api_key_file",
            "key_watch",
        ] {
            if let Some(value) = runtime_provider.get(key) {
                draft_provider[key] = value.clone();
            }
        }
    }
}

fn provider_has_key_source(provider: &serde_json::Value) -> bool {
    ["api_key", "api_key_env", "api_key_url", "api_key_file"]
        .iter()
        .any(|key| provider.get(*key).is_some_and(json_value_is_present))
        || provider.get("key_watch").is_some_and(json_value_is_present)
}

fn json_value_is_present(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::String(value) => !value.trim().is_empty(),
        serde_json::Value::Array(value) => !value.is_empty(),
        serde_json::Value::Object(value) => !value.is_empty(),
        _ => true,
    }
}

#[derive(Debug, Deserialize)]
struct ServerRequest {
    executable: Option<String>,
    config_path: String,
    #[serde(default)]
    replace_existing: bool,
}

#[derive(Debug, Deserialize)]
struct WorkspacePathRequest {
    path: String,
}

#[tauri::command]
async fn create_workspace(request: WorkspacePathRequest) -> Result<(), String> {
    spawn_blocking_result(move || {
        let path = PathBuf::from(request.path);
        fs::create_dir_all(&path).map_err(string_error)?;
        Ok(())
    })
    .await
}

#[tauri::command]
async fn delete_workspace(request: WorkspacePathRequest) -> Result<(), String> {
    spawn_blocking_result(move || {
        let path = PathBuf::from(request.path);
        if !path.exists() {
            return Ok(());
        }
        if !path.is_dir() {
            return Err(format!(
                "workspace is not a directory: {}",
                path_to_string(&path)
            ));
        }
        fs::remove_dir_all(&path).map_err(string_error)?;
        Ok(())
    })
    .await
}

#[tauri::command]
async fn delete_ai_session(request: WorkspacePathRequest) -> Result<(), String> {
    spawn_blocking_result(move || {
        let path = PathBuf::from(request.path);
        if !path.exists() {
            return Ok(());
        }
        if !path.is_file() {
            return Err(format!(
                "session path is not a file: {}",
                path_to_string(&path)
            ));
        }
        fs::remove_file(&path).map_err(string_error)?;
        Ok(())
    })
    .await
}

#[tauri::command]
async fn reveal_workspace(request: WorkspacePathRequest) -> Result<(), String> {
    spawn_blocking_result(move || reveal_workspace_inner(request)).await
}

fn reveal_workspace_inner(request: WorkspacePathRequest) -> Result<(), String> {
    let path = PathBuf::from(request.path);
    if !path.exists() {
        return Err(format!(
            "workspace does not exist: {}",
            path_to_string(&path)
        ));
    }

    #[cfg(windows)]
    {
        Command::new("explorer")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("failed to open Explorer: {}", e))?;
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("failed to open Finder: {}", e))?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("failed to open file manager: {}", e))?;
    }
    Ok(())
}

#[tauri::command]
async fn launch_cli(request: LaunchRequest) -> Result<(), String> {
    spawn_blocking_result(move || launch_cli_inner(request)).await
}

fn launch_cli_inner(request: LaunchRequest) -> Result<(), String> {
    let env = client_gateway_env(
        &request.listen,
        &request.tool,
        request.provider_name.as_deref(),
        request.client_model.as_deref(),
        request.client_reasoning_effort.as_deref(),
    );
    let cli_cmd = cli_command(&request.tool);
    let args = launch_args(
        &request.tool,
        &request.listen,
        request.client_model.as_deref(),
        request.client_reasoning_effort.as_deref(),
    );

    #[cfg(windows)]
    {
        let command_line = windows_cli_command_line(cli_cmd, &args);
        let mut command = Command::new("cmd");
        command
            .args(["/C", "start", "", "cmd", "/K"])
            .arg(command_line)
            .current_dir(&request.directory);
        apply_client_env(&mut command, &env);
        command
            .spawn()
            .map_err(|e| format!("failed to launch CLI: {}", e))?;
    }
    #[cfg(not(windows))]
    {
        let mut command = Command::new(cli_cmd);
        command.args(&args).current_dir(&request.directory);
        apply_client_env(&mut command, &env);
        command
            .spawn()
            .map_err(|e| format!("failed to launch CLI: {}", e))?;
    }
    Ok(())
}

#[tauri::command]
async fn launch_vscode(request: LaunchRequest) -> Result<(), String> {
    spawn_blocking_result(move || launch_vscode_inner(request)).await
}

#[tauri::command]
async fn scan_ai_sessions() -> Result<Vec<AiSession>, String> {
    spawn_blocking_result(scan_ai_sessions_inner).await
}

#[tauri::command]
async fn resume_ai_session(request: ResumeSessionRequest) -> Result<(), String> {
    spawn_blocking_result(move || resume_ai_session_inner(request)).await
}

fn scan_ai_sessions_inner() -> Result<Vec<AiSession>, String> {
    let home = dirs::home_dir().ok_or("cannot find home directory")?;
    let mut sessions = Vec::new();
    collect_codex_sessions(&home.join(".codex").join("sessions"), &mut sessions);
    collect_claude_sessions(&home.join(".claude").join("projects"), &mut sessions);
    sessions.sort_by(|a, b| b.updated_at_ms.cmp(&a.updated_at_ms));
    sessions.truncate(300);
    Ok(sessions)
}

fn resume_ai_session_inner(request: ResumeSessionRequest) -> Result<(), String> {
    let env = client_gateway_env(
        &request.listen,
        &request.tool,
        request.provider_name.as_deref(),
        request.client_model.as_deref(),
        request.client_reasoning_effort.as_deref(),
    );
    let cli_cmd = cli_command(&request.tool);
    let args: Vec<String> = match request.tool.as_str() {
        "claude" => vec![
            "--setting-sources".into(),
            claude_setting_sources_arg().into(),
            "--model".into(),
            normalized_client_model(&request.tool, request.client_model.as_deref()),
            "--resume".into(),
            request.id.clone(),
        ],
        "codex" => {
            let mut args = codex_gateway_config_args(&request.listen);
            args.push("--model".into());
            args.push(normalized_client_model(
                &request.tool,
                request.client_model.as_deref(),
            ));
            if let Some(effort) =
                normalized_reasoning_effort(request.client_reasoning_effort.as_deref())
            {
                args.push("-c".into());
                args.push(format!("model_reasoning_effort={effort}"));
            }
            args.push("resume".into());
            args.push(request.id.clone());
            args
        }
        _ => vec!["resume".into(), request.id.clone()],
    };

    #[cfg(windows)]
    {
        let command_line = windows_cli_command_line(cli_cmd, &args);
        let mut command = Command::new("cmd");
        command
            .args(["/C", "start", "", "cmd", "/K"])
            .arg(command_line)
            .current_dir(&request.cwd);
        apply_client_env(&mut command, &env);
        command
            .spawn()
            .map_err(|e| format!("failed to resume session: {}", e))?;
    }
    #[cfg(not(windows))]
    {
        let mut command = Command::new(cli_cmd);
        command.args(args).current_dir(&request.cwd);
        apply_client_env(&mut command, &env);
        command
            .spawn()
            .map_err(|e| format!("failed to resume session: {}", e))?;
    }
    Ok(())
}

#[cfg(windows)]
fn windows_cli_command_line(cli_cmd: &str, args: &[String]) -> String {
    let mut command_line = cli_cmd.to_string();
    for arg in args {
        command_line.push(' ');
        command_line.push_str(&windows_cmd_quote_arg(arg));
    }
    command_line
}

#[cfg(windows)]
fn windows_cmd_quote_arg(arg: &str) -> String {
    if !arg.is_empty()
        && !arg
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '&' | '|' | '<' | '>' | '^' | '"'))
    {
        return arg.to_string();
    }
    format!("\"{}\"", arg.replace('"', "\\\""))
}

fn collect_codex_sessions(root: &Path, sessions: &mut Vec<AiSession>) {
    for path in jsonl_files(root, 800) {
        let Ok(file) = fs::File::open(&path) else {
            continue;
        };
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let Some(payload) = value.get("payload") else {
            continue;
        };
        let Some(id) = payload.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(cwd) = payload.get("cwd").and_then(|v| v.as_str()) else {
            continue;
        };
        let (preview, message_count) = scan_session_preview(&mut reader, "codex", 220);
        sessions.push(AiSession {
            id: id.to_string(),
            tool: "codex".into(),
            cwd: cwd.to_string(),
            path: path_to_string(&path),
            title: session_title_from_path(&path, id),
            preview,
            message_count,
            updated_at_ms: file_modified_ms(&path),
        });
    }
}

fn collect_claude_sessions(root: &Path, sessions: &mut Vec<AiSession>) {
    for path in jsonl_files(root, 800) {
        let Ok(file) = fs::File::open(&path) else {
            continue;
        };
        let reader = BufReader::new(file);
        let mut found: Option<AiSession> = None;
        let mut preview = String::new();
        let mut message_count = 0usize;
        for line in reader.lines().map_while(Result::ok).take(40) {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            let (line_preview, line_message_count) = session_preview_from_value(&value, "claude");
            message_count += line_message_count;
            if preview.is_empty() {
                preview = line_preview;
            }
            let id = value
                .get("sessionId")
                .and_then(|v| v.as_str())
                .or_else(|| value.get("session_id").and_then(|v| v.as_str()));
            let cwd = value.get("cwd").and_then(|v| v.as_str());
            if let (Some(id), Some(cwd)) = (id, cwd) {
                found = Some(AiSession {
                    id: id.to_string(),
                    tool: "claude".into(),
                    cwd: cwd.to_string(),
                    path: path_to_string(&path),
                    title: session_title_from_path(&path, id),
                    preview: preview.clone(),
                    message_count,
                    updated_at_ms: file_modified_ms(&path),
                });
                break;
            }
        }
        if let Some(session) = found {
            sessions.push(session);
        }
    }
}

fn scan_session_preview<R: BufRead>(
    reader: &mut R,
    tool: &str,
    max_lines: usize,
) -> (String, usize) {
    let mut preview = String::new();
    let mut message_count = 0usize;
    let mut line = String::new();
    for _ in 0..max_lines {
        line.clear();
        let Ok(bytes) = reader.read_line(&mut line) else {
            break;
        };
        if bytes == 0 {
            break;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let (line_preview, line_message_count) = session_preview_from_value(&value, tool);
        message_count += line_message_count;
        if preview.is_empty() {
            preview = line_preview;
        }
    }
    (preview, message_count)
}

fn session_preview_from_value(value: &serde_json::Value, tool: &str) -> (String, usize) {
    match tool {
        "claude" => claude_preview_from_value(value),
        _ => codex_preview_from_value(value),
    }
}

fn codex_preview_from_value(value: &serde_json::Value) -> (String, usize) {
    let Some(payload) = value.get("payload") else {
        return (String::new(), 0);
    };
    let payload_type = payload
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let role = payload
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let is_user =
        role == "user" || payload_type == "user_message" || payload_type == "input_message";
    let is_event = value.get("type").and_then(|v| v.as_str()) == Some("event_msg");
    let count = usize::from(is_user || (is_event && payload.get("message").is_some()));

    let text = if is_user {
        text_from_json(payload.get("content")).or_else(|| text_from_json(payload.get("message")))
    } else if is_event {
        text_from_json(payload.get("message"))
            .or_else(|| text_from_json(payload.get("text")))
            .or_else(|| text_from_json(payload.get("text_elements")))
    } else if payload_type == "summary" {
        text_from_json(payload.get("summary"))
    } else {
        None
    };

    (text.unwrap_or_default(), count)
}

fn claude_preview_from_value(value: &serde_json::Value) -> (String, usize) {
    let value_type = value
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let message = value.get("message");
    let role = message
        .and_then(|v| v.get("role"))
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let is_user = value_type == "user" || role == "user";
    let count = usize::from(is_user);
    let text = if is_user {
        message.and_then(|v| text_from_json(v.get("content")))
    } else {
        None
    };
    (text.unwrap_or_default(), count)
}

fn text_from_json(value: Option<&serde_json::Value>) -> Option<String> {
    let value = value?;
    match value {
        serde_json::Value::String(text) => clean_preview_text(text),
        serde_json::Value::Array(items) => {
            let text = items
                .iter()
                .filter_map(|item| {
                    if let Some(text) = item.as_str() {
                        return Some(text.to_string());
                    }
                    item.get("text")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("content").and_then(|v| v.as_str()))
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
                .join(" ");
            clean_preview_text(&text)
        }
        serde_json::Value::Object(map) => map
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| map.get("content").and_then(|v| v.as_str()))
            .and_then(clean_preview_text),
        _ => None,
    }
}

fn clean_preview_text(text: &str) -> Option<String> {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = compact.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut preview = trimmed.chars().take(180).collect::<String>();
    if trimmed.chars().count() > 180 {
        preview.push('…');
    }
    Some(preview)
}

fn jsonl_files(root: &Path, limit: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_jsonl_files(root, &mut files, limit);
    files.sort_by_key(|path| std::cmp::Reverse(file_modified_ms(path)));
    files.truncate(limit);
    files
}

fn collect_jsonl_files(dir: &Path, files: &mut Vec<PathBuf>, limit: usize) {
    if files.len() >= limit || !dir.exists() {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if files.len() >= limit {
            return;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_jsonl_files(&path, files, limit);
        } else if path.extension().and_then(|v| v.to_str()) == Some("jsonl") {
            files.push(path);
        }
    }
}

fn file_modified_ms(path: &Path) -> u128 {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn session_title_from_path(path: &Path, id: &str) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.strip_prefix("rollout-").unwrap_or(value).to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| id.to_string())
}

fn launch_vscode_inner(request: LaunchRequest) -> Result<(), String> {
    let env = client_gateway_env(
        &request.listen,
        &request.tool,
        request.provider_name.as_deref(),
        request.client_model.as_deref(),
        request.client_reasoning_effort.as_deref(),
    );

    // Try common VS Code locations on Windows
    #[cfg(windows)]
    {
        let candidates = [
            "code",
            "code.cmd",
            &format!(
                "{}\\AppData\\Local\\Programs\\Microsoft VS Code\\bin\\code.cmd",
                std::env::var("USERPROFILE").unwrap_or_default()
            ),
        ];
        for cmd in &candidates {
            let mut command = Command::new(cmd);
            command.arg(&request.directory);
            apply_client_env(&mut command, &env);
            if command.spawn().is_ok() {
                return Ok(());
            }
        }
        Err(
            "cannot find VS Code. Please ensure 'code' is in your PATH or VS Code is installed"
                .into(),
        )
    }

    #[cfg(not(windows))]
    {
        let mut command = Command::new("code");
        command.arg(&request.directory);
        apply_client_env(&mut command, &env);
        command
            .spawn()
            .map_err(|e| format!("failed to launch VS Code: {}", e))?;
        Ok(())
    }
}

fn client_gateway_env(
    listen: &str,
    tool: &str,
    provider_name: Option<&str>,
    client_model: Option<&str>,
    client_reasoning_effort: Option<&str>,
) -> Vec<(&'static str, String)> {
    let host = gateway_host(listen);
    let model = normalized_client_model(tool, client_model);
    let client_key = provider_name
        .map(provider_hint_api_key)
        .unwrap_or_else(|| "ferryllm".into());
    match tool {
        "claude" => {
            let mut env = vec![
                ("ANTHROPIC_API_KEY", client_key),
                ("ANTHROPIC_BASE_URL", format!("http://{}", host)),
                ("ANTHROPIC_MODEL", model.clone()),
            (
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "claude-3-5-haiku-latest".into(),
            ),
            ("ANTHROPIC_DEFAULT_SONNET_MODEL", model),
            ("ANTHROPIC_DEFAULT_OPUS_MODEL", "claude-opus-4-1".into()),
            ];
            if let Some(tokens) = claude_thinking_tokens(client_reasoning_effort) {
                env.push(("MAX_THINKING_TOKENS", tokens.to_string()));
            }
            env
        }
        "opencode" => vec![
            ("OPENAI_API_KEY", client_key),
            ("OPENAI_BASE_URL", format!("http://{}/v1", host)),
            (
                "OPENCODE_CONFIG_CONTENT",
                opencode_gateway_config(&host, &model),
            ),
        ],
        _ => vec![
            ("OPENAI_API_KEY", client_key),
            ("OPENAI_BASE_URL", format!("http://{}/v1", host)),
        ],
    }
}

fn provider_hint_api_key(provider_name: &str) -> String {
    format!("ferryllm-provider:{}", hex_encode(provider_name.as_bytes()))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn opencode_gateway_config(host: &str, model: &str) -> String {
    let model_limit = serde_json::json!({
        "limit": { "context": 200000, "output": 8192 }
    });
    let mut models = serde_json::Map::new();
    models.insert(
        model.to_string(),
        serde_json::json!({
            "name": format!("{} via ferryllm", model),
            "limit": { "context": 200000, "output": 8192 }
        }),
    );
    for (id, name) in [
        ("claude-sonnet-4-5", "Claude Sonnet 4.5 via ferryllm"),
        ("gpt-5.4", "GPT 5.4 via ferryllm"),
        ("gpt-5.5", "GPT 5.5 via ferryllm"),
    ] {
        models
            .entry(id.to_string())
            .or_insert_with(|| serde_json::json!({ "name": name, "limit": model_limit["limit"].clone() }));
    }
    serde_json::json!({
        "$schema": "https://opencode.ai/config.json",
        "enabled_providers": ["ferryllm"],
        "provider": {
            "ferryllm": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "ferryllm",
                "options": {
                    "baseURL": format!("http://{}/v1", host),
                    "apiKey": "ferryllm"
                },
                "models": models
            }
        },
        "model": format!("ferryllm/{model}"),
        "small_model": format!("ferryllm/{model}")
    })
    .to_string()
}

fn apply_client_env(command: &mut Command, env: &[(&'static str, String)]) {
    command.env_remove("ANTHROPIC_API_KEY");
    command.env_remove("ANTHROPIC_AUTH_TOKEN");
    command.env_remove("ANTHROPIC_BASE_URL");
    command.env_remove("ANTHROPIC_MODEL");
    command.env_remove("ANTHROPIC_DEFAULT_HAIKU_MODEL");
    command.env_remove("ANTHROPIC_DEFAULT_SONNET_MODEL");
    command.env_remove("ANTHROPIC_DEFAULT_OPUS_MODEL");
    command.env_remove("MAX_THINKING_TOKENS");
    command.env_remove("CLAUDE_CODE_USE_BEDROCK");
    command.env_remove("CLAUDE_CODE_USE_VERTEX");
    command.env_remove("OPENAI_API_KEY");
    command.env_remove("OPENAI_BASE_URL");
    command.env_remove("OPENCODE_CONFIG");
    command.env_remove("OPENCODE_CONFIG_CONTENT");
    command.env_remove("OPENCODE_MODEL");
    for (key, value) in env {
        command.env(key, value);
    }
}

#[tauri::command]
async fn list_config_files() -> Result<Vec<ConfigFile>, String> {
    spawn_blocking_result(move || {
        let mut files = Vec::new();
        let examples_dir = repo_root()?.join("examples").join("config");
        if examples_dir.exists() {
            for entry in fs::read_dir(&examples_dir).map_err(string_error)? {
                let entry = entry.map_err(string_error)?;
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
                    files.push(ConfigFile {
                        name: path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("config.toml")
                            .to_string(),
                        path: path_to_string(&path),
                    });
                }
            }
        }
        files.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(files)
    })
    .await
}

#[tauri::command]
async fn discover_ferryllm(app: AppHandle) -> Result<String, String> {
    spawn_blocking_result(move || discover_ferryllm_inner(app)).await
}

fn discover_ferryllm_inner(app: AppHandle) -> Result<String, String> {
    // 1. Check for bundled sidecar in resource directory
    let resource_dir = app.path().resource_dir().map_err(string_error)?;
    let exe_name = if cfg!(windows) {
        "ferryllm-x86_64-pc-windows-msvc.exe"
    } else {
        "ferryllm-x86_64-unknown-linux-gnu"
    };
    let bundled = resource_dir.join(exe_name);
    if bundled.exists() {
        return Ok(path_to_string(&bundled));
    }

    // 2. Check local target/debug directory first in dev mode so the desktop
    // shell does not accidentally launch an older ferryllm found on PATH.
    let exe = if cfg!(windows) {
        "ferryllm.exe"
    } else {
        "ferryllm"
    };
    let local = repo_root()?.join("target").join("debug").join(exe);
    if local.exists() {
        return Ok(path_to_string(&local));
    }

    // 3. Check system PATH
    if let Some(path) = find_in_path("ferryllm") {
        return Ok(path_to_string(&path));
    }

    Ok("ferryllm".into())
}

#[tauri::command]
async fn read_config_file(path: String) -> Result<ConfigDocument, String> {
    spawn_blocking_result(move || {
        let raw = fs::read_to_string(&path).map_err(string_error)?;
        let value: toml::Value = toml::from_str(&raw).map_err(string_error)?;
        let config = serde_json::to_value(value).map_err(string_error)?;
        Ok(ConfigDocument { path, raw, config })
    })
    .await
}

#[tauri::command]
async fn save_config_file(
    app: AppHandle,
    state: State<'_, DesktopState>,
    request: SaveConfigRequest,
) -> Result<SaveResult, String> {
    let executable = normalized_executable(request.executable.as_deref());
    let runtime = Arc::clone(&state.runtime);
    spawn_blocking_result(move || {
        let path = normalized_config_path(&request.path)?;
        let config = normalize_runtime_config(request.config);
        let validation = validate_config_value_with(&executable, &path, &config)?;

        let mut reloaded = false;

        if validation.ok {
            write_config_atomically(&path, &config)?;

            if request.hot_reload && is_running(&runtime) {
                restart_with(&app, &runtime, executable, path_to_string(&path), false)?;
                reloaded = true;
            }
        }

        Ok(SaveResult {
            path: path_to_string(&path),
            validation,
            reloaded,
        })
    })
    .await
}

#[tauri::command]
async fn validate_config_file(executable: Option<String>, config_path: String) -> CommandResult {
    tauri::async_runtime::spawn_blocking(move || {
        validate_with(&normalized_executable(executable.as_deref()), &config_path)
    })
    .await
    .unwrap_or_else(|err| CommandResult {
        ok: false,
        code: None,
        stdout: String::new(),
        stderr: format!("failed to join validation task: {}", err),
    })
}

#[tauri::command]
async fn validate_config_document(request: ValidateConfigRequest) -> CommandResult {
    tauri::async_runtime::spawn_blocking(move || {
        let executable = normalized_executable(request.executable.as_deref());
        match default_config_toml_path()
            .and_then(|path| validate_config_value_with(&executable, &path, &request.config))
        {
            Ok(result) => result,
            Err(err) => CommandResult {
                ok: false,
                code: None,
                stdout: String::new(),
                stderr: err,
            },
        }
    })
    .await
    .unwrap_or_else(|err| CommandResult {
        ok: false,
        code: None,
        stdout: String::new(),
        stderr: format!("failed to join validation task: {}", err),
    })
}

#[tauri::command]
async fn start_server(
    app: AppHandle,
    state: State<'_, DesktopState>,
    request: ServerRequest,
) -> Result<ProcessStatus, String> {
    let runtime = Arc::clone(&state.runtime);
    spawn_blocking_result(move || {
        start_with(
            &app,
            &runtime,
            normalized_executable(request.executable.as_deref()),
            request.config_path,
            request.replace_existing,
        )?;
        status_inner(&runtime)
    })
    .await
}

#[tauri::command]
async fn stop_server(state: State<'_, DesktopState>) -> Result<ProcessStatus, String> {
    let runtime = Arc::clone(&state.runtime);
    spawn_blocking_result(move || {
        stop_inner(&runtime)?;
        status_inner(&runtime)
    })
    .await
}

#[tauri::command]
async fn restart_server(
    app: AppHandle,
    state: State<'_, DesktopState>,
    request: ServerRequest,
) -> Result<ProcessStatus, String> {
    let runtime = Arc::clone(&state.runtime);
    spawn_blocking_result(move || {
        restart_with(
            &app,
            &runtime,
            normalized_executable(request.executable.as_deref()),
            request.config_path,
            true,
        )?;
        status_inner(&runtime)
    })
    .await
}

#[tauri::command]
async fn server_status(state: State<'_, DesktopState>) -> Result<ProcessStatus, String> {
    let runtime = Arc::clone(&state.runtime);
    spawn_blocking_result(move || status_inner(&runtime)).await
}

#[tauri::command]
async fn fetch_server_snapshot(
    state: State<'_, DesktopState>,
    listen: String,
) -> Result<ServerSnapshot, String> {
    let runtime = Arc::clone(&state.runtime);
    spawn_blocking_result(move || {
        let status = status_inner(&runtime)?;
        let base = dashboard_base_host(&listen);
        Ok(ServerSnapshot {
            status,
            health: http_probe(&base, "/health"),
            ready: http_probe(&base, "/readyz"),
            metrics: http_probe(&base, "/metrics"),
            models: http_probe(&base, "/v1/models"),
            fetched_at_ms: unix_ms(),
        })
    })
    .await
}

#[tauri::command]
async fn probe_provider(request: ProviderProbeRequest) -> Result<ProbeResult, String> {
    spawn_blocking_result(move || probe_provider_inner(request)).await
}

fn start_with(
    app: &AppHandle,
    runtime: &Arc<Mutex<RuntimeState>>,
    executable: String,
    config_path: String,
    replace_existing: bool,
) -> Result<(), String> {
    {
        let mut runtime_state = runtime.lock().map_err(lock_error)?;
        reap_if_exited(&mut runtime_state);
        if runtime_state.child.is_some() {
            return Err("ferryllm is already running".into());
        }
    }

    if gateway_health_ok_for_config(&config_path) {
        if replace_existing {
            stop_external_gateways_for_config(&config_path)?;
        } else {
            let mut runtime_state = runtime.lock().map_err(lock_error)?;
            runtime_state.executable = executable;
            runtime_state.config_path = config_path.clone();
            push_log(
                &mut runtime_state,
                "system",
                format!("detected existing ferryllm on {}", listen_for_config_path(&config_path)),
            );
            return Ok(());
        }
    }

    let mut child = Command::new(&executable)
        .arg("serve")
        .arg("--config")
        .arg(&config_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("failed to start '{}': {}", executable, err))?;

    if let Some(stdout) = child.stdout.take() {
        spawn_log_reader(app.clone(), "stdout", stdout);
    }
    if let Some(stderr) = child.stderr.take() {
        spawn_log_reader(app.clone(), "stderr", stderr);
    }

    let mut runtime_state = runtime.lock().map_err(lock_error)?;
    push_log(
        &mut runtime_state,
        "system",
        format!("started ferryllm with config {}", config_path),
    );
    runtime_state.executable = executable;
    runtime_state.config_path = config_path;
    runtime_state.child = Some(child);
    Ok(())
}

fn stop_inner(runtime: &Arc<Mutex<RuntimeState>>) -> Result<(), String> {
    let mut runtime_state = runtime.lock().map_err(lock_error)?;
    if let Some(mut child) = runtime_state.child.take() {
        let _ = child.kill();
        wait_for_exit(&mut child, Duration::from_secs(3))?;
        push_log(&mut runtime_state, "system", "stopped ferryllm");
    }
    Ok(())
}

fn restart_with(
    app: &AppHandle,
    runtime: &Arc<Mutex<RuntimeState>>,
    executable: String,
    config_path: String,
    replace_existing: bool,
) -> Result<(), String> {
    stop_inner(runtime)?;
    start_with(app, runtime, executable, config_path, replace_existing)
}

fn status_inner(runtime: &Arc<Mutex<RuntimeState>>) -> Result<ProcessStatus, String> {
    let (managed, pid, executable, config_path, logs) = {
        let mut runtime_state = runtime.lock().map_err(lock_error)?;
        reap_if_exited(&mut runtime_state);
        let pid = runtime_state.child.as_ref().map(Child::id);
        let managed = runtime_state.child.is_some();
        let config_path = effective_config_path(&runtime_state.config_path);
        (
            managed,
            pid,
            runtime_state.executable.clone(),
            config_path,
            runtime_state.logs.iter().cloned().collect(),
        )
    };
    let external = !managed && gateway_health_ok_for_config(&config_path);
    let source = if managed {
        "managed"
    } else if external {
        "external"
    } else {
        "stopped"
    };
    Ok(ProcessStatus {
        running: managed || external,
        managed,
        source: source.into(),
        executable,
        config_path,
        pid,
        logs,
    })
}

fn validate_with(executable: &str, config_path: &str) -> CommandResult {
    match Command::new(executable)
        .arg("check-config")
        .arg("--config")
        .arg(config_path)
        .output()
    {
        Ok(output) => CommandResult {
            ok: output.status.success(),
            code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        },
        Err(err) => CommandResult {
            ok: false,
            code: None,
            stdout: String::new(),
            stderr: format!("failed to run '{}': {}", executable, err),
        },
    }
}

fn validate_config_value_with(
    executable: &str,
    target_path: &Path,
    config: &serde_json::Value,
) -> Result<CommandResult, String> {
    let temp_path = temp_config_path(target_path);
    write_config_atomically(&temp_path, config)?;
    let result = validate_with(executable, &path_to_string(&temp_path));
    let _ = fs::remove_file(&temp_path);
    Ok(result)
}

fn normalize_runtime_config(mut config: serde_json::Value) -> serde_json::Value {
    if let Some(providers) = config
        .get_mut("providers")
        .and_then(|value| value.as_array_mut())
    {
        for provider in providers {
            if let Some(base_url) = provider
                .get_mut("base_url")
                .and_then(|value| value.as_str())
            {
                let trimmed = base_url.trim_end_matches('/').to_string();
                if trimmed != base_url {
                    if let Some(object) = provider.as_object_mut() {
                        object.insert("base_url".into(), serde_json::Value::String(trimmed));
                    }
                }
            }
        }
    }

    prune_unavailable_env_providers(&mut config);

    if let Some(routes) = config.get_mut("routes").and_then(|value| value.as_array_mut()) {
        for route in routes {
            if let Some(object) = route.as_object_mut() {
                object.remove("client_kind");
            }
        }
    }

    let has_routes = config
        .get("routes")
        .and_then(|value| value.as_array())
        .map(|routes| !routes.is_empty())
        .unwrap_or(false);
    if !has_routes {
        let only_provider = config
            .get("providers")
            .and_then(|value| value.as_array())
            .and_then(|providers| {
                if providers.len() == 1 {
                    let provider = &providers[0];
                    let name = provider
                        .get("name")
                        .and_then(|value| value.as_str())
                        .filter(|name| !name.trim().is_empty())
                        .map(str::to_string)?;
                    let base_url = provider
                        .get("base_url")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string();
                    Some((name, base_url))
                } else {
                    None
                }
            });
        if let Some((provider_name, _base_url)) = only_provider {
            if let Some(object) = config.as_object_mut() {
                object.insert(
                    "routes".into(),
                    serde_json::json!([
                        {
                            "match": "*",
                            "match_type": "prefix",
                            "provider": provider_name,
                        }
                    ]),
                );
            }
        }
    }

    config
}

fn prune_unavailable_env_providers(config: &mut serde_json::Value) {
    let Some(providers) = config
        .get_mut("providers")
        .and_then(|value| value.as_array_mut())
    else {
        return;
    };

    let mut kept_names = HashSet::new();
    providers.retain(|provider| {
        let name = provider
            .get("name")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let keep = provider_env_key_available(provider);
        if keep && !name.is_empty() {
            kept_names.insert(name);
        }
        keep
    });

    if let Some(routes) = config.get_mut("routes").and_then(|value| value.as_array_mut()) {
        routes.retain_mut(|route| {
            let Some(object) = route.as_object_mut() else {
                return false;
            };
            let provider_ok = object
                .get("provider")
                .and_then(|value| value.as_str())
                .is_some_and(|provider| kept_names.contains(provider));
            if !provider_ok {
                return false;
            }
            if let Some(fallbacks) = object
                .get_mut("fallback_providers")
                .and_then(|value| value.as_array_mut())
            {
                fallbacks.retain(|fallback| {
                    fallback
                        .as_str()
                        .is_some_and(|provider| kept_names.contains(provider))
                });
            }
            true
        });
    }
}

fn provider_env_key_available(provider: &serde_json::Value) -> bool {
    let Some(env) = provider.get("api_key_env").and_then(|value| value.as_str()) else {
        return true;
    };
    let env = env.trim();
    if env.is_empty() {
        return false;
    }
    if provider
        .get("api_key")
        .is_some_and(json_value_is_present)
        || provider
            .get("api_key_url")
            .is_some_and(json_value_is_present)
        || provider
            .get("api_key_file")
            .is_some_and(json_value_is_present)
        || provider.get("key_watch").is_some_and(json_value_is_present)
    {
        return true;
    }
    std::env::var(env)
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
}

fn is_running(runtime: &Arc<Mutex<RuntimeState>>) -> bool {
    runtime
        .lock()
        .map(|mut runtime_state| {
            reap_if_exited(&mut runtime_state);
            runtime_state.child.is_some()
        })
        .unwrap_or(false)
}

fn spawn_log_reader<R>(app: AppHandle, stream: &'static str, reader: R)
where
    R: std::io::Read + Send + 'static,
{
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            let entry = LogEntry {
                ts_ms: unix_ms(),
                stream: stream.to_string(),
                line: strip_ansi(&line),
            };
            if let Some(state) = app.try_state::<DesktopState>() {
                if let Ok(mut runtime) = state.runtime.lock() {
                    push_entry(&mut runtime, entry.clone());
                }
            }
            let _ = app.emit("server-log", entry);
        }
    });
}

fn push_log(runtime: &mut RuntimeState, stream: &str, line: impl Into<String>) {
    let line = line.into();
    push_entry(
        runtime,
        LogEntry {
            ts_ms: unix_ms(),
            stream: stream.to_string(),
            line: strip_ansi(&line),
        },
    );
}

fn strip_ansi(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if ('@'..='~').contains(&next) {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn push_entry(runtime: &mut RuntimeState, entry: LogEntry) {
    if runtime.logs.len() >= MAX_LOG_LINES {
        runtime.logs.pop_front();
    }
    runtime.logs.push_back(entry);
}

fn reap_if_exited(runtime: &mut RuntimeState) {
    if let Some(child) = runtime.child.as_mut() {
        if matches!(child.try_wait(), Ok(Some(_))) {
            runtime.child = None;
        }
    }
}

fn normalized_executable(value: Option<&str>) -> String {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("ferryllm")
        .to_string()
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;
    let candidates = if cfg!(windows) {
        vec![format!("{name}.exe"), name.to_string()]
    } else {
        vec![name.to_string()]
    };

    for dir in env::split_paths(&paths) {
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn repo_root() -> Result<PathBuf, String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "failed to resolve repository root".to_string())
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

fn default_config_toml_path() -> Result<PathBuf, String> {
    Ok(dirs::config_dir()
        .ok_or("cannot find config directory")?
        .join("ferryllm")
        .join("config.toml"))
}

fn normalized_config_path(path: &str) -> Result<PathBuf, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        default_config_toml_path()
    } else {
        Ok(PathBuf::from(trimmed))
    }
}

fn temp_config_path(target_path: &Path) -> PathBuf {
    let mut temp_path = target_path.to_path_buf();
    let file_name = target_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config.toml");
    temp_path.set_file_name(format!("{file_name}.{}.tmp", unix_ms()));
    temp_path
}

fn write_config_atomically(path: &Path, config: &serde_json::Value) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(string_error)?;
    }

    let temp_path = temp_config_path(path);
    let toml = toml::to_string_pretty(config).map_err(string_error)?;
    fs::write(&temp_path, toml).map_err(string_error)?;

    if path.exists() {
        let backup_path = backup_config_path(path);
        if backup_path.exists() {
            fs::remove_file(&backup_path).map_err(string_error)?;
        }
        fs::rename(path, &backup_path).map_err(string_error)?;
        if let Err(err) = fs::rename(&temp_path, path) {
            let _ = fs::rename(&backup_path, path);
            let _ = fs::remove_file(&temp_path);
            return Err(err.to_string());
        }
        let _ = fs::remove_file(&backup_path);
    } else if let Err(err) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(err.to_string());
    }

    Ok(())
}

fn backup_config_path(path: &Path) -> PathBuf {
    let mut backup_path = path.to_path_buf();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config.toml");
    backup_path.set_file_name(format!("{file_name}.bak"));
    backup_path
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> Result<(), String> {
    let started_at = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) if started_at.elapsed() >= timeout => {
                return Err(format!(
                    "process did not exit within {}ms after kill",
                    timeout.as_millis()
                ));
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(err) => return Err(err.to_string()),
        }
    }
}

fn effective_config_path(config_path: &str) -> String {
    let config_path = config_path.trim();
    if !config_path.is_empty() {
        return config_path.into();
    }
    default_config_toml_path()
        .map(|path| path_to_string(&path))
        .unwrap_or_default()
}

fn listen_for_config_path(config_path: &str) -> String {
    config_listen_from_path(config_path).unwrap_or_else(|| "127.0.0.1:3000".into())
}

fn config_listen_from_path(config_path: &str) -> Option<String> {
    let path = effective_config_path(config_path);
    let raw = fs::read_to_string(path).ok()?;
    let value: toml::Value = toml::from_str(&raw).ok()?;
    value
        .get("server")
        .and_then(|server| server.get("listen"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|listen| !listen.is_empty())
        .map(str::to_string)
}

fn gateway_health_ok_for_config(config_path: &str) -> bool {
    let base = dashboard_base_host(&listen_for_config_path(config_path));
    http_probe(&base, "/health").ok
}

fn stop_external_gateways_for_config(config_path: &str) -> Result<(), String> {
    let listen = listen_for_config_path(config_path);
    let base = dashboard_base_host(&listen);
    let (_, port) = parse_host_port(&base)?;
    let pids = listening_pids(port)?;
    let current_pid = std::process::id();
    let mut stopped = Vec::new();

    for pid in pids.into_iter().filter(|pid| *pid != current_pid) {
        let command_line = process_command_line(pid).unwrap_or_default();
        let lower = command_line.to_lowercase();
        if !(lower.contains("ferryllm") && lower.contains("serve"))
            || lower.contains("ferryllm-desktop")
        {
            return Err(format!(
                "listen port {} is occupied by non-ferryllm process PID {}",
                port, pid
            ));
        }
        terminate_process_tree(pid)?;
        stopped.push(pid);
    }

    if stopped.is_empty() {
        return Err(format!(
            "listen port {} responded as a gateway, but no ferryllm serve process was found",
            port
        ));
    }

    let started_at = Instant::now();
    while gateway_health_ok_for_config(config_path) {
        if started_at.elapsed() >= Duration::from_secs(5) {
            return Err(format!(
                "external ferryllm on {} did not stop after replacing PIDs {:?}",
                listen, stopped
            ));
        }
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

#[cfg(windows)]
fn listening_pids(port: u16) -> Result<Vec<u32>, String> {
    let script = format!(
        "Get-NetTCPConnection -LocalPort {} -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess",
        port
    );
    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .map_err(string_error)?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(parse_pid_lines(&String::from_utf8_lossy(&output.stdout)))
}

#[cfg(not(windows))]
fn listening_pids(port: u16) -> Result<Vec<u32>, String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(format!("lsof -tiTCP:{} -sTCP:LISTEN 2>/dev/null || true", port))
        .output()
        .map_err(string_error)?;
    Ok(parse_pid_lines(&String::from_utf8_lossy(&output.stdout)))
}

fn parse_pid_lines(text: &str) -> Vec<u32> {
    text.lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect()
}

#[cfg(windows)]
fn process_command_line(pid: u32) -> Result<String, String> {
    let script = format!(
        "(Get-CimInstance Win32_Process -Filter \"ProcessId={}\").CommandLine",
        pid
    );
    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .map_err(string_error)?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

#[cfg(not(windows))]
fn process_command_line(pid: u32) -> Result<String, String> {
    let output = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
        .map_err(string_error)?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

#[cfg(windows)]
fn terminate_process_tree(pid: u32) -> Result<(), String> {
    let output = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output()
        .map_err(string_error)?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

#[cfg(not(windows))]
fn terminate_process_tree(pid: u32) -> Result<(), String> {
    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .map_err(string_error)?;
    if !status.success() {
        return Err(format!("failed to terminate PID {}", pid));
    }
    thread::sleep(Duration::from_millis(500));
    let _ = Command::new("kill")
        .args(["-KILL", &pid.to_string()])
        .status();
    Ok(())
}

fn dashboard_base_host(listen: &str) -> String {
    let listen = listen.trim();
    if listen.is_empty() {
        return "127.0.0.1:3000".into();
    }
    if listen.starts_with("0.0.0.0") {
        listen.replacen("0.0.0.0", "127.0.0.1", 1)
    } else {
        listen.to_string()
    }
}

fn http_probe(base: &str, path: &str) -> ProbeResult {
    match http_get_local(base, path) {
        Ok((status, body)) => ProbeResult {
            ok: (200..300).contains(&status),
            status: Some(status),
            body,
            error: None,
        },
        Err(error) => ProbeResult {
            ok: false,
            status: None,
            body: String::new(),
            error: Some(error),
        },
    }
}

fn probe_provider_inner(request: ProviderProbeRequest) -> Result<ProbeResult, String> {
    let key = resolve_probe_api_key(&request);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .map_err(string_error)?;
    let paths: Vec<&str> = if request.mode == "usage" {
        vec![
            "/dashboard/billing/credit_grants",
            "/v1/dashboard/billing/credit_grants",
            "/api/user/self",
            "/api/token/self",
        ]
    } else if request.provider_type == "anthropic" {
        vec!["/v1/models", "/"]
    } else {
        vec!["/v1/models", "/models", "/"]
    };

    let mut last_error = None::<String>;
    for path in paths {
        let url = format!("{}{}", trim_url(&request.base_url), path);
        let mut req = client.get(&url);
        if let Some(key) = &key {
            if request.provider_type == "anthropic" {
                req = req
                    .header("x-api-key", key)
                    .header("anthropic-version", "2023-06-01");
            } else {
                req = req.header("Authorization", format!("Bearer {key}"));
            }
        }
        match req.send() {
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                if status.is_success() {
                    let body = if request.mode == "models" {
                        summarize_models_body(&body)
                    } else {
                        summarize_probe_body(&body)
                    };
                    return Ok(ProbeResult {
                        ok: true,
                        status: Some(status.as_u16()),
                        body,
                        error: None,
                    });
                }
                last_error = Some(format!("{} returned {}", url, status));
                if request.mode != "usage" && status.as_u16() == 401 {
                    break;
                }
            }
            Err(err) => {
                last_error = Some(format!("{}: {}", url, err));
            }
        }
    }

    Ok(ProbeResult {
        ok: false,
        status: None,
        body: String::new(),
        error: Some(
            last_error.unwrap_or_else(|| format!("provider '{}' probe failed", request.name)),
        ),
    })
}

fn resolve_probe_api_key(request: &ProviderProbeRequest) -> Option<String> {
    if let Some(key) = request
        .api_key
        .as_ref()
        .map(String::as_str)
        .map(str::trim)
        .filter(|key| !key.is_empty())
    {
        return Some(key.to_string());
    }
    if let Some(env_name) = request
        .api_key_env
        .as_ref()
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if let Ok(key) = env::var(env_name) {
            return Some(key);
        }
    }
    if let Some(file) = request
        .api_key_file
        .as_ref()
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if let Ok(key) = fs::read_to_string(file) {
            let key = key.trim();
            if !key.is_empty() {
                return Some(key.to_string());
            }
        }
    }
    if let Some(url) = request
        .api_key_url
        .as_ref()
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if let Ok(client) = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
        {
            if let Ok(resp) = client.get(url).send() {
                if resp.status().is_success() {
                    if let Ok(body) = resp.text() {
                        let key = body.trim();
                        if !key.is_empty() {
                            return Some(key.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

fn trim_url(value: &str) -> String {
    value.trim().trim_end_matches('/').to_string()
}

fn summarize_probe_body(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.len() <= 800 {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..800])
    }
}

fn summarize_models_body(body: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return summarize_probe_body(body);
    };
    let items = value
        .get("data")
        .or_else(|| value.get("models"))
        .and_then(|value| value.as_array())
        .cloned()
        .or_else(|| value.as_array().cloned())
        .unwrap_or_default();
    let mut models = Vec::<String>::new();
    for item in items {
        let id = item
            .as_str()
            .map(str::to_string)
            .or_else(|| item.get("id").and_then(|value| value.as_str()).map(str::to_string))
            .or_else(|| {
                item.get("name")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            });
        if let Some(id) = id {
            let id = id.trim();
            if !id.is_empty() && !models.iter().any(|existing| existing == id) {
                models.push(id.to_string());
            }
        }
    }
    models.sort();
    serde_json::json!({ "models": models }).to_string()
}

fn http_get_local(base: &str, path: &str) -> Result<(u16, String), String> {
    let (host, port) = parse_host_port(base)?;
    let mut stream = TcpStream::connect((host.as_str(), port)).map_err(string_error)?;
    let timeout = Some(Duration::from_millis(1200));
    stream.set_read_timeout(timeout).map_err(string_error)?;
    stream.set_write_timeout(timeout).map_err(string_error)?;

    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: */*\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).map_err(string_error)?;

    let mut response = String::new();
    stream.read_to_string(&mut response).map_err(string_error)?;
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| "invalid HTTP response".to_string())?;
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or_else(|| "missing HTTP status".to_string())?;
    Ok((status, body.to_string()))
}

fn parse_host_port(base: &str) -> Result<(String, u16), String> {
    let without_scheme = base
        .strip_prefix("http://")
        .or_else(|| base.strip_prefix("https://"))
        .unwrap_or(base);
    let host_port = without_scheme.split('/').next().unwrap_or(without_scheme);
    let (host, port) = host_port
        .rsplit_once(':')
        .ok_or_else(|| format!("listen address '{base}' must include a port"))?;
    let port = port
        .parse::<u16>()
        .map_err(|_| format!("invalid listen port in '{base}'"))?;
    let host = host.trim_matches(['[', ']']).to_string();
    if host.is_empty() {
        return Err(format!("invalid listen host in '{base}'"));
    }
    Ok((host, port))
}

fn unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn lock_error<T>(_: std::sync::PoisonError<T>) -> String {
    "desktop state lock is poisoned".into()
}

fn string_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

async fn spawn_blocking_result<T>(
    task: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String>
where
    T: Send + 'static,
{
    tauri::async_runtime::spawn_blocking(task)
        .await
        .map_err(|err| format!("failed to join background task: {}", err))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(DesktopState {
            runtime: Arc::new(Mutex::new(RuntimeState::default())),
        })
        .invoke_handler(tauri::generate_handler![
            list_config_files,
            discover_ferryllm,
            read_config_file,
            save_config_file,
            validate_config_file,
            validate_config_document,
            start_server,
            stop_server,
            restart_server,
            server_status,
            fetch_server_snapshot,
            probe_provider,
            scan_ai_sessions,
            resume_ai_session,
            create_workspace,
            delete_workspace,
            delete_ai_session,
            reveal_workspace,
            launch_cli,
            launch_vscode,
            write_config_to_default,
            save_launcher_state,
            load_launcher_state,
            save_config_to_default,
            load_config_from_default
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_launch_args_force_local_ferryllm_provider() {
        let args = launch_args("codex", "127.0.0.1:3000", Some("gpt-5.4--ferryllm-test"), None);

        assert!(args.contains(&"model_provider=ferryllm".to_string()));
        assert!(args.contains(
            &"model_providers.ferryllm.base_url=http://127.0.0.1:3000/v1".to_string()
        ));
        assert!(args
            .windows(2)
            .any(|pair| pair[0] == "--model" && pair[1] == "gpt-5.4--ferryllm-test"));
    }

    #[test]
    fn gateway_host_normalizes_wildcard_listen_address() {
        let args = codex_gateway_config_args("0.0.0.0:3000");

        assert!(args.contains(
            &"model_providers.ferryllm.base_url=http://127.0.0.1:3000/v1".to_string()
        ));
    }
}

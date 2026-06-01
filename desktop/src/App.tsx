import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  CirclePlus,
  Cloud,
  Copy,
  Code2,
  Database,
  Download,
  DollarSign,
  Eye,
  EyeOff,
  FileCog,
  FolderOpen,
  Gauge,
  GripVertical,
  HardDriveDownload,
  History,
  KeyRound,
  Lock,
  Monitor,
  Moon,
  MoreVertical,
  Network,
  Play,
  Plus,
  RefreshCw,
  Route,
  RotateCw,
  Save,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Square,
  Star,
  Sun,
  Terminal,
  Trash2,
  Upload,
  Wrench,
  Zap,
} from "lucide-react";
import { cn } from "./lib/utils";
import "./App.css";

/* ── Types ─────────────────────────────────────────────────────────── */

type JsonValue = null | boolean | number | string | JsonValue[] | JsonObject;
type JsonObject = { [key: string]: JsonValue };

type ProviderConfig = {
  name: string;
  type: string;
  base_url: string;
  api_key?: string;
  api_key_env?: string;
  api_key_url?: string;
  api_key_file?: string;
  key_watch?: KeyWatchConfig[];
};
type KeyWatchConfig = { file: string; path: string; url_path?: string };
type ProviderPresetCategory = "official" | "aggregator" | "china" | "local";
type ProviderPreset = ProviderConfig & {
  description: string;
  routes?: RouteConfig[];
  category: ProviderPresetCategory;
  featured?: boolean;
};
type ProviderRuntimeRule = {
  markers: string[];
  clientModelRewrites?: Record<string, string>;
};

const DEFAULT_CLIENT_MODEL = "claude-sonnet-4-5";
const DEFAULT_OPENAI_CLIENT_MODEL = "gpt-5.4";

type RouteConfig = {
  match: string;
  match_type?: "prefix" | "exact";
  provider: string;
  fallback_providers?: string[];
  rewrite_model?: string;
  client_kind?: "claude" | "openai";
};

type AppConfig = {
  server?: JsonObject;
  logging?: JsonObject;
  auth?: JsonObject;
  metrics?: JsonObject;
  prompt_cache?: JsonObject;
  providers?: ProviderConfig[];
  routes?: RouteConfig[];
};

type CommandResult = { ok: boolean; code: number | null; stdout: string; stderr: string };
type SaveResult = { path: string; validation: CommandResult; reloaded: boolean };
type LogEntry = { ts_ms: number; stream: string; line: string };
type ProcessStatus = {
  running: boolean;
  managed: boolean;
  source: "managed" | "external" | "stopped";
  executable: string;
  config_path: string;
  pid: number | null;
  logs: LogEntry[];
};
type ProbeResult = { ok: boolean; status: number | null; body: string; error: string | null };
type ServerSnapshot = {
  status: ProcessStatus;
  health: ProbeResult;
  ready: ProbeResult;
  metrics: ProbeResult;
  models: ProbeResult;
  fetched_at_ms: number;
};

function gatewayStatusLabel(status?: ProcessStatus | null) {
  if (!status?.running) return "Stopped";
  return status.source === "external" ? "External" : "Running";
}

function gatewayStatusTone(status?: ProcessStatus | null): "success" | "warning" | "muted" {
  if (!status?.running) return "warning";
  return status.source === "external" ? "warning" : "success";
}

function gatewayStatusDetail(status?: ProcessStatus | null) {
  if (!status?.running) return "Process is stopped";
  if (status.source === "external") return "Detected on listen address";
  return status.pid ? `PID ${status.pid}` : "Managed process";
}

function commandResultMessage(result: CommandResult) {
  return (result.stderr || result.stdout || "Config validation failed").trim();
}
type AISession = {
  id: string;
  tool: AITool;
  cwd: string;
  path: string;
  title: string;
  preview: string;
  message_count: number;
  updated_at_ms: number;
};
type ModelMetric = {
  provider: string;
  model: string;
  requests: number;
  errors: number;
  latencyMicros: number;
  tokens: number;
};
type MetricSample = {
  ts: number;
  values: Record<string, number>;
  models: ModelMetric[];
};

type View = "dashboard" | "usage-logs" | "providers" | "provider-detail" | "launcher" | "settings";
type DetailTab = "settings" | "models" | "routes" | "runtime";
type SettingsTab = "general" | "runtime" | "security" | "cache" | "advanced" | "about";
type PresetCategoryFilter = "all" | ProviderPresetCategory;
type AITool = "codex" | "claude" | "opencode";
type ReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "ultracode";
const REASONING_OPTIONS = ["", "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultracode"];

type RecentLaunch = {
  id: string;
  directory: string;
  tool: AITool;
  launchType: "cli" | "vscode";
  providerName: string;
  providerType: string;
  reasoningEffort?: ReasoningEffort;
  configPath: string;
  lastUsed: number;
};

type LaunchTarget = "cli" | "vscode";
type WorkspaceBinding = {
  providerName: string;
  providerType: string;
  tool: AITool;
  launchType: LaunchTarget;
  reasoningEffort?: ReasoningEffort;
  updatedAt: number;
};
type WorkspaceEntry = {
  path: string;
  name: string;
  pinned: boolean;
  lastUsed: number;
  defaultTool: AITool;
  defaultLaunchType: LaunchTarget;
  defaultReasoningEffort?: ReasoningEffort;
  defaultProviderName: string;
  defaultProviderType: string;
  sessions: RecentLaunch[];
};

/* ── Toast ─────────────────────────────────────────────────────────── */

type Toast = { id: number; message: string; type: "info" | "success" | "warning" | "error" };

function isTauriRuntime() {
  return Boolean((window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
}

function ToastContainer({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
      <AnimatePresence>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className={cn(
              "flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium shadow-lg backdrop-blur-sm",
              "border border-border bg-surface/95 text-fg",
              t.type === "success" && "border-success/40 bg-success-soft",
              t.type === "warning" && "border-accent/40 bg-accent-soft",
              t.type === "error" && "border-danger/40 bg-danger-soft"
            )}
            onClick={() => onDismiss(t.id)}
          >
            {t.type === "success" && <CheckCircle2 size={16} className="text-success shrink-0" />}
            {t.type === "warning" && <AlertTriangle size={16} className="text-accent shrink-0" />}
            {t.type === "error" && <Zap size={16} className="text-danger shrink-0" />}
            <span>{t.message}</span>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

/* ── Helpers ────────────────────────────────────────────────────────── */

const emptyConfig: AppConfig = {
  server: { listen: "127.0.0.1:3000", request_timeout_secs: 120, body_limit_mb: 32 },
  logging: { level: "info", format: "text", ansi: false },
  auth: { enabled: false },
  metrics: { enabled: true },
  prompt_cache: {
    auto_inject_anthropic_cache_control: true,
    cache_system: true,
    cache_tools: true,
    cache_last_user_message: true,
    openai_prompt_cache_key: "ferryllm",
    debug_log_request_shape: true,
  },
  providers: [],
  routes: [],
};

function cloneConfig(c: AppConfig): AppConfig { return JSON.parse(JSON.stringify(c)); }
function valueAsString(v: JsonValue | undefined): string { return v == null ? "" : String(v); }
function valueAsNumber(v: JsonValue | undefined): string { return typeof v === "number" ? String(v) : typeof v === "string" ? v : ""; }
function valueAsBool(v: JsonValue | undefined, fb = false): boolean { return typeof v === "boolean" ? v : fb; }
function splitCsv(v: string): string[] { return v.split(",").map((s) => s.trim()).filter(Boolean); }
const providerPresets: ProviderPreset[] = [
  {
    name: "openai",
    type: "openai",
    base_url: "https://api.openai.com",
    api_key_env: "OPENAI_API_KEY",
    description: "Official OpenAI Chat Completions API",
    category: "official",
    featured: true,
    routes: [{ match: "gpt-", match_type: "prefix", provider: "openai" }],
  },
  {
    name: "anthropic",
    type: "anthropic",
    base_url: "https://api.anthropic.com",
    api_key_env: "ANTHROPIC_API_KEY",
    description: "Official Anthropic Messages API",
    category: "official",
    featured: true,
    routes: [{ match: "claude-", match_type: "prefix", provider: "anthropic" }],
  },
  {
    name: "openrouter",
    type: "openai",
    base_url: "https://openrouter.ai/api",
    api_key_env: "OPENROUTER_API_KEY",
    description: "OpenAI-compatible routing across many public models",
    category: "aggregator",
    featured: true,
    routes: [{ match: "openrouter/", match_type: "prefix", provider: "openrouter" }],
  },
  {
    name: "deepseek",
    type: "openai",
    base_url: "https://api.deepseek.com",
    api_key_env: "DEEPSEEK_API_KEY",
    description: "DeepSeek OpenAI-compatible endpoint",
    category: "china",
    featured: true,
    routes: [{ match: "deepseek-", match_type: "prefix", provider: "deepseek" }],
  },
  {
    name: "kimi",
    type: "anthropic",
    base_url: "https://api.moonshot.ai/anthropic",
    api_key_env: "MOONSHOT_API_KEY",
    description: "Moonshot/Kimi Anthropic-compatible endpoint",
    category: "china",
    routes: [{ match: "kimi-", match_type: "prefix", provider: "kimi" }],
  },
  {
    name: "zai",
    type: "anthropic",
    base_url: "https://api.z.ai/api/anthropic",
    api_key_env: "ZAI_API_KEY",
    description: "z.ai GLM Anthropic-compatible endpoint",
    category: "china",
    routes: [{ match: "glm-", match_type: "prefix", provider: "zai" }],
  },
  {
    name: "oneapi",
    type: "openai",
    base_url: "http://127.0.0.1:3000",
    api_key_env: "ONEAPI_API_KEY",
    description: "Local OneAPI/NewAPI style OpenAI-compatible gateway",
    category: "local",
    featured: true,
    routes: [{ match: "*", match_type: "prefix", provider: "oneapi" }],
  },
  {
    name: "ollama",
    type: "openai",
    base_url: "http://127.0.0.1:11434",
    api_key: "ollama",
    description: "Local Ollama OpenAI-compatible endpoint",
    category: "local",
    routes: [{ match: "ollama/", match_type: "prefix", provider: "ollama" }],
  },
  { name: "azure-openai", type: "openai", base_url: "https://YOUR-RESOURCE.openai.azure.com", api_key_env: "AZURE_OPENAI_API_KEY", description: "Azure OpenAI compatible deployment endpoint", category: "official", routes: [{ match: "azure/", match_type: "prefix", provider: "azure-openai" }] },
  { name: "gemini", type: "gemini", base_url: "https://generativelanguage.googleapis.com", api_key_env: "GEMINI_API_KEY", description: "Official Google Gemini API", category: "official", featured: true, routes: [{ match: "gemini-", match_type: "prefix", provider: "gemini" }] },
  { name: "groq", type: "openai", base_url: "https://api.groq.com/openai", api_key_env: "GROQ_API_KEY", description: "Groq OpenAI-compatible endpoint", category: "aggregator", routes: [{ match: "groq/", match_type: "prefix", provider: "groq" }] },
  { name: "together", type: "openai", base_url: "https://api.together.xyz", api_key_env: "TOGETHER_API_KEY", description: "Together AI OpenAI-compatible endpoint", category: "aggregator", routes: [{ match: "together/", match_type: "prefix", provider: "together" }] },
  { name: "fireworks", type: "openai", base_url: "https://api.fireworks.ai/inference", api_key_env: "FIREWORKS_API_KEY", description: "Fireworks AI OpenAI-compatible endpoint", category: "aggregator", routes: [{ match: "accounts/fireworks/", match_type: "prefix", provider: "fireworks" }] },
  { name: "perplexity", type: "openai", base_url: "https://api.perplexity.ai", api_key_env: "PERPLEXITY_API_KEY", description: "Perplexity OpenAI-compatible endpoint", category: "aggregator", routes: [{ match: "sonar", match_type: "prefix", provider: "perplexity" }] },
  { name: "mistral", type: "openai", base_url: "https://api.mistral.ai", api_key_env: "MISTRAL_API_KEY", description: "Mistral OpenAI-compatible endpoint", category: "official", routes: [{ match: "mistral", match_type: "prefix", provider: "mistral" }] },
  { name: "cohere", type: "openai", base_url: "https://api.cohere.com", api_key_env: "COHERE_API_KEY", description: "Cohere OpenAI-compatible endpoint", category: "official", routes: [{ match: "command-", match_type: "prefix", provider: "cohere" }] },
  { name: "xai", type: "openai", base_url: "https://api.x.ai", api_key_env: "XAI_API_KEY", description: "xAI OpenAI-compatible endpoint", category: "official", routes: [{ match: "grok-", match_type: "prefix", provider: "xai" }] },
  { name: "dashscope", type: "openai", base_url: "https://dashscope.aliyuncs.com/compatible-mode", api_key_env: "DASHSCOPE_API_KEY", description: "Alibaba DashScope OpenAI-compatible endpoint", category: "china", featured: true, routes: [{ match: "qwen", match_type: "prefix", provider: "dashscope" }] },
  { name: "volcengine", type: "openai", base_url: "https://ark.cn-beijing.volces.com/api", api_key_env: "ARK_API_KEY", description: "ByteDance Volcano Ark OpenAI-compatible endpoint", category: "china", routes: [{ match: "doubao", match_type: "prefix", provider: "volcengine" }] },
  { name: "baidu-qianfan", type: "openai", base_url: "https://qianfan.baidubce.com", api_key_env: "QIANFAN_API_KEY", description: "Baidu Qianfan OpenAI-compatible endpoint", category: "china", routes: [{ match: "ernie", match_type: "prefix", provider: "baidu-qianfan" }] },
  { name: "tencent-hunyuan", type: "openai", base_url: "https://api.hunyuan.cloud.tencent.com", api_key_env: "HUNYUAN_API_KEY", description: "Tencent Hunyuan OpenAI-compatible endpoint", category: "china", routes: [{ match: "hunyuan", match_type: "prefix", provider: "tencent-hunyuan" }] },
  { name: "minimax", type: "openai", base_url: "https://api.minimax.chat", api_key_env: "MINIMAX_API_KEY", description: "MiniMax OpenAI-compatible endpoint", category: "china", routes: [{ match: "abab", match_type: "prefix", provider: "minimax" }] },
  { name: "siliconflow", type: "openai", base_url: "https://api.siliconflow.cn", api_key_env: "SILICONFLOW_API_KEY", description: "SiliconFlow OpenAI-compatible endpoint", category: "china", featured: true, routes: [{ match: "Qwen/", match_type: "prefix", provider: "siliconflow" }] },
  { name: "modelscope", type: "openai", base_url: "https://api-inference.modelscope.cn", api_key_env: "MODELSCOPE_API_KEY", description: "ModelScope OpenAI-compatible endpoint", category: "china", routes: [{ match: "modelscope/", match_type: "prefix", provider: "modelscope" }] },
  { name: "baichuan", type: "openai", base_url: "https://api.baichuan-ai.com", api_key_env: "BAICHUAN_API_KEY", description: "Baichuan OpenAI-compatible endpoint", category: "china", routes: [{ match: "baichuan", match_type: "prefix", provider: "baichuan" }] },
  { name: "stepfun", type: "openai", base_url: "https://api.stepfun.com", api_key_env: "STEPFUN_API_KEY", description: "StepFun OpenAI-compatible endpoint", category: "china", routes: [{ match: "step-", match_type: "prefix", provider: "stepfun" }] },
  { name: "aihubmix", type: "openai", base_url: "https://aihubmix.com", api_key_env: "AIHUBMIX_API_KEY", description: "AIHubMix OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "aihubmix" }] },
  { name: "302ai", type: "openai", base_url: "https://api.302.ai", api_key_env: "AI302_API_KEY", description: "302.AI OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "302ai" }] },
  { name: "burncloud", type: "openai", base_url: "https://ai.burncloud.com", api_key_env: "BURNCLOUD_API_KEY", description: "BurnCloud OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "burncloud" }] },
  { name: "shengsuanyun", type: "openai", base_url: "https://router.shengsuanyun.com/api/v1", api_key_env: "SHENGSUANYUN_API_KEY", description: "Shengsuanyun OpenAI-compatible coding gateway", category: "aggregator", featured: true, routes: [{ match: "*", match_type: "prefix", provider: "shengsuanyun" }] },
  { name: "pateway", type: "openai", base_url: "https://api.pateway.ai/v1", api_key_env: "PATEWAY_API_KEY", description: "PatewayAI OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "pateway" }] },
  { name: "doubao-seed", type: "openai", base_url: "https://ark.cn-beijing.volces.com/api/v3", api_key_env: "ARK_API_KEY", description: "ByteDance Doubao Seed coding endpoint", category: "china", featured: true, routes: [{ match: "doubao", match_type: "prefix", provider: "doubao-seed" }] },
  { name: "zhipu-glm", type: "openai", base_url: "https://open.bigmodel.cn/api/paas/v4", api_key_env: "ZHIPU_API_KEY", description: "Zhipu GLM OpenAI-compatible endpoint", category: "china", routes: [{ match: "glm-", match_type: "prefix", provider: "zhipu-glm" }] },
  { name: "bailian", type: "openai", base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1", api_key_env: "BAILIAN_API_KEY", description: "Alibaba Bailian OpenAI-compatible endpoint", category: "china", routes: [{ match: "qwen", match_type: "prefix", provider: "bailian" }] },
  { name: "longcat", type: "openai", base_url: "https://api.longcat.chat/openai/v1", api_key_env: "LONGCAT_API_KEY", description: "LongCat OpenAI-compatible endpoint", category: "china", routes: [{ match: "LongCat", match_type: "prefix", provider: "longcat" }] },
  { name: "xiaomi-mimo", type: "openai", base_url: "https://api.xiaomimimo.com/v1", api_key_env: "MIMO_API_KEY", description: "Xiaomi MiMo OpenAI-compatible endpoint", category: "china", routes: [{ match: "mimo", match_type: "prefix", provider: "xiaomi-mimo" }] },
  { name: "novita", type: "openai", base_url: "https://api.novita.ai/openai/v1", api_key_env: "NOVITA_API_KEY", description: "Novita AI OpenAI-compatible endpoint", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "novita" }] },
  { name: "nvidia", type: "openai", base_url: "https://integrate.api.nvidia.com/v1", api_key_env: "NVIDIA_API_KEY", description: "NVIDIA NIM OpenAI-compatible endpoint", category: "official", routes: [{ match: "nvidia/", match_type: "prefix", provider: "nvidia" }] },
  { name: "dmxapi", type: "openai", base_url: "https://www.dmxapi.cn/v1", api_key_env: "DMXAPI_API_KEY", description: "DMXAPI OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "dmxapi" }] },
  { name: "packycode", type: "openai", base_url: "https://www.packyapi.com/v1", api_key_env: "PACKYCODE_API_KEY", description: "PackyCode OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "packycode" }] },
  { name: "apikeyfun", type: "openai", base_url: "https://api.apikey.fun/v1", api_key_env: "APIKEYFUN_API_KEY", description: "APIKEY.FUN OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "apikeyfun" }] },
  { name: "apinebula", type: "openai", base_url: "https://apinebula.com/v1", api_key_env: "APINEBULA_API_KEY", description: "APINebula OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "apinebula" }] },
  { name: "atlascloud", type: "openai", base_url: "https://api.atlascloud.ai/v1", api_key_env: "ATLASCLOUD_API_KEY", description: "AtlasCloud OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "atlascloud" }] },
  { name: "sudocode", type: "openai", base_url: "https://api.sudocode.ai/v1", api_key_env: "SUDOCODE_API_KEY", description: "SudoCode OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "sudocode" }] },
  { name: "claudecn", type: "openai", base_url: "https://claudecn.top/v1", api_key_env: "CLAUDECN_API_KEY", description: "ClaudeCN OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "claudecn" }] },
  { name: "runapi", type: "openai", base_url: "https://runapi.co/v1", api_key_env: "RUNAPI_API_KEY", description: "RunAPI OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "runapi" }] },
  { name: "relaxycode", type: "openai", base_url: "https://www.relaxycode.com/v1", api_key_env: "RELAXYCODE_API_KEY", description: "RelaxyCode OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "relaxycode" }] },
  { name: "cubence", type: "openai", base_url: "https://api.cubence.com/v1", api_key_env: "CUBENCE_API_KEY", description: "Cubence OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "cubence" }] },
  { name: "aicodemirror", type: "openai", base_url: "https://api.aicodemirror.com/api/codex/backend-api/codex", api_key_env: "AICODEMIRROR_API_KEY", description: "AICodeMirror OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "aicodemirror" }] },
  { name: "crazyrouter", type: "openai", base_url: "https://cn.crazyrouter.com/v1", api_key_env: "CRAZYROUTER_API_KEY", description: "CrazyRouter OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "crazyrouter" }] },
  { name: "sssaicode", type: "openai", base_url: "https://node-hk.sssaicode.com/api/v1", api_key_env: "SSSAICODE_API_KEY", description: "SSSAiCode OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "sssaicode" }] },
  { name: "micu", type: "openai", base_url: "https://www.micuapi.ai/v1", api_key_env: "MICU_API_KEY", description: "Micu OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "micu" }] },
  { name: "ctok", type: "openai", base_url: "https://api.ctok.ai/v1", api_key_env: "CTOK_API_KEY", description: "CTok.ai OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "ctok" }] },
  { name: "lemondata", type: "openai", base_url: "https://api.lemondata.cc/v1", api_key_env: "LEMONDATA_API_KEY", description: "LemonData OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "lemondata" }] },
  { name: "pipellm", type: "openai", base_url: "https://code.pipellm.ai/v1", api_key_env: "PIPELLM_API_KEY", description: "PIPELLM OpenAI-compatible coding gateway", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "pipellm" }] },
  { name: "therouter", type: "openai", base_url: "https://api.therouter.ai/v1", api_key_env: "THEROUTER_API_KEY", description: "TheRouter OpenAI-compatible aggregator", category: "aggregator", routes: [{ match: "*", match_type: "prefix", provider: "therouter" }] },
  { name: "lmstudio", type: "openai", base_url: "http://127.0.0.1:1234", api_key: "lm-studio", description: "Local LM Studio OpenAI-compatible endpoint", category: "local", routes: [{ match: "*", match_type: "prefix", provider: "lmstudio" }] },
  { name: "vllm", type: "openai", base_url: "http://127.0.0.1:8000", api_key: "vllm", description: "Local vLLM OpenAI-compatible endpoint", category: "local", routes: [{ match: "*", match_type: "prefix", provider: "vllm" }] },
  { name: "llama-cpp", type: "openai", base_url: "http://127.0.0.1:8080", api_key: "llama-cpp", description: "Local llama.cpp server endpoint", category: "local", routes: [{ match: "*", match_type: "prefix", provider: "llama-cpp" }] },
];
const MIN_COLLAPSED_PROVIDER_PRESET_COUNT = 8;
const providerRuntimeRules: ProviderRuntimeRule[] = [
  {
    markers: ["dstopology", "api.dstopology.com"],
    clientModelRewrites: {
      "claude-sonnet-4-5": "gpt-5.4",
    },
  },
];

function redactConfigSecrets(c: AppConfig): AppConfig {
  const next = cloneConfig(c);
  next.providers = (next.providers ?? []).map((provider) => {
    const { api_key: _apiKey, ...rest } = provider;
    return rest;
  });
  return next;
}

function normalizeConfig(c: AppConfig): AppConfig {
  return {
    ...emptyConfig,
    ...c,
    server: { ...emptyConfig.server, ...(c.server ?? {}) },
    logging: { ...emptyConfig.logging, ...(c.logging ?? {}) },
    auth: { ...emptyConfig.auth, ...(c.auth ?? {}) },
    metrics: { ...emptyConfig.metrics, ...(c.metrics ?? {}) },
    prompt_cache: { ...emptyConfig.prompt_cache, ...(c.prompt_cache ?? {}) },
    providers: c.providers ?? [],
    routes: c.routes ?? [],
  };
}

function findProviderRuntimeRule(provider: ProviderConfig): ProviderRuntimeRule | undefined {
  const id = `${provider.name} ${provider.base_url}`.toLowerCase();
  return providerRuntimeRules.find((rule) => rule.markers.some((marker) => id.includes(marker)));
}

function runtimeConfigForGateway(c: AppConfig): AppConfig {
  const next = cloneConfig(c);
  next.providers = next.providers ?? [];
  const existingRoutes = next.routes ?? [];
  const aliasRoutes = providerAliasRoutes(next.providers, existingRoutes);
  next.routes = mergeRoutes(aliasRoutes, existingRoutes).map(runtimeRouteConfig);
  return next;
}

function runtimeRouteConfig(route: RouteConfig): RouteConfig {
  return {
    match: route.match,
    match_type: route.match_type,
    provider: route.provider,
    fallback_providers: route.fallback_providers,
    rewrite_model: route.rewrite_model,
  };
}

function stableHash(value: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36);
}

function providerSlug(provider: ProviderConfig): string {
  const slug = provider.name
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  const fingerprint = stableHash(`${provider.name}\0${provider.type}\0${provider.base_url}`);
  return `${slug ? `${slug}-` : ""}${fingerprint}`;
}

function providerAliasModel(provider: ProviderConfig | undefined, baseModel = DEFAULT_CLIENT_MODEL): string {
  if (!provider) return baseModel;
  return `${baseModel}--ferryllm-${providerSlug(provider)}`;
}

function providerAliasTarget(provider: ProviderConfig, routes: RouteConfig[], baseModel = DEFAULT_CLIENT_MODEL): string {
  const rewrites = provider ? findProviderRuntimeRule(provider)?.clientModelRewrites ?? {} : {};
  if (rewrites[baseModel]) return rewrites[baseModel];
  const exactRoute = routes.find((route) =>
    route.provider === provider.name
    && route.match === baseModel
    && (route.match_type ?? "prefix") === "exact"
  );
  if (exactRoute?.rewrite_model) return exactRoute.rewrite_model;
  if (baseModel === DEFAULT_CLIENT_MODEL && provider.type !== "anthropic") {
    return openAiClientModelForProvider(provider);
  }
  if (baseModel === DEFAULT_OPENAI_CLIENT_MODEL && provider.type === "anthropic") {
    return DEFAULT_CLIENT_MODEL;
  }
  return baseModel;
}

function openAiClientModelForProvider(provider: ProviderConfig): string {
  return findProviderRuntimeRule(provider)?.clientModelRewrites?.[DEFAULT_CLIENT_MODEL] ?? DEFAULT_OPENAI_CLIENT_MODEL;
}

function providerAliasRoutes(providers: ProviderConfig[], routes: RouteConfig[]): RouteConfig[] {
  return providers
    .filter((provider) => provider.name)
    .flatMap((provider) => {
      const baseModels = Array.from(new Set([DEFAULT_CLIENT_MODEL, openAiClientModelForProvider(provider)]));
      return baseModels.flatMap((baseModel) => {
        const kind = baseModel === DEFAULT_CLIENT_MODEL ? "claude" as const : "openai" as const;
        const hasUserMapping = routes.some((route) =>
          route.provider === provider.name
          && route.client_kind === kind
          && (route.match_type ?? "prefix") === "exact"
          && route.match.trim()
        );
        if (hasUserMapping) return [];
        return [{
          match: providerAliasModel(provider, baseModel),
          match_type: "exact" as const,
          provider: provider.name,
          rewrite_model: providerAliasTarget(provider, routes, baseModel),
          client_kind: kind,
        }];
      });
    });
}

function mergeRoutes(preferred: RouteConfig[], existing: RouteConfig[]): RouteConfig[] {
  const seen = new Set<string>();
  const merged: RouteConfig[] = [];
  for (const route of [...preferred, ...existing]) {
    const key = `${route.match_type ?? "prefix"}\0${route.match}\0${route.provider}\0${route.rewrite_model ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(route);
  }
  return merged;
}

function clientKindForTool(tool: AITool): "claude" | "openai" {
  return tool === "claude" ? "claude" : "openai";
}

function clientModelForProvider(provider: ProviderConfig | undefined, routes: RouteConfig[], tool: AITool): string {
  if (!provider) return tool === "claude" ? DEFAULT_CLIENT_MODEL : DEFAULT_OPENAI_CLIENT_MODEL;
  const kind = clientKindForTool(tool);
  const configured = routes.find((route) =>
    route.provider === provider.name
    && route.client_kind === kind
    && (route.match_type ?? "prefix") === "exact"
    && route.match.trim()
  );
  if (configured) return configured.match;
  if (kind === "claude") return providerAliasModel(provider, DEFAULT_CLIENT_MODEL);
  return providerAliasModel(provider, openAiClientModelForProvider(provider));
}

function parsePrometheusMetrics(body: string): MetricSample {
  const values: Record<string, number> = {};
  const labeled = new Map<string, ModelMetric>();

  for (const rawLine of body.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$/i);
    if (!match) continue;
    const [, name, labelText, rawValue] = match;
    const value = Number(rawValue);
    if (!Number.isFinite(value)) continue;

    if (!labelText) {
      values[name] = value;
      continue;
    }

    const labels = parseMetricLabels(labelText);
    const provider = labels.provider ?? "";
    const model = labels.model ?? "";
    if (!provider && !model) continue;
    const key = `${provider}\u0000${model}`;
    const metric = labeled.get(key) ?? { provider, model, requests: 0, errors: 0, latencyMicros: 0, tokens: 0 };
    if (name === "ferryllm_labeled_requests") metric.requests = value;
    if (name === "ferryllm_labeled_errors") metric.errors = value;
    if (name === "ferryllm_labeled_latency_micros") metric.latencyMicros = value;
    if (name === "ferryllm_labeled_tokens") metric.tokens = value;
    labeled.set(key, metric);
  }

  return {
    ts: Date.now(),
    values,
    models: Array.from(labeled.values()).sort((a, b) => b.requests - a.requests),
  };
}

function parseMetricLabels(value: string): Record<string, string> {
  const labels: Record<string, string> = {};
  const re = /(\w+)="((?:\\.|[^"\\])*)"/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(value))) {
    labels[match[1]] = match[2].replace(/\\"/g, '"').replace(/\\\\/g, "\\").replace(/\\n/g, "\n");
  }
  return labels;
}

function delta(latest: MetricSample | undefined, previous: MetricSample | undefined, name: string): number {
  if (!latest) return 0;
  const current = latest.values[name] ?? 0;
  const before = previous?.values[name] ?? 0;
  return Math.max(0, current - before);
}

function fmtInt(value: number | undefined): string {
  return Math.round(value ?? 0).toLocaleString();
}

function fmtPct(value: number): string {
  return `${Math.max(0, Math.min(100, value)).toFixed(1)}%`;
}

function fmtLatency(micros: number): string {
  if (!Number.isFinite(micros) || micros <= 0) return "0ms";
  if (micros >= 1_000_000) return `${(micros / 1_000_000).toFixed(2)}s`;
  return `${Math.round(micros / 1000)}ms`;
}

function sparklinePath(points: number[], width = 320, height = 96): string {
  if (points.length === 0) return "";
  const max = Math.max(...points, 1);
  const min = Math.min(...points, 0);
  const span = Math.max(max - min, 1);
  return points
    .map((point, index) => {
      const x = points.length === 1 ? width : (index / (points.length - 1)) * width;
      const y = height - ((point - min) / span) * (height - 12) - 6;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function sparklineAreaPath(points: number[], width = 320, height = 96): string {
  const line = sparklinePath(points, width, height);
  if (!line) return "";
  return `${line} L ${width} ${height} L 0 ${height} Z`;
}

function uniqueName(base: string, existing: Set<string>): string {
  const clean = base.trim() || "provider";
  if (!existing.has(clean)) return clean;
  let index = 2;
  while (existing.has(`${clean}-${index}`)) index += 1;
  return `${clean}-${index}`;
}

type ProviderKeyInfo = { label: string; detail: string; ok: boolean };
type ProviderSummary = {
  key: ProviderKeyInfo;
  routeCount: number;
  fallbackCount: number;
  requests: number;
  errors: number;
  status: "live" | "configured" | "attention";
  statusLabel: string;
};

function providerInitial(provider: Pick<ProviderConfig, "name" | "type">): string {
  return (provider.name || provider.type || "P").slice(0, 1).toUpperCase();
}

function providerDisplayName(provider: Pick<ProviderConfig, "name">): string {
  const name = provider.name || "provider";
  const known: Record<string, string> = {
    openai: "OpenAI",
    anthropic: "Anthropic",
    openrouter: "OpenRouter",
    deepseek: "DeepSeek",
    kimi: "Kimi",
    zai: "Z.ai GLM",
    oneapi: "OneAPI / NewAPI",
    ollama: "Ollama",
    "azure-openai": "Azure OpenAI",
    gemini: "Gemini",
    groq: "Groq",
    together: "Together",
    fireworks: "Fireworks",
    perplexity: "Perplexity",
    mistral: "Mistral",
    cohere: "Cohere",
    xai: "xAI",
    dashscope: "DashScope",
    volcengine: "Volcengine",
    "baidu-qianfan": "Baidu Qianfan",
    "tencent-hunyuan": "Tencent Hunyuan",
    minimax: "MiniMax",
    siliconflow: "SiliconFlow",
    modelscope: "ModelScope",
    baichuan: "Baichuan",
    stepfun: "StepFun",
    aihubmix: "AIHubMix",
    "302ai": "302.AI",
    burncloud: "BurnCloud",
    shengsuanyun: "Shengsuanyun",
    pateway: "PatewayAI",
    "doubao-seed": "DouBaoSeed",
    "zhipu-glm": "Zhipu GLM",
    bailian: "Bailian",
    longcat: "Longcat",
    "xiaomi-mimo": "Xiaomi MiMo",
    novita: "Novita AI",
    dmxapi: "DMXAPI",
    packycode: "PackyCode",
    apikeyfun: "APIKEY.FUN",
    apinebula: "APINebula",
    atlascloud: "AtlasCloud",
    sudocode: "SudoCode",
    claudecn: "ClaudeCN",
    relaxycode: "RelaxyCode",
    aicodemirror: "AICodeMirror",
    crazyrouter: "CrazyRouter",
    sssaicode: "SSSAiCode",
    ctok: "CTok.ai",
    pipellm: "PIPELLM",
    therouter: "TheRouter",
    lmstudio: "LM Studio",
    vllm: "vLLM",
    "llama-cpp": "llama.cpp",
  };
  return known[name.toLowerCase()] ?? name;
}

const PROVIDER_LOGO_FILES: Record<string, string> = {
  "aihubmix-color": "aihubmix-color.svg",
  apikeyfun: "apikeyfun.png",
  apinebula: "apinebula_icon.png",
  atlascloud: "atlascloud_icon.png",
  byteplus: "byteplus.png",
  claudeapi: "ClaudeApi.png",
  claudecn: "claudecn.png",
  eflowcode: "eflowcode.png",
  hermes: "hermes.png",
  huoshan: "huoshan.png",
  lemondata: "lemondata.png",
  "longcat-color": "longcat-color.svg",
  "modelscope-color": "modelscope-color.svg",
  pateway: "pateway.jpg",
  pipellm: "pipellm.png",
  relaxycode: "relaxcode.png",
  runapi: "runapi.jpg",
  sudocode: "sudocode.png",
};

function providerLogoKey(provider: Pick<ProviderConfig, "name" | "type">): string | undefined {
  const name = (provider.name || "").toLowerCase();
  const type = (provider.type || "").toLowerCase();
  const id = name;
  if (id.includes("aicodemirror")) return "aicodemirror";
  if (id.includes("aihubmix")) return "aihubmix-color";
  if (id.includes("apikeyfun")) return "apikeyfun";
  if (id.includes("apinebula")) return "apinebula";
  if (id.includes("atlascloud")) return "atlascloud";
  if (id.includes("azure")) return "azure";
  if (id.includes("aws") || id.includes("bedrock")) return "aws";
  if (id.includes("bailian")) return "bailian";
  if (id.includes("baichuan")) return undefined;
  if (id.includes("baidu") || id.includes("qianfan") || id.includes("ernie") || id.includes("wenxin")) return "baidu";
  if (id.includes("byteplus")) return "byteplus";
  if (id.includes("claudecn")) return "claudecn";
  if (id.includes("claudeapi")) return "claudeapi";
  if (id.includes("anthropic") || id === "claude") return "anthropic";
  if (id.includes("cloudflare")) return "cloudflare";
  if (id.includes("cohere")) return "cohere";
  if (id.includes("ctok")) return "ctok";
  if (id.includes("cubence")) return "cubence";
  if (id.includes("crazyrouter")) return "crazyrouter";
  if (id.includes("dashscope") || id.includes("qwen")) return "qwen";
  if (id.includes("deepseek")) return "deepseek";
  if (id.includes("dmxapi")) return "dds";
  if (id.includes("eflow") || id.includes("e-flow")) return "eflowcode";
  if (id.includes("fireworks")) return undefined;
  if (id.includes("gemini") || id.includes("google")) return "gemini";
  if (id.includes("groq")) return undefined;
  if (id.includes("huggingface")) return "huggingface";
  if (id.includes("huoshan")) return "huoshan";
  if (id.includes("kimi") || id.includes("moonshot")) return "kimi";
  if (id.includes("lemondata")) return "lemondata";
  if (id.includes("llama")) return "meta";
  if (id.includes("longcat")) return "longcat-color";
  if (id.includes("micu")) return "micu";
  if (id.includes("minimax")) return "minimax";
  if (id.includes("mistral")) return "mistral";
  if (id.includes("modelscope")) return "modelscope-color";
  if (id.includes("newapi") || id.includes("oneapi")) return "newapi";
  if (id.includes("novita")) return "novita";
  if (id.includes("nvidia") || id.includes("vllm")) return "nvidia";
  if (id.includes("ollama")) return "ollama";
  if (id.includes("openai")) return "openai";
  if (id.includes("openrouter") || id.includes("therouter")) return "openrouter";
  if (id.includes("packy")) return "packycode";
  if (id.includes("pateway")) return "pateway";
  if (id.includes("perplexity")) return "perplexity";
  if (id.includes("pipellm")) return "pipellm";
  if (id.includes("relaxy")) return "relaxycode";
  if (id.includes("runapi")) return "runapi";
  if (id.includes("shengsuanyun")) return "shengsuanyun";
  if (id.includes("silicon")) return "siliconflow";
  if (id.includes("sssaicode")) return "sssaicode";
  if (id.includes("stepfun")) return "stepfun";
  if (id.includes("sudocode")) return "sudocode";
  if (id.includes("tencent") || id.includes("hunyuan")) return "hunyuan";
  if (id.includes("together")) return undefined;
  if (id.includes("ucloud")) return "ucloud";
  if (id.includes("volc") || id.includes("doubao") || id.includes("byte")) return "doubao";
  if (id.includes("xai") || id.includes("grok")) return "xai";
  if (id.includes("xiaomi") || id.includes("mimo")) return "xiaomimimo";
  if (id.includes("yi-") || id.includes("lingyi") || id.includes("zeroone")) return "zeroone";
  if (id.includes("zhipu")) return "zhipu";
  if (id.includes("zai") || id.includes("z.ai") || id.includes("glm")) return "chatglm";
  if (!name && type === "anthropic") return "anthropic";
  if (!name && type === "gemini") return "gemini";
  return undefined;
}

function providerLogoSrc(provider: Pick<ProviderConfig, "name" | "type">): string | undefined {
  const key = providerLogoKey(provider);
  return key ? `/provider-logos/${PROVIDER_LOGO_FILES[key] ?? `${key}.svg`}` : undefined;
}

function providerLogoClass(provider: Pick<ProviderConfig, "name" | "type">): string {
  const id = `${provider.name} ${provider.type}`.toLowerCase();
  if (id.includes("anthropic") || id.includes("claude")) return "provider-logo-amber";
  if (id.includes("openrouter") || id.includes("oneapi")) return "provider-logo-violet";
  if (id.includes("deepseek") || id.includes("zai") || id.includes("glm")) return "provider-logo-cyan";
  if (id.includes("ollama") || id.includes("local")) return "provider-logo-emerald";
  return "provider-logo-blue";
}

function ProviderLogo({
  provider,
  className,
}: {
  provider: Pick<ProviderConfig, "name" | "type">;
  className?: string;
}) {
  const src = providerLogoSrc(provider);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [src]);

  return (
    <span className={cn("provider-logo", src && !failed ? "provider-logo-real" : providerLogoClass(provider), className)}>
      {src && !failed ? (
        <img
          src={src}
          alt=""
          className="provider-logo-image"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        providerInitial(provider)
      )}
    </span>
  );
}

function providerKeyInfo(provider: ProviderConfig): ProviderKeyInfo {
  if (provider.api_key) return { label: "Direct key", detail: "stored in config", ok: true };
  if (provider.api_key_env) return { label: "Env key", detail: provider.api_key_env, ok: true };
  if (provider.api_key_url) return { label: "URL key", detail: provider.api_key_url, ok: true };
  if (provider.api_key_file) return { label: "File key", detail: provider.api_key_file, ok: true };
  if (provider.key_watch?.length) return { label: "Watched key", detail: `${provider.key_watch.length} watch${provider.key_watch.length === 1 ? "" : "es"}`, ok: true };
  return { label: "No key source", detail: "add env, file, URL, or direct key", ok: false };
}

function parseProviderModels(body: string): string[] {
  try {
    const value = JSON.parse(body);
    const candidates: unknown[] = Array.isArray(value)
      ? value
      : Array.isArray(value.models)
        ? value.models
        : Array.isArray(value.data)
          ? value.data
          : [];
    return Array.from(new Set<string>(candidates
      .map((item: unknown) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const object = item as Record<string, unknown>;
          return typeof object.id === "string" ? object.id : typeof object.name === "string" ? object.name : "";
        }
        return "";
      })
      .map((item: string) => item.trim())
      .filter(Boolean))).sort((a, b) => a.localeCompare(b));
  } catch {
    return [];
  }
}

function providerMappingRows(provider: ProviderConfig, routes: RouteConfig[]) {
  const defaults = [
    {
      kind: "claude" as const,
      label: "Claude clients",
      fallbackClient: providerAliasModel(provider, DEFAULT_CLIENT_MODEL),
      fallbackUpstream: providerAliasTarget(provider, routes, DEFAULT_CLIENT_MODEL),
    },
    {
      kind: "openai" as const,
      label: "Codex / OpenCode",
      fallbackClient: providerAliasModel(provider, openAiClientModelForProvider(provider)),
      fallbackUpstream: providerAliasTarget(provider, routes, openAiClientModelForProvider(provider)),
    },
  ];
  return defaults.map((row) => {
    const routeIndex = routes.findIndex((route) =>
      route.provider === provider.name
      && route.client_kind === row.kind
      && (route.match_type ?? "prefix") === "exact"
    );
    const route = routeIndex >= 0 ? routes[routeIndex] : undefined;
    return {
      ...row,
      routeIndex,
      clientModel: route?.match || row.fallbackClient,
      upstreamModel: route?.rewrite_model || row.fallbackUpstream,
    };
  });
}

function workspaceName(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || path || "Workspace";
}

function buildWorkspaces(launches: RecentLaunch[], pinnedPaths: Set<string>, bindings: Record<string, WorkspaceBinding>): WorkspaceEntry[] {
  const byPath = new Map<string, RecentLaunch[]>();
  for (const launch of launches) {
    const list = byPath.get(launch.directory) ?? [];
    list.push(launch);
    byPath.set(launch.directory, list);
  }

  return Array.from(byPath.entries())
    .map(([path, sessions]) => {
      const sorted = [...sessions].sort((a, b) => b.lastUsed - a.lastUsed);
      const latest = sorted[0];
      const binding = bindings[path];
      return {
        path,
        name: workspaceName(path),
        pinned: pinnedPaths.has(path),
        lastUsed: latest?.lastUsed ?? 0,
        defaultTool: binding?.tool ?? latest?.tool ?? "codex",
        defaultLaunchType: binding?.launchType ?? latest?.launchType ?? "cli",
        defaultReasoningEffort: binding?.reasoningEffort ?? latest?.reasoningEffort,
        defaultProviderName: binding?.providerName ?? latest?.providerName ?? "",
        defaultProviderType: binding?.providerType ?? latest?.providerType ?? "openai",
        sessions: sorted,
      };
    })
    .sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.lastUsed - a.lastUsed || a.name.localeCompare(b.name));
}

function formatRelativeTime(ts: number): string {
  if (!ts) return "Never";
  const diff = Date.now() - ts;
  const minute = 60_000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diff < minute) return "Just now";
  if (diff < hour) return `${Math.floor(diff / minute)}m ago`;
  if (diff < day) return `${Math.floor(diff / hour)}h ago`;
  if (diff < day * 7) return `${Math.floor(diff / day)}d ago`;
  return new Date(ts).toLocaleDateString();
}

function buildProviderSummary(provider: ProviderConfig, routes: RouteConfig[], latestSample: MetricSample | undefined): ProviderSummary {
  const ownedRoutes = routes.filter((route) => route.provider === provider.name);
  const fallbackRoutes = routes.filter((route) => (route.fallback_providers ?? []).includes(provider.name));
  const modelRows = latestSample?.models.filter((row) => row.provider === provider.name) ?? [];
  const requests = modelRows.reduce((sum, row) => sum + row.requests, 0);
  const errors = modelRows.reduce((sum, row) => sum + row.errors, 0);
  const key = providerKeyInfo(provider);
  const status = !provider.base_url || !key.ok || errors > 0 ? "attention" : requests > 0 ? "live" : "configured";
  const statusLabel = !provider.base_url
    ? "Needs URL"
    : !key.ok
      ? "Needs key"
      : errors > 0
        ? `${fmtInt(errors)} errors`
        : requests > 0
          ? "Live traffic"
          : "Configured";
  return { key, routeCount: ownedRoutes.length, fallbackCount: fallbackRoutes.length, requests, errors, status, statusLabel };
}

/* ── Animated card wrapper ─────────────────────────────────────────── */

function Card({ children, active, onClick, className }: {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
  className?: string;
}) {
  return (
    <motion.div
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      onClick={onClick}
      onKeyDown={(event) => {
        if (!onClick) return;
        if (event.target !== event.currentTarget) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick();
        }
      }}
      className={cn(
        onClick && "cursor-pointer",
        "group flex w-full items-center gap-4 rounded-2xl border p-5 text-left transition-all duration-200",
        "bg-surface hover:shadow-md",
        active
          ? "border-primary/50 shadow-[0_0_0_1px_rgba(37,99,235,0.15)] ring-1 ring-primary/20"
          : "border-border hover:border-border-strong",
        className
      )}
    >
      {children}
    </motion.div>
  );
}

/* ── Main App ──────────────────────────────────────────────────────── */

function App() {
  const [executable, setExecutable] = useState("ferryllm");
  const [config, setConfig] = useState<AppConfig>(emptyConfig);
  const [configLoaded, setConfigLoaded] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState(0);
  const [selectedRoute, setSelectedRoute] = useState(0);
  const hotReload = true;
  const [status, setStatus] = useState<ProcessStatus | null>(null);
  const [snapshot, setSnapshot] = useState<ServerSnapshot | null>(null);
  const [metricSamples, setMetricSamples] = useState<MetricSample[]>([]);
  const [dashboardBusy, setDashboardBusy] = useState(false);
  const [aiSessions, setAiSessions] = useState<AISession[]>([]);
  const [sessionScanBusy, setSessionScanBusy] = useState(false);
  const [validation, setValidation] = useState<CommandResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [activeView, setActiveView] = useState<View>("dashboard");
  const [detailTab, setDetailTab] = useState<DetailTab>("settings");
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("general");
  const [darkMode, setDarkMode] = useState(() => {
    const s = localStorage.getItem("ferryllm-theme");
    if (s) return s === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [recentLaunches, setRecentLaunches] = useState<RecentLaunch[]>(() => {
    try { return JSON.parse(localStorage.getItem("ferryllm-launches") || "[]"); } catch { return []; }
  });
  const [workspaceBindings, setWorkspaceBindings] = useState<Record<string, WorkspaceBinding>>(() => {
    try { return JSON.parse(localStorage.getItem("ferryllm-workspace-bindings") || "{}"); } catch { return {}; }
  });
  const [launcherStateLoaded, setLauncherStateLoaded] = useState(() => !isTauriRuntime());
  const [launchTool, setLaunchTool] = useState<AITool>("codex");
  const [launcherTarget] = useState<LaunchTarget>("cli");
  const [workspaceSearch, setWorkspaceSearch] = useState("");
  const [workspacePins, setWorkspacePins] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem("ferryllm-workspace-pins") || "[]"); } catch { return []; }
  });
  const [selectedWorkspacePath, setSelectedWorkspacePath] = useState("");
  const [providerSearch, setProviderSearch] = useState("");
  const [presetFilter, setPresetFilter] = useState<PresetCategoryFilter>("all");
  const [presetsExpanded, setPresetsExpanded] = useState(false);
  const [providerModelCatalog, setProviderModelCatalog] = useState<Record<string, string[]>>(() => {
    try { return JSON.parse(localStorage.getItem("ferryllm-provider-models") || "{}"); } catch { return {}; }
  });
  const [modelProbeBusy, setModelProbeBusy] = useState(false);
  const toastId = useRef(0);

  const providers = config.providers ?? [];
  const routes = config.routes ?? [];
  const selectedProviderConfig = providers[selectedProvider];
  const selectedRouteConfig = routes[selectedRoute];
  const providerNames = useMemo(() => providers.map((p) => p.name).filter(Boolean), [providers]);
  const providerQuery = providerSearch.trim().toLowerCase();
  const visibleProviders = useMemo(() => {
    if (!providerQuery) return providers.map((provider, index) => ({ provider, index }));
    return providers
      .map((provider, index) => ({ provider, index }))
      .filter(({ provider }) => `${provider.name} ${provider.type} ${provider.base_url}`.toLowerCase().includes(providerQuery));
  }, [providerQuery, providers]);
  const visibleProviderPresets = useMemo(() => {
    const queryFiltered = providerPresets.filter((preset) =>
      !providerQuery || `${preset.name} ${preset.type} ${preset.base_url} ${preset.description} ${preset.category}`.toLowerCase().includes(providerQuery)
    );
    const categoryFiltered = presetFilter === "all" ? queryFiltered : queryFiltered.filter((preset) => preset.category === presetFilter);
    return [...categoryFiltered].sort((a, b) => Number(Boolean(b.featured)) - Number(Boolean(a.featured)) || a.name.localeCompare(b.name));
  }, [presetFilter, providerQuery]);
  const shouldOfferPresetExpand = !providerQuery && visibleProviderPresets.length > MIN_COLLAPSED_PROVIDER_PRESET_COUNT;
  const shouldCollapsePresets = shouldOfferPresetExpand && !presetsExpanded && !providerQuery;
  const displayedProviderPresets = visibleProviderPresets;
  const latestSample = metricSamples.length ? metricSamples[metricSamples.length - 1] : undefined;
  const providerSummaries = useMemo(
    () => providers.map((provider) => buildProviderSummary(provider, routes, latestSample)),
    [providers, routes, latestSample]
  );
  const workspacePinSet = useMemo(() => new Set(workspacePins), [workspacePins]);
  const workspaces = useMemo(() => buildWorkspaces(recentLaunches, workspacePinSet, workspaceBindings), [recentLaunches, workspaceBindings, workspacePinSet]);
  const workspaceQuery = workspaceSearch.trim().toLowerCase();
  const filteredWorkspaces = useMemo(() => {
    if (!workspaceQuery) return workspaces;
    return workspaces.filter((workspace) =>
      `${workspace.name} ${workspace.path} ${workspace.defaultTool} ${workspace.defaultLaunchType} ${workspace.defaultProviderName}`.toLowerCase().includes(workspaceQuery)
    );
  }, [workspaceQuery, workspaces]);
  const selectedWorkspace = useMemo(() => {
    return workspaces.find((workspace) => workspace.path === selectedWorkspacePath) ?? filteredWorkspaces[0] ?? workspaces[0];
  }, [filteredWorkspaces, selectedWorkspacePath, workspaces]);
  const favoriteWorkspaces = filteredWorkspaces.filter((workspace) => workspace.pinned);
  const recentWorkspaces = filteredWorkspaces.filter((workspace) => !workspace.pinned);
  const selectedWorkspaceSessions = useMemo(() => {
    if (!selectedWorkspace) return [];
    return aiSessions
      .filter((session) => session.cwd.toLowerCase() === selectedWorkspace.path.toLowerCase())
      .sort((a, b) => b.updated_at_ms - a.updated_at_ms);
  }, [aiSessions, selectedWorkspace]);
  const doctorItems = useMemo(() => {
    const namedProviders = providers.filter((provider) => provider.name.trim());
    const duplicateNames = namedProviders.length - new Set(namedProviders.map((provider) => provider.name)).size;
    const keyedProviders = providers.filter((provider) => providerKeyInfo(provider).ok).length;
    const routedProviders = new Set(routes.map((route) => route.provider).filter(Boolean)).size;
    return [
      { label: "Providers named", ok: providers.length > 0 && duplicateNames === 0, detail: duplicateNames ? `${duplicateNames} duplicate` : `${namedProviders.length}/${providers.length}` },
      { label: "Key sources", ok: providers.length > 0 && keyedProviders === providers.length, detail: `${keyedProviders}/${providers.length || 0}` },
      { label: "Routes assigned", ok: routes.length > 0 && routedProviders > 0, detail: `${routes.length} route${routes.length === 1 ? "" : "s"}` },
      { label: "Runtime probes", ok: Boolean(status?.running), detail: status?.running ? `PID ${status.pid ?? "-"}` : "server stopped" },
    ];
  }, [providers, routes, status?.pid, status?.running]);
  const doctorReadyCount = doctorItems.filter((item) => item.ok).length;
  const dashboard = useMemo(() => {
    const requests = latestSample?.values.ferryllm_requests_total ?? 0;
    const ok = latestSample?.values.ferryllm_requests_ok_total ?? 0;
    const errors = latestSample?.values.ferryllm_requests_error_total ?? 0;
    const upstreamErrors = latestSample?.values.ferryllm_upstream_errors_total ?? 0;
    const latencyTotal = latestSample?.values.ferryllm_request_latency_micros_total ?? 0;
    const cachePrompt = latestSample?.values.ferryllm_cache_prompt_tokens_total ?? 0;
    const cached = latestSample?.values.ferryllm_prompt_cached_tokens_total ?? 0;
    const cacheHit = cachePrompt > 0 ? (cached / cachePrompt) * 100 : 0;
    const successRate = requests > 0 ? (ok / requests) * 100 : 0;
    const avgLatency = requests > 0 ? latencyTotal / requests : 0;
    const requestRatePoints = metricSamples.map((sample, index) =>
      index === 0 ? 0 : delta(sample, metricSamples[index - 1], "ferryllm_requests_total")
    );
    const latencyPoints = metricSamples.map((sample, index) => {
      if (index === 0) return 0;
      const requestDelta = delta(sample, metricSamples[index - 1], "ferryllm_requests_total");
      const latencyDelta = delta(sample, metricSamples[index - 1], "ferryllm_request_latency_micros_total");
      return requestDelta > 0 ? latencyDelta / requestDelta : 0;
    });
    return { requests, ok, errors, upstreamErrors, cacheHit, successRate, avgLatency, requestRatePoints, latencyPoints };
  }, [latestSample, metricSamples]);

  function addToast(message: string, type: Toast["type"] = "info") {
    const id = ++toastId.current;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 3500);
  }

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", darkMode ? "dark" : "light");
    localStorage.setItem("ferryllm-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  useEffect(() => {
    if (!isTauriRuntime()) return;
    void bootstrap();
    const unlisten = listen<LogEntry>("server-log", () => {
      void refreshStatus();
      if (activeView === "dashboard") void refreshDashboard(false);
    });
    return () => { void unlisten.then((u) => u()); };
  }, [activeView]);

  useEffect(() => {
    if (activeView !== "dashboard" || !isTauriRuntime()) return;
    void refreshDashboard(false);
    const interval = window.setInterval(() => void refreshDashboard(false), 3000);
    return () => window.clearInterval(interval);
  }, [activeView, config.server?.listen]);

  useEffect(() => {
    if (activeView !== "launcher" || !isTauriRuntime()) return;
    void refreshAiSessions(false);
    const interval = window.setInterval(() => void refreshAiSessions(false), 10000);
    return () => window.clearInterval(interval);
  }, [activeView]);

  useEffect(() => {
    localStorage.setItem("ferryllm-provider-models", JSON.stringify(providerModelCatalog));
  }, [providerModelCatalog]);

  useEffect(() => {
    if (!workspaces.length) {
      if (selectedWorkspacePath) setSelectedWorkspacePath("");
      return;
    }
    if (!selectedWorkspacePath || !workspaces.some((workspace) => workspace.path === selectedWorkspacePath)) {
      setSelectedWorkspacePath(workspaces[0].path);
    }
  }, [selectedWorkspacePath, workspaces]);

  useEffect(() => {
    (async () => {
      if (!isTauriRuntime()) {
        setLauncherStateLoaded(true);
        return;
      }
      try {
        const stored = await invoke<string | null>("load_launcher_state");
        if (stored) {
          const parsed = JSON.parse(stored);
          if (Array.isArray(parsed.launches)) setRecentLaunches(parsed.launches);
          if (parsed.bindings && typeof parsed.bindings === "object") setWorkspaceBindings(parsed.bindings);
          if (Array.isArray(parsed.pins)) setWorkspacePins(parsed.pins);
        }
      } catch {
        /* keep localStorage fallback */
      } finally {
        setLauncherStateLoaded(true);
      }
    })();
  }, []);

  useEffect(() => {
    if (!launcherStateLoaded) return;
    const state = { launches: recentLaunches, bindings: workspaceBindings, pins: workspacePins };
    localStorage.setItem("ferryllm-launches", JSON.stringify(recentLaunches));
    localStorage.setItem("ferryllm-workspace-bindings", JSON.stringify(workspaceBindings));
    localStorage.setItem("ferryllm-workspace-pins", JSON.stringify(workspacePins));
    if (isTauriRuntime()) {
      invoke("save_launcher_state", { request: { state } }).catch(() => {});
    }
  }, [launcherStateLoaded, recentLaunches, workspaceBindings, workspacePins]);

  /* ── Load config from file on startup ── */
  useEffect(() => {
    (async () => {
      if (!isTauriRuntime()) {
        setConfigLoaded(true);
        return;
      }
      try {
        const stored = await invoke<string | null>("load_config_from_default");
        if (stored) {
          setConfig(normalizeConfig(JSON.parse(stored)));
        }
      } catch { /* ignore, use empty config */ }
      setConfigLoaded(true);
    })();
  }, []);

  /* ── Auto-save config to file + localStorage ── */
  useEffect(() => {
    if (!configLoaded) return;
    const timeout = window.setTimeout(() => {
      const redacted = redactConfigSecrets(config);
      localStorage.setItem("ferryllm-config", JSON.stringify(redacted));
      if (isTauriRuntime()) {
        invoke("save_config_to_default", { request: { config } }).catch(() => {});
      }
    }, 500);
    return () => window.clearTimeout(timeout);
  }, [config, configLoaded]);

  /* ── Backend calls ─────────────────────────────────────── */

  async function bootstrap() {
    try {
      const [discovered, st] = await Promise.all([
        invoke<string>("discover_ferryllm"),
        invoke<ProcessStatus>("server_status"),
      ]);
      setExecutable(discovered);
      setStatus(st);
    } catch (e) { addToast(String(e), "error"); }
  }

  async function persistRunnableConfig(hotReloadConfig = false): Promise<SaveResult> {
    return await invoke<SaveResult>("save_config_file", {
      request: { path: "", config: runtimeConfigForGateway(config), executable, hot_reload: hotReloadConfig },
    });
  }

  async function saveConfig() {
    setBusy(true);
    try {
      const r = await persistRunnableConfig(hotReload);
      setValidation(r.validation);
      await refreshStatus();
      addToast(r.reloaded ? "Saved & hot reloaded" : "Saved & validated", "success");
    } catch (e) { addToast(String(e), "error"); }
    finally { setBusy(false); }
  }

  async function validateConfig() {
    setBusy(true);
    try {
      const r = await invoke<CommandResult>("validate_config_document", { request: { executable, config } });
      setValidation(r);
      addToast(r.ok ? "Config is valid" : commandResultMessage(r), r.ok ? "success" : "error");
    } catch (e) { addToast(String(e), "error"); }
    finally { setBusy(false); }
  }

  async function startServer() {
    setBusy(true);
    try {
      const saved = await persistRunnableConfig(false);
      setValidation(saved.validation);
      if (!saved.validation.ok) {
        addToast(commandResultMessage(saved.validation), "error");
        return;
      }
      const current = await invoke<ProcessStatus>("server_status");
      const s = await invoke<ProcessStatus>("start_server", {
        request: { executable, config_path: saved.path, replace_existing: current.source === "external" },
      });
      setStatus(s);
      addToast(s.source === "external" ? "Detected existing ferryllm gateway" : "ferryllm started", "success");
    } catch (e) { addToast(String(e), "error"); }
    finally { setBusy(false); }
  }

  async function stopServer() {
    setBusy(true);
    try {
      const s = await invoke<ProcessStatus>("stop_server");
      setStatus(s);
      addToast(s.source === "external" ? "Gateway is running outside this window" : "ferryllm stopped", s.source === "external" ? "warning" : "success");
    } catch (e) { addToast(String(e), "error"); }
    finally { setBusy(false); }
  }

  async function restartServer() {
    setBusy(true);
    try {
      const saved = await persistRunnableConfig(false);
      setValidation(saved.validation);
      if (!saved.validation.ok) {
        addToast(commandResultMessage(saved.validation), "error");
        return;
      }
      const s = await invoke<ProcessStatus>("restart_server", { request: { executable, config_path: saved.path, replace_existing: true } });
      setStatus(s);
      addToast(s.source === "external" ? "Detected existing ferryllm gateway" : "ferryllm restarted", "success");
    } catch (e) { addToast(String(e), "error"); }
    finally { setBusy(false); }
  }

  async function refreshStatus() {
    setStatus(await invoke<ProcessStatus>("server_status"));
  }

  async function ensureGatewayRunning(): Promise<boolean> {
    const saved = await persistRunnableConfig(true);
    setValidation(saved.validation);
    if (!saved.validation.ok) {
      addToast(commandResultMessage(saved.validation), "error");
      return false;
    }

    const current = await invoke<ProcessStatus>("server_status");
    if (current.running && current.source !== "external") {
      setStatus(current);
      return true;
    }

    const started = await invoke<ProcessStatus>("start_server", {
      request: { executable, config_path: saved.path, replace_existing: current.source === "external" },
    });
    setStatus(started);
    if (!started.running) {
      addToast("Gateway failed to start", "error");
      return false;
    }
    addToast(started.source === "external" ? "Detected existing ferryllm gateway" : "Gateway started", "success");
    return true;
  }

  async function refreshDashboard(showError = true) {
    setDashboardBusy(true);
    try {
      const listen = valueAsString(config.server?.listen) || "127.0.0.1:3000";
      const next = await invoke<ServerSnapshot>("fetch_server_snapshot", { listen });
      setSnapshot(next);
      setStatus(next.status);
      if (next.metrics.ok) {
        const sample = parsePrometheusMetrics(next.metrics.body);
        setMetricSamples((prev) => [...prev, sample].slice(-120));
      }
    } catch (e) {
      if (showError) addToast(String(e), "error");
    } finally {
      setDashboardBusy(false);
    }
  }

  async function refreshAiSessions(showError = true) {
    setSessionScanBusy(true);
    try {
      setAiSessions(await invoke<AISession[]>("scan_ai_sessions"));
    } catch (e) {
      if (showError) addToast(String(e), "error");
    } finally {
      setSessionScanBusy(false);
    }
  }

  function recordLaunch(directory: string, tool: AITool, launchType: LaunchTarget, provider = selectedProviderConfig, reasoningEffort?: ReasoningEffort) {
    const now = Date.now();
    const entry: RecentLaunch = {
      id: now.toString(36),
      directory,
      tool,
      launchType,
      providerName: provider?.name || "unconfigured",
      providerType: provider?.type || "openai",
      reasoningEffort,
      configPath: "localStorage",
      lastUsed: now,
    };
    if (provider) {
      setWorkspaceBindings((prev) => {
        const next = {
          ...prev,
          [directory]: {
            providerName: provider.name,
            providerType: provider.type,
            tool,
            launchType,
            reasoningEffort,
            updatedAt: now,
          },
        };
        localStorage.setItem("ferryllm-workspace-bindings", JSON.stringify(next));
        return next;
      });
    }
    setRecentLaunches((prev) => {
      const deduped = prev.filter((r) => !(r.directory === directory && r.tool === tool && r.launchType === launchType));
      const next = [entry, ...deduped].slice(0, 50);
      localStorage.setItem("ferryllm-launches", JSON.stringify(next));
      return next;
    });
    setSelectedWorkspacePath(directory);
  }

  function resolveWorkspaceProvider(workspace: Pick<WorkspaceEntry, "defaultProviderName" | "defaultProviderType">): ProviderConfig | undefined {
    const provider = providers.find((item) => item.name === workspace.defaultProviderName)
      ?? providers.find((item) => item.type === workspace.defaultProviderType && workspace.defaultProviderName)
      ?? selectedProviderConfig
      ?? providers[0];
    if (provider) {
      const index = providers.findIndex((item) => item.name === provider.name);
      if (index >= 0 && index !== selectedProvider) setSelectedProvider(index);
    }
    return provider;
  }

  function updateWorkspaceGateway(path: string, providerName: string) {
    const provider = providers.find((item) => item.name === providerName);
    if (!provider) return;
    setWorkspaceBindings((prev) => {
      const current = prev[path];
      const workspace = workspaces.find((item) => item.path === path);
      const next = {
        ...prev,
        [path]: {
          providerName: provider.name,
          providerType: provider.type,
          tool: current?.tool ?? workspace?.defaultTool ?? launchTool,
          launchType: current?.launchType ?? workspace?.defaultLaunchType ?? launcherTarget,
          reasoningEffort: current?.reasoningEffort ?? workspace?.defaultReasoningEffort,
          updatedAt: Date.now(),
        },
      };
      localStorage.setItem("ferryllm-workspace-bindings", JSON.stringify(next));
      return next;
    });
    const index = providers.findIndex((item) => item.name === provider.name);
    if (index >= 0) setSelectedProvider(index);
  }

  function updateWorkspaceReasoning(path: string, reasoningEffort: string) {
    setWorkspaceBindings((prev) => {
      const current = prev[path];
      const workspace = workspaces.find((item) => item.path === path);
      const provider = resolveWorkspaceProvider(workspace ?? {
        defaultProviderName: current?.providerName ?? "",
        defaultProviderType: current?.providerType ?? "openai",
      });
      const next = {
        ...prev,
        [path]: {
          providerName: current?.providerName ?? workspace?.defaultProviderName ?? provider?.name ?? "",
          providerType: current?.providerType ?? workspace?.defaultProviderType ?? provider?.type ?? "openai",
          tool: current?.tool ?? workspace?.defaultTool ?? launchTool,
          launchType: current?.launchType ?? workspace?.defaultLaunchType ?? launcherTarget,
          reasoningEffort: (reasoningEffort || undefined) as ReasoningEffort | undefined,
          updatedAt: Date.now(),
        },
      };
      localStorage.setItem("ferryllm-workspace-bindings", JSON.stringify(next));
      return next;
    });
  }

  function deleteLaunch(id: string) {
    setRecentLaunches((prev) => {
      const next = prev.filter((r) => r.id !== id);
      localStorage.setItem("ferryllm-launches", JSON.stringify(next));
      return next;
    });
  }

  function deleteWorkspace(path: string) {
    setRecentLaunches((prev) => {
      const next = prev.filter((r) => r.directory !== path);
      localStorage.setItem("ferryllm-launches", JSON.stringify(next));
      return next;
    });
    setWorkspacePins((prev) => {
      const next = prev.filter((item) => item !== path);
      localStorage.setItem("ferryllm-workspace-pins", JSON.stringify(next));
      return next;
    });
    setWorkspaceBindings((prev) => {
      const next = { ...prev };
      delete next[path];
      localStorage.setItem("ferryllm-workspace-bindings", JSON.stringify(next));
      return next;
    });
  }

  async function deleteWorkspaceDirectory(path: string) {
    const ok = window.confirm(`Delete this workspace directory?\n\n${path}\n\nThis removes the folder from disk and cannot be undone.`);
    if (!ok) return;
    try {
      await invoke("delete_workspace", { request: { path } });
      deleteWorkspace(path);
      addToast("Workspace directory deleted", "success");
      void refreshAiSessions(false);
    } catch (e) { addToast(String(e), "error"); }
  }

  async function deleteNativeSession(session: AISession) {
    const ok = window.confirm(`Delete this ${session.tool} session file?\n\n${session.path}`);
    if (!ok) return;
    try {
      await invoke("delete_ai_session", { request: { path: session.path } });
      setAiSessions((prev) => prev.filter((item) => !(item.tool === session.tool && item.id === session.id && item.path === session.path)));
      addToast("Session deleted", "success");
    } catch (e) { addToast(String(e), "error"); }
  }

  function toggleWorkspacePin(path: string) {
    setWorkspacePins((prev) => {
      const next = prev.includes(path) ? prev.filter((item) => item !== path) : [path, ...prev];
      localStorage.setItem("ferryllm-workspace-pins", JSON.stringify(next));
      return next;
    });
  }

  async function addWorkspaceOnly() {
    const dir = await open({ directory: true, title: "Select workspace" });
    if (!dir) return;
    recordLaunch(dir, launchTool, launcherTarget, selectedProviderConfig);
    void refreshAiSessions(false);
    addToast("Workspace added", "success");
  }

  async function createWorkspace() {
    const parent = await open({ directory: true, title: "Select parent folder" });
    if (!parent) return;
    const name = window.prompt("Project name");
    const cleanName = name?.trim();
    if (!cleanName) return;
    if (/[<>:"/\\|?*]/.test(cleanName)) {
      addToast("Project name contains invalid path characters", "error");
      return;
    }
    const separator = parent.includes("\\") ? "\\" : "/";
    const path = `${parent.replace(/[\\/]+$/, "")}${separator}${cleanName}`;
    try {
      await invoke("create_workspace", { request: { path } });
      recordLaunch(path, launchTool, launcherTarget, selectedProviderConfig);
      void refreshAiSessions(false);
      addToast("Project created", "success");
    } catch (e) { addToast(String(e), "error"); }
  }

  async function revealWorkspace(path: string) {
    try {
      await invoke("reveal_workspace", { request: { path } });
    } catch (e) { addToast(String(e), "error"); }
  }

  async function launchWorkspace(workspace: WorkspaceEntry, target: LaunchTarget = launcherTarget, tool: AITool = launchTool) {
    const provider = resolveWorkspaceProvider(workspace);
    if (!provider) {
      addToast("Add a provider before launching a workspace", "error");
      setActiveView("providers");
      return;
    }
    const listen = valueAsString(config.server?.listen) || "127.0.0.1:3000";
    const clientModel = clientModelForProvider(provider, routes, tool);
    const reasoningEffort = workspace.defaultReasoningEffort;
    try {
      if (!(await ensureGatewayRunning())) return;
      if (target === "vscode") {
        await invoke("launch_vscode", { request: { directory: workspace.path, listen, provider_type: provider.type, tool, client_model: clientModel, client_reasoning_effort: reasoningEffort } });
      } else {
        await invoke("launch_cli", { request: { directory: workspace.path, listen, provider_type: provider.type, tool, client_model: clientModel, client_reasoning_effort: reasoningEffort } });
      }
      recordLaunch(workspace.path, tool, target, provider, reasoningEffort);
      addToast(`${target === "vscode" ? "VS Code" : `${tool} CLI`} launched`, "success");
    } catch (e) { addToast(String(e), "error"); }
  }

  async function resumeAiSession(session: AISession) {
    const workspace = workspaces.find((item) => item.path.toLowerCase() === session.cwd.toLowerCase());
    const provider = workspace ? resolveWorkspaceProvider(workspace) : selectedProviderConfig;
    if (!provider) {
      addToast("Add a provider before resuming a session", "error");
      setActiveView("providers");
      return;
    }
    const listen = valueAsString(config.server?.listen) || "127.0.0.1:3000";
    const clientModel = clientModelForProvider(provider, routes, session.tool);
    const reasoningEffort = workspace?.defaultReasoningEffort;
    try {
      if (!(await ensureGatewayRunning())) return;
      await invoke("resume_ai_session", {
        request: { id: session.id, cwd: session.cwd, listen, provider_type: provider.type, tool: session.tool, client_model: clientModel, client_reasoning_effort: reasoningEffort },
      });
      recordLaunch(session.cwd, session.tool, "cli", provider, reasoningEffort);
      addToast(`${session.tool} session resumed`, "success");
    } catch (e) { addToast(String(e), "error"); }
  }

  async function quickLaunch(item: RecentLaunch) {
    const listen = valueAsString(config.server?.listen) || "127.0.0.1:3000";
    const providerIndex = providers.findIndex((provider) => provider.name === item.providerName);
    if (providerIndex >= 0 && providerIndex !== selectedProvider) setSelectedProvider(providerIndex);
    const provider = providers[providerIndex]
      ?? providers.find((candidate) => candidate.type === item.providerType && providers.filter((inner) => inner.type === item.providerType).length === 1)
      ?? selectedProviderConfig;
    if (!provider) {
      addToast("Add a provider before launching a workspace", "error");
      setActiveView("providers");
      return;
    }
    const clientModel = clientModelForProvider(provider, routes, item.tool);
    const reasoningEffort = item.reasoningEffort;
    try {
      if (!(await ensureGatewayRunning())) return;
      if (item.launchType === "vscode") {
        await invoke("launch_vscode", { request: { directory: item.directory, listen, provider_type: provider.type, tool: item.tool, client_model: clientModel, client_reasoning_effort: reasoningEffort } });
      } else {
        await invoke("launch_cli", { request: { directory: item.directory, listen, provider_type: provider.type, tool: item.tool, client_model: clientModel, client_reasoning_effort: reasoningEffort } });
      }
      setRecentLaunches((prev) => {
        const next = prev.map((r) => r.id === item.id ? { ...r, lastUsed: Date.now() } : r).sort((a, b) => b.lastUsed - a.lastUsed);
        localStorage.setItem("ferryllm-launches", JSON.stringify(next));
        return next;
      });
      setSelectedWorkspacePath(item.directory);
      addToast(`${item.launchType === "vscode" ? "VS Code" : item.tool + " CLI"} launched`, "success");
    } catch (e) { addToast(String(e), "error"); }
  }

  /* ── Config mutation helpers ──────────────────────────── */

  function updateSection(section: keyof AppConfig, key: string, value: JsonValue | undefined) {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      const target = ((next[section] as JsonObject | undefined) ?? {}) as JsonObject;
      if (value === undefined || value === "") delete target[key];
      else target[key] = value;
      next[section] = target as never;
      return next;
    });
  }

  function updateProvider(index: number, patch: Partial<ProviderConfig>) {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      const list = [...(next.providers ?? [])];
      list[index] = { ...list[index], ...patch };
      next.providers = list;
      return next;
    });
  }

  function addProvider() {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      next.providers = [...(next.providers ?? []), { name: `provider-${(next.providers ?? []).length + 1}`, type: "openai", base_url: "" }];
      setSelectedProvider(next.providers.length - 1);
      setDetailTab("settings");
      setActiveView("provider-detail");
      return next;
    });
  }

  function addProviderFromPreset(preset: ProviderPreset) {
    const existingIndex = providers.findIndex((p) => p.name === preset.name);
    if (existingIndex >= 0) {
      setSelectedProvider(existingIndex);
      addToast(`${providerDisplayName(preset)} already exists`, "info");
      return;
    }
    setConfig((cur) => {
      const next = cloneConfig(cur);
      const existingNames = new Set((next.providers ?? []).map((p) => p.name));
      const name = uniqueName(preset.name, existingNames);
      const { description: _description, routes: presetRoutes, ...provider } = preset;
      next.providers = [...(next.providers ?? []), { ...provider, name }];
      const addedRoutes = (presetRoutes ?? []).map((route) => ({ ...route, provider: name }));
      next.routes = [...(next.routes ?? []), ...addedRoutes];
      setSelectedProvider(next.providers.length - 1);
      setDetailTab("settings");
      setActiveView("provider-detail");
      return next;
    });
  }

  function duplicateProvider(index: number) {
    setConfig((cur) => {
      const source = cur.providers?.[index];
      if (!source) return cur;
      const next = cloneConfig(cur);
      const existingNames = new Set((next.providers ?? []).map((p) => p.name));
      const copy = { ...source, name: uniqueName(`${source.name || "provider"}-copy`, existingNames) };
      next.providers = [...(next.providers ?? []), copy];
      setSelectedProvider(next.providers.length - 1);
      setDetailTab("settings");
      setActiveView("provider-detail");
      return next;
    });
  }

  function removeProvider(index: number) {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      const removedName = next.providers?.[index]?.name;
      next.providers = (next.providers ?? []).filter((_, i) => i !== index);
      if (removedName) {
        next.routes = (next.routes ?? [])
          .filter((route) => route.provider !== removedName)
          .map((route) => ({
            ...route,
            fallback_providers: (route.fallback_providers ?? []).filter((name) => name !== removedName),
          }));
      }
      setSelectedProvider(Math.max(0, Math.min(index, next.providers.length - 1)));
      return next;
    });
  }

  async function testProvider(provider: ProviderConfig) {
    try {
      const result = await invoke<ProbeResult>("probe_provider", { request: { ...provider, mode: "test" } });
      addToast(result.ok ? `${provider.name} connection ok` : result.error ?? `${provider.name} test failed`, result.ok ? "success" : "error");
    } catch (e) {
      addToast(String(e), "error");
    }
  }

  async function queryProviderUsage(provider: ProviderConfig) {
    try {
      const result = await invoke<ProbeResult>("probe_provider", { request: { ...provider, mode: "usage" } });
      if (result.ok) {
        addToast(`${provider.name} usage endpoint responded`, "success");
        setValidation({ ok: true, code: result.status, stdout: result.body || "usage endpoint ok", stderr: "" });
      } else {
        addToast(result.error ?? `${provider.name} usage endpoint not found`, "error");
      }
    } catch (e) {
      addToast(String(e), "error");
    }
  }

  async function fetchProviderModels(provider: ProviderConfig) {
    setModelProbeBusy(true);
    try {
      const result = await invoke<ProbeResult>("probe_provider", { request: { ...provider, mode: "models" } });
      if (!result.ok) {
        addToast(result.error ?? `${provider.name} model endpoint not found`, "error");
        return;
      }
      const models = parseProviderModels(result.body);
      setProviderModelCatalog((prev) => ({ ...prev, [provider.name]: models }));
      setValidation({ ok: true, code: result.status, stdout: models.length ? models.join("\n") : result.body || "model endpoint ok", stderr: "" });
      addToast(models.length ? `${models.length} models loaded` : `${provider.name} model endpoint responded`, "success");
    } catch (e) {
      addToast(String(e), "error");
    } finally {
      setModelProbeBusy(false);
    }
  }

  function updateProviderModelMapping(provider: ProviderConfig, kind: "claude" | "openai", patch: Partial<{ clientModel: string; upstreamModel: string }>) {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      const list = [...(next.routes ?? [])];
      const rows = providerMappingRows(provider, list);
      const current = rows.find((row) => row.kind === kind);
      if (!current) return cur;
      const routeIndex = current.routeIndex >= 0
        ? current.routeIndex
        : list.findIndex((route) =>
          route.provider === provider.name
          && route.match === current.clientModel
          && (route.match_type ?? "prefix") === "exact"
        );
      const route: RouteConfig = {
        match: patch.clientModel ?? current.clientModel,
        match_type: "exact",
        provider: provider.name,
        rewrite_model: patch.upstreamModel ?? current.upstreamModel,
        client_kind: kind,
      };
      if (routeIndex >= 0) list[routeIndex] = { ...list[routeIndex], ...route };
      else list.unshift(route);
      next.routes = list;
      return next;
    });
  }

  function addProviderModelMapping(provider: ProviderConfig, upstreamModel: string) {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      const existing = new Set((next.routes ?? [])
        .filter((route) => route.provider === provider.name)
        .map((route) => route.match));
      const baseClient = upstreamModel || DEFAULT_OPENAI_CLIENT_MODEL;
      const client = uniqueName(`${baseClient}--${providerSlug(provider)}`, existing);
      next.routes = [
        {
          match: client,
          match_type: "exact",
          provider: provider.name,
          rewrite_model: upstreamModel || baseClient,
          client_kind: "openai",
        },
        ...(next.routes ?? []),
      ];
      setSelectedRoute(0);
      setDetailTab("models");
      return next;
    });
  }

  function updateKeyWatch(pi: number, wi: number, patch: Partial<KeyWatchConfig>) {
    const watches = [...(providers[pi].key_watch ?? [])];
    watches[wi] = { ...watches[wi], ...patch };
    updateProvider(pi, { key_watch: watches });
  }

  function addKeyWatch(pi: number) {
    updateProvider(pi, {
      key_watch: [...(providers[pi].key_watch ?? []), { file: "", path: "", url_path: "" }],
      api_key: undefined, api_key_env: undefined, api_key_url: undefined, api_key_file: undefined,
    });
  }

  function updateRoute(index: number, patch: Partial<RouteConfig>) {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      const list = [...(next.routes ?? [])];
      list[index] = { ...list[index], ...patch };
      next.routes = list;
      return next;
    });
  }

  function addRoute() {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      next.routes = [...(next.routes ?? []), { match: "*", match_type: "prefix", provider: providerNames[0] ?? "", fallback_providers: [] }];
      setSelectedRoute(next.routes.length - 1);
      setDetailTab("routes");
      return next;
    });
  }

  function removeRoute(index: number) {
    setConfig((cur) => {
      const next = cloneConfig(cur);
      next.routes = (next.routes ?? []).filter((_, i) => i !== index);
      setSelectedRoute(0);
      return next;
    });
  }

  /* ── Render ───────────────────────────────────────────── */

  return (
    <div className="app-shell min-h-screen bg-bg text-fg">
      <AppSidebar
        activeView={activeView}
        darkMode={darkMode}
        provider={selectedProviderConfig}
        providerCount={providers.length}
        logCount={status?.logs.length ?? 0}
        status={status}
        onSetView={setActiveView}
        onToggleTheme={() => setDarkMode((d) => !d)}
      />

      {/* ── Main ───────────────────────────────── */}
      <main className={cn("h-screen min-w-0 flex-1 overflow-y-auto px-4", activeView === "providers" ? "py-3" : "py-4")}>
        <AnimatePresence mode="wait">
          {/* ── Dashboard ── */}
          {activeView === "dashboard" && (
            <DashboardView
              snapshot={snapshot}
              status={status}
              latestSample={latestSample}
              dashboard={dashboard}
              logs={status?.logs ?? []}
              onRefresh={() => void refreshDashboard()}
              refreshing={dashboardBusy}
            />
          )}

          {activeView === "usage-logs" && (
            <UsageLogsView logs={status?.logs ?? []} launches={recentLaunches} onBack={() => setActiveView("dashboard")} />
          )}

          {/* ── Providers List ── */}
          {activeView === "providers" && (
            <motion.div
              key="providers"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.2 }}
              className="grid w-full gap-2"
            >
              <div className="grid gap-2">
                <section className="rounded-lg border border-border bg-surface">
                  <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      {([
                        { value: "all" as PresetCategoryFilter, label: "All" },
                        { value: "official" as PresetCategoryFilter, label: "Official" },
                        { value: "aggregator" as PresetCategoryFilter, label: "Aggregator" },
                        { value: "china" as PresetCategoryFilter, label: "CN" },
                        { value: "local" as PresetCategoryFilter, label: "Local" },
                      ]).map((item) => (
                        <button
                          key={item.value}
                          type="button"
                          onClick={() => setPresetFilter(item.value)}
                          className={cn(
                            "inline-flex h-8 items-center rounded-lg px-3 text-xs font-bold transition-colors",
                            presetFilter === item.value
                              ? "bg-primary text-white"
                              : "bg-muted-soft text-muted hover:bg-border hover:text-heading"
                          )}
                        >
                          {item.label}
                        </button>
                      ))}
                    </div>
                    <div className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-1.5">
                      <div className="relative w-[220px] max-w-full">
                        <Search size={14} className="provider-search-icon pointer-events-none absolute top-1/2 -translate-y-1/2 text-icon" />
                        <input
                          value={providerSearch}
                          onChange={(event) => setProviderSearch(event.currentTarget.value)}
                          placeholder="Search providers"
                          className="provider-search-input h-9"
                        />
                      </div>
                      <IconAction
                        title={`Doctor ${doctorReadyCount}/${doctorItems.length}: ${doctorItems.map((item) => `${item.label} ${item.ok ? "ok" : item.detail}`).join("; ")}`}
                        icon={doctorReadyCount === doctorItems.length ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                        onClick={() => addToast(
                          doctorReadyCount === doctorItems.length
                            ? "Doctor passed"
                            : doctorItems.filter((item) => !item.ok).map((item) => `${item.label}: ${item.detail}`).join("; "),
                          doctorReadyCount === doctorItems.length ? "success" : "info"
                        )}
                      />
                      <IconAction title="Settings" icon={<Settings size={14} />} onClick={() => setActiveView("settings")} />
                      <IconAction title="Save config" icon={<Save size={14} />} onClick={() => void saveConfig()} />
                      <Button icon={<Plus size={14} />} onClick={addProvider}>Custom</Button>
                    </div>
                  </div>
                  <div className={cn("provider-preset-grid", shouldCollapsePresets && "is-collapsed", presetsExpanded && shouldOfferPresetExpand && "is-expanded")}>
                    {displayedProviderPresets.map((preset) => (
                      <button
                        key={preset.name}
                        type="button"
                        onClick={() => addProviderFromPreset(preset)}
                        className="provider-preset-card group/preset"
                        title={`${providerDisplayName(preset)} - ${preset.description}\n${preset.base_url}`}
                      >
                        <ProviderLogo provider={preset} className="preset-provider-logo text-xs" />
                        <span className="min-w-0 truncate text-sm font-bold text-heading">{providerDisplayName(preset)}</span>
                        {preset.featured ? (
                          <span className="preset-featured" title="Recommended preset">
                            <Star size={11} fill="currentColor" />
                          </span>
                        ) : null}
                      </button>
                    ))}
                    {shouldOfferPresetExpand && !providerQuery ? (
                      <button
                        type="button"
                        onClick={() => setPresetsExpanded((value) => !value)}
                        className="provider-preset-more"
                        title={presetsExpanded ? "Collapse presets" : "Show more presets"}
                      >
                        {presetsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        {presetsExpanded ? "Collapse" : "More"}
                      </button>
                    ) : null}
                    {!visibleProviderPresets.length && (
                      <div className="w-full rounded-lg border border-border bg-bg px-4 py-4 text-center text-sm text-muted">
                        No presets match "{providerSearch}".
                      </div>
                    )}
                  </div>
                </section>

                <section className="grid gap-3" aria-label="Providers">
                  {providers.length && visibleProviders.length ? visibleProviders.map(({ provider: p, index: i }) => {
                    const summary = providerSummaries[i];
                    return (
                      <Card
                        key={`${p.name}-${i}`}
                        active={selectedProvider === i}
                        className="provider-console-card"
                        onClick={() => { setSelectedProvider(i); setDetailTab("settings"); setActiveView("provider-detail"); }}
                      >
                        <GripVertical size={16} className="text-icon shrink-0 opacity-30 group-hover:opacity-70 transition-opacity" />
                        <ProviderLogo provider={p} className="size-12 text-base" />
                        <div className="grid min-w-0 flex-1 gap-3">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="flex min-w-0 items-center gap-2">
                                <strong className="truncate text-base font-bold text-heading">{providerDisplayName(p)}</strong>
                                <span className="rounded-full bg-info-soft px-2.5 py-0.5 text-xs font-bold text-primary">{p.type}</span>
                                <span className={cn(
                                  "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-bold",
                                  summary.status === "live" && "bg-success-soft text-success",
                                  summary.status === "configured" && "bg-muted-soft text-muted",
                                  summary.status === "attention" && "bg-danger-soft text-danger"
                                )}>
                                  <span className="h-1.5 w-1.5 rounded-full bg-current" />
                                  {summary.statusLabel}
                                </span>
                              </div>
                              <p className="mt-1 truncate font-mono text-xs text-muted">{p.base_url || "No base URL configured"}</p>
                            </div>
                            <div className="provider-action-strip flex shrink-0 items-center gap-1">
                              <IconAction title="Test provider" icon={<Activity size={14} />} onClick={(event) => { event.stopPropagation(); void testProvider(p); }} />
                              <IconAction title="Query usage" icon={<DollarSign size={14} />} onClick={(event) => { event.stopPropagation(); void queryProviderUsage(p); }} />
                              <IconAction title="Copy provider" icon={<Copy size={14} />} onClick={(event) => { event.stopPropagation(); duplicateProvider(i); }} />
                              <IconAction title="Delete provider" danger icon={<Trash2 size={14} />} onClick={(event) => { event.stopPropagation(); removeProvider(i); }} />
                            </div>
                          </div>
                          <div className="grid gap-2 sm:grid-cols-4">
                            <ProviderMiniStat icon={<Database size={13} />} label="Key source" value={summary.key.label} detail={summary.key.detail} ok={summary.key.ok} />
                            <ProviderMiniStat icon={<Route size={13} />} label="Routes" value={fmtInt(summary.routeCount)} detail="primary matches" />
                            <ProviderMiniStat icon={<Network size={13} />} label="Fallbacks" value={fmtInt(summary.fallbackCount)} detail="referencing this" />
                            <ProviderMiniStat icon={<Gauge size={13} />} label="Traffic" value={fmtInt(summary.requests)} detail={summary.errors ? `${fmtInt(summary.errors)} errors` : "requests"} ok={!summary.errors} />
                          </div>
                        </div>
                      </Card>
                    );
                  }) : providers.length ? (
                    <EmptyState compact title="No matching providers" action="Clear search" onAction={() => setProviderSearch("")} />
                  ) : (
                    <EmptyState compact title="No providers" action="Add provider" onAction={addProvider} />
                  )}
                </section>
              </div>
            </motion.div>
          )}

          {/* ── Provider Detail Page ── */}
          {activeView === "provider-detail" && selectedProviderConfig && (
            <motion.div
              key="provider-detail"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="w-full space-y-5"
            >
              {/* Provider name header */}
              <div className="flex items-center gap-3">
                <Button icon={<ChevronDown size={14} className="rotate-90" />} onClick={() => setActiveView("providers")}>Back</Button>
                <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-bg text-lg font-bold text-primary">
                  {selectedProviderConfig.name?.slice(0, 1).toUpperCase() || "P"}
                </div>
                <div>
                  <h2 className="text-lg font-bold text-heading">{selectedProviderConfig.name || "Unnamed Provider"}</h2>
                  <span className="text-xs text-muted">{selectedProviderConfig.type}</span>
                </div>
              </div>

              {/* Service bar */}
              <div className="rounded-2xl border border-border bg-surface p-5 shadow-sm">
                <div className="flex flex-wrap items-center gap-3">
                  <div className={cn(
                    "flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium",
                    status?.running
                      ? status.source === "external" ? "bg-accent-soft text-accent" : "bg-success-soft text-success"
                      : "bg-muted-soft text-muted"
                  )}>
                    <Terminal size={14} />
                    <span>{gatewayStatusLabel(status)}</span>
                    {status?.pid && <strong className="font-bold">PID {status.pid}</strong>}
                  </div>
                  <Button variant="success" icon={<Play size={14} />} onClick={startServer} disabled={busy || status?.running}>Start</Button>
                  <Button variant="danger" icon={<Square size={14} />} onClick={stopServer} disabled={busy || !status?.managed}>Stop</Button>
                  <Button icon={<RotateCw size={14} />} onClick={restartServer} disabled={busy}>Restart</Button>
                  <Button variant="primary" icon={<Save size={14} />} onClick={saveConfig} disabled={busy}>Save</Button>
                  <Button icon={<CheckCircle2 size={14} />} onClick={validateConfig} disabled={busy}>Validate</Button>
                  <div className="ml-auto flex items-center gap-3 border-l border-border pl-4">
                    <span className="text-xs font-semibold text-muted">Workspace launches live in Launcher.</span>
                    <Button icon={<FolderOpen size={14} />} onClick={() => setActiveView("launcher")}>Open Launcher</Button>
                  </div>
                </div>
              </div>

              {/* Sub-tabs: Settings | Routes | Runtime */}
              <nav className="flex items-center gap-1 rounded-xl bg-muted-soft p-1">
                {([
                  { tab: "settings" as DetailTab, icon: <Settings size={14} />, label: "Settings" },
                  { tab: "models" as DetailTab, icon: <Database size={14} />, label: "Models" },
                  { tab: "routes" as DetailTab, icon: <Route size={14} />, label: "Routes" },
                  { tab: "runtime" as DetailTab, icon: <Gauge size={14} />, label: "Runtime" },
                ]).map(({ tab, icon, label }) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => setDetailTab(tab)}
                    className={cn(
                      "relative flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-all duration-200",
                      detailTab === tab ? "text-heading" : "text-muted hover:text-icon-hover"
                    )}
                  >
                    {detailTab === tab && (
                      <motion.div
                        layoutId="detailTab"
                        className="absolute inset-0 rounded-lg bg-surface shadow-sm"
                        transition={{ type: "spring", stiffness: 400, damping: 30 }}
                      />
                    )}
                    <span className="relative z-10 flex items-center gap-2">{icon} {label}</span>
                  </button>
                ))}
              </nav>

              {/* Sub-tab content */}
              <AnimatePresence mode="wait">
                {detailTab === "settings" && (
                  <motion.div key="dt-settings" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }}>
                    <div className="rounded-2xl border border-border bg-surface shadow-sm overflow-hidden">
                      <PanelHeader title="Provider Settings" subtitle="Use env, URL, file, or watched config paths for API keys." />
                      <div className="grid gap-5 p-6">
                        <div className="grid grid-cols-2 gap-4">
                          <TextField label="Provider Name" value={selectedProviderConfig.name} onChange={(v) => updateProvider(selectedProvider, { name: v })} />
                          <SelectField label="Type" value={selectedProviderConfig.type} options={["openai", "openai_responses", "anthropic", "gemini"]} onChange={(v) => updateProvider(selectedProvider, { type: v })} />
                        </div>
                        <TextField label="Base URL" value={selectedProviderConfig.base_url} onChange={(v) => updateProvider(selectedProvider, { base_url: v })} />
                        <TextField secret label="API Key (direct)" value={selectedProviderConfig.api_key ?? ""} onChange={(v) => updateProvider(selectedProvider, { api_key: v || undefined, api_key_env: undefined, api_key_url: undefined, api_key_file: undefined, key_watch: undefined })} />
                        <div className="grid grid-cols-2 gap-4">
                          <TextField label="API key env" value={selectedProviderConfig.api_key_env ?? ""} onChange={(v) => updateProvider(selectedProvider, { api_key_env: v || undefined, api_key: undefined, api_key_url: undefined, api_key_file: undefined, key_watch: undefined })} />
                          <TextField label="API key URL" value={selectedProviderConfig.api_key_url ?? ""} onChange={(v) => updateProvider(selectedProvider, { api_key_url: v || undefined, api_key: undefined, api_key_env: undefined, api_key_file: undefined, key_watch: undefined })} />
                        </div>
                        <TextField label="API key file" value={selectedProviderConfig.api_key_file ?? ""} onChange={(v) => updateProvider(selectedProvider, { api_key_file: v || undefined, api_key: undefined, api_key_env: undefined, api_key_url: undefined, key_watch: undefined })} />
                        <div className="flex justify-end gap-2">
                          <Button icon={<Activity size={14} />} onClick={() => addKeyWatch(selectedProvider)}>Add key watch</Button>
                          <Button variant="danger" icon={<Trash2 size={14} />} onClick={() => { removeProvider(selectedProvider); setActiveView("providers"); }}>Remove provider</Button>
                        </div>
                        {(selectedProviderConfig.key_watch ?? []).map((w, wi) => (
                          <div key={wi} className="grid gap-3 rounded-xl border border-border bg-muted-soft p-4">
                            <TextField label="Watch file" value={w.file} onChange={(v) => updateKeyWatch(selectedProvider, wi, { file: v })} />
                            <TextField label="Key path" value={w.path} onChange={(v) => updateKeyWatch(selectedProvider, wi, { path: v })} />
                            <TextField label="URL path" value={w.url_path ?? ""} onChange={(v) => updateKeyWatch(selectedProvider, wi, { url_path: v || undefined })} />
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}

                {detailTab === "models" && (
                  <motion.div key="dt-models" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }} className="grid gap-5">
                    <section className="rounded-2xl border border-border bg-surface shadow-sm overflow-hidden">
                      <PanelHeader title="Model Mappings" subtitle="Client-visible model names are exact routes; upstream models are sent to this provider." />
                      <div className="grid gap-4 p-5">
                        {providerMappingRows(selectedProviderConfig, routes).map((row) => (
                          <div key={row.kind} className="grid gap-3 rounded-xl border border-border bg-bg p-4">
                            <div className="flex items-center justify-between gap-3">
                              <div>
                                <h3 className="text-sm font-bold text-heading">{row.label}</h3>
                                <p className="mt-1 text-xs text-muted">{row.routeIndex >= 0 ? "Custom exact route" : "Generated fallback route"}</p>
                              </div>
                              <span className="rounded-full bg-info-soft px-2.5 py-1 text-[11px] font-bold text-primary">{row.kind}</span>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                              <TextField label="Client model name" value={row.clientModel} onChange={(v) => updateProviderModelMapping(selectedProviderConfig, row.kind, { clientModel: v })} />
                              <TextField label="Upstream model" value={row.upstreamModel} onChange={(v) => updateProviderModelMapping(selectedProviderConfig, row.kind, { upstreamModel: v })} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>

                    <section className="rounded-2xl border border-border bg-surface shadow-sm overflow-hidden">
                      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4">
                        <div>
                          <h2 className="text-sm font-bold text-heading">Available Models</h2>
                          <p className="mt-1 text-xs text-muted">Fetched from the provider model endpoint when available.</p>
                        </div>
                        <Button icon={<RefreshCw size={14} />} onClick={() => void fetchProviderModels(selectedProviderConfig)} disabled={modelProbeBusy}>
                          {modelProbeBusy ? "Loading" : "Fetch models"}
                        </Button>
                      </div>
                      <div className="max-h-80 overflow-auto p-4">
                        {(providerModelCatalog[selectedProviderConfig.name] ?? []).length ? (
                          <div className="grid gap-2">
                            {(providerModelCatalog[selectedProviderConfig.name] ?? []).map((model) => (
                              <div key={model} className="flex items-center justify-between gap-3 rounded-lg border border-border bg-bg px-3 py-2">
                                <span className="min-w-0 truncate font-mono text-xs text-fg">{model}</span>
                                <div className="flex shrink-0 gap-2">
                                  <Button icon={<Route size={13} />} onClick={() => updateProviderModelMapping(selectedProviderConfig, "claude", { upstreamModel: model })}>Use for Claude</Button>
                                  <Button icon={<Route size={13} />} onClick={() => updateProviderModelMapping(selectedProviderConfig, "openai", { upstreamModel: model })}>Use for Codex</Button>
                                  <Button icon={<Plus size={13} />} onClick={() => addProviderModelMapping(selectedProviderConfig, model)}>Add alias</Button>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <EmptyState compact title="No models loaded" action="Fetch models" onAction={() => void fetchProviderModels(selectedProviderConfig)} />
                        )}
                      </div>
                    </section>
                  </motion.div>
                )}

                {detailTab === "routes" && (
                  <motion.div key="dt-routes" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }} className="grid grid-cols-[minmax(300px,1fr)_minmax(280px,0.6fr)] gap-5 items-start">
                    {/* Routes list */}
                    <section className="flex flex-col gap-3" aria-label="Routes">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold text-heading">Routes</h3>
                        <Button icon={<Plus size={14} />} onClick={addRoute}>Add route</Button>
                      </div>
                      {routes.length ? routes.map((r, i) => (
                        <Card key={`${r.match}-${i}`} active={selectedRoute === i} onClick={() => setSelectedRoute(i)}>
                          <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-border bg-bg shrink-0">
                            <Network size={16} className="text-purple-500" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <strong className="truncate text-sm font-bold text-heading">{r.match || "Empty"}</strong>
                              <span className="rounded-full bg-info-soft px-2 py-0.5 text-[11px] font-semibold text-primary">{r.match_type ?? "prefix"}</span>
                            </div>
                            <p className="mt-0.5 truncate text-xs text-muted">
                              {r.provider || "No provider"}{r.rewrite_model ? ` → ${r.rewrite_model}` : ""}
                            </p>
                          </div>
                          {r.fallback_providers?.length ? (
                            <span className="rounded-full bg-success-soft px-2.5 py-0.5 text-[11px] font-bold text-success">fallback</span>
                          ) : null}
                        </Card>
                      )) : (
                        <EmptyState title="No routes" action="Add route" onAction={addRoute} />
                      )}
                    </section>
                    {/* Route detail */}
                    <section className="rounded-2xl border border-border bg-surface shadow-sm overflow-hidden">
                      {selectedRouteConfig ? (
                        <>
                          <PanelHeader title="Route Details" subtitle="Match model names, choose a provider, optionally rewrite." />
                          <div className="grid gap-4 p-5">
                            <div className="grid grid-cols-2 gap-3">
                              <TextField label="Match" value={selectedRouteConfig.match} onChange={(v) => updateRoute(selectedRoute, { match: v })} />
                              <SelectField label="Match type" value={selectedRouteConfig.match_type ?? "prefix"} options={["prefix", "exact"]} onChange={(v) => updateRoute(selectedRoute, { match_type: v as "prefix" | "exact" })} />
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                              <SelectField label="Provider" value={selectedRouteConfig.provider} options={providerNames} onChange={(v) => updateRoute(selectedRoute, { provider: v })} />
                              <TextField label="Rewrite model" value={selectedRouteConfig.rewrite_model ?? ""} onChange={(v) => updateRoute(selectedRoute, { rewrite_model: v || undefined })} />
                            </div>
                            <SelectField label="Client kind" value={selectedRouteConfig.client_kind ?? ""} options={["", "claude", "openai"]} onChange={(v) => updateRoute(selectedRoute, { client_kind: (v || undefined) as "claude" | "openai" | undefined })} />
                            <TextField label="Fallback providers CSV" value={(selectedRouteConfig.fallback_providers ?? []).join(", ")} onChange={(v) => updateRoute(selectedRoute, { fallback_providers: splitCsv(v) })} />
                            <div className="flex justify-end">
                              <Button variant="danger" icon={<Trash2 size={14} />} onClick={() => removeRoute(selectedRoute)}>Remove</Button>
                            </div>
                          </div>
                        </>
                      ) : <EmptyState title="No route selected" action="Add route" onAction={addRoute} />}
                    </section>
                  </motion.div>
                )}

                {detailTab === "runtime" && (
                  <motion.div key="dt-runtime" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }}>
                    <div className="rounded-2xl border border-border bg-surface shadow-sm overflow-hidden">
                      <PanelHeader title="Runtime Settings" subtitle="Server, access, metrics, prompt cache, and logging." />
                      <div className="grid grid-cols-2 gap-5 p-6">
                        <TextField label="Listen" value={valueAsString(config.server?.listen)} onChange={(v) => updateSection("server", "listen", v)} />
                        <NumberField label="Timeout seconds" value={valueAsNumber(config.server?.request_timeout_secs)} onChange={(v) => updateSection("server", "request_timeout_secs", v)} />
                        <NumberField label="Body limit MB" value={valueAsNumber(config.server?.body_limit_mb)} onChange={(v) => updateSection("server", "body_limit_mb", v)} />
                        <NumberField label="Max concurrency" optional value={valueAsNumber(config.server?.max_concurrent_requests)} onChange={(v) => updateSection("server", "max_concurrent_requests", v)} />
                        <NumberField label="Rate/minute" optional value={valueAsNumber(config.server?.rate_limit_per_minute)} onChange={(v) => updateSection("server", "rate_limit_per_minute", v)} />
                        <SelectField label="Reasoning policy" value={valueAsString(config.server?.reasoning_policy || "fill_missing")} options={["preserve", "fill_missing", "cap", "force"]} onChange={(v) => updateSection("server", "reasoning_policy", v)} />
                        <SelectField label="Default reasoning" value={valueAsString(config.server?.default_reasoning_effort)} options={REASONING_OPTIONS} onChange={(v) => updateSection("server", "default_reasoning_effort", v)} />
                        <SelectField label="Max reasoning" value={valueAsString(config.server?.max_reasoning_effort)} options={REASONING_OPTIONS} onChange={(v) => updateSection("server", "max_reasoning_effort", v)} />
                        <NumberField label="Retry attempts" value={valueAsNumber(config.server?.retry_attempts)} onChange={(v) => updateSection("server", "retry_attempts", v)} />
                        <NumberField label="Retry backoff ms" value={valueAsNumber(config.server?.retry_backoff_ms)} onChange={(v) => updateSection("server", "retry_backoff_ms", v)} />
                        <NumberField label="Circuit failures" optional value={valueAsNumber(config.server?.circuit_breaker_failures)} onChange={(v) => updateSection("server", "circuit_breaker_failures", v)} />
                        <NumberField label="Circuit cooldown" optional value={valueAsNumber(config.server?.circuit_breaker_cooldown_secs)} onChange={(v) => updateSection("server", "circuit_breaker_cooldown_secs", v)} />
                          <SelectField label="Log level" value={valueAsString(config.logging?.level)} options={["trace", "debug", "info", "warn", "error"]} onChange={(v) => updateSection("logging", "level", v)} />
                          <SelectField label="Log format" value={valueAsString(config.logging?.format)} options={["text", "json"]} onChange={(v) => updateSection("logging", "format", v)} />
                          <BoolField label="ANSI logs" checked={valueAsBool(config.logging?.ansi)} onChange={(v) => updateSection("logging", "ansi", v)} />
                        <BoolField label="Auth enabled" checked={valueAsBool(config.auth?.enabled)} onChange={(v) => updateSection("auth", "enabled", v)} />
                        <TextField label="API keys env" value={valueAsString(config.auth?.api_keys_env)} onChange={(v) => updateSection("auth", "api_keys_env", v)} />
                        <NumberField label="Per-key rate/minute" optional value={valueAsNumber(config.auth?.per_key_rate_limit_per_minute)} onChange={(v) => updateSection("auth", "per_key_rate_limit_per_minute", v)} />
                        <NumberField label="Per-key concurrency" optional value={valueAsNumber(config.auth?.per_key_max_concurrent_requests)} onChange={(v) => updateSection("auth", "per_key_max_concurrent_requests", v)} />
                        <BoolField label="Metrics enabled" checked={valueAsBool(config.metrics?.enabled, true)} onChange={(v) => updateSection("metrics", "enabled", v)} />
                        <BoolField label="Anthropic cache control" checked={valueAsBool(config.prompt_cache?.auto_inject_anthropic_cache_control, true)} onChange={(v) => updateSection("prompt_cache", "auto_inject_anthropic_cache_control", v)} />
                        <BoolField label="Cache system" checked={valueAsBool(config.prompt_cache?.cache_system, true)} onChange={(v) => updateSection("prompt_cache", "cache_system", v)} />
                        <BoolField label="Cache tools" checked={valueAsBool(config.prompt_cache?.cache_tools, true)} onChange={(v) => updateSection("prompt_cache", "cache_tools", v)} />
                        <BoolField label="Cache last user" checked={valueAsBool(config.prompt_cache?.cache_last_user_message, true)} onChange={(v) => updateSection("prompt_cache", "cache_last_user_message", v)} />
                        <TextField label="OpenAI cache key" value={valueAsString(config.prompt_cache?.openai_prompt_cache_key)} onChange={(v) => updateSection("prompt_cache", "openai_prompt_cache_key", v)} />
                        <TextField label="Retention" value={valueAsString(config.prompt_cache?.openai_prompt_cache_retention)} onChange={(v) => updateSection("prompt_cache", "openai_prompt_cache_retention", v)} />
                        <BoolField label="Debug shape" checked={valueAsBool(config.prompt_cache?.debug_log_request_shape, true)} onChange={(v) => updateSection("prompt_cache", "debug_log_request_shape", v)} />
                        <TextField label="Relocate byte range" value={valueAsString(config.prompt_cache?.relocate_system_prefix_range)} onChange={(v) => updateSection("prompt_cache", "relocate_system_prefix_range", v)} />
                        <BoolField label="Log relocated text" checked={valueAsBool(config.prompt_cache?.log_relocated_system_text)} onChange={(v) => updateSection("prompt_cache", "log_relocated_system_text", v)} />
                        <TextField label="Strip prefixes CSV" value={Array.isArray(config.prompt_cache?.strip_system_line_prefixes) ? (config.prompt_cache?.strip_system_line_prefixes as JsonValue[]).join(", ") : ""} onChange={(v) => updateSection("prompt_cache", "strip_system_line_prefixes", splitCsv(v))} />
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Validation */}
              <div className="rounded-2xl border border-border bg-surface overflow-hidden">
                <h2 className="border-b border-border px-5 py-4 text-sm font-bold text-heading">Validation</h2>
                <pre className={cn(
                  "h-40 overflow-auto p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap",
                  validation?.ok ? "text-success" : "text-fg"
                )}>
                  {validation ? `${validation.stdout}${validation.stderr}`.trim() || "config ok" : "Not validated yet."}
                </pre>
              </div>
            </motion.div>
          )}

          {/* ── Settings View ── */}
          {activeView === "settings" && (
            <motion.div
              key="settings"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.2 }}
              className="grid w-full gap-4"
            >
              <section className="rounded-lg border border-border bg-surface">
                <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
                  <div className="flex items-start gap-3">
                    <Button icon={<ChevronDown size={14} className="rotate-90" />} onClick={() => setActiveView("launcher")}>Back</Button>
                    <div>
                    <p className="text-xs font-bold uppercase text-muted">Global settings</p>
                    <h1 className="mt-1 text-xl font-bold text-heading">Settings</h1>
                    <p className="mt-1 text-sm text-muted">Runtime, security, cache, data, and desktop preferences.</p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button icon={<CheckCircle2 size={14} />} onClick={validateConfig} disabled={busy}>Validate</Button>
                    <Button variant="primary" icon={<Save size={14} />} onClick={saveConfig} disabled={busy}>Save</Button>
                  </div>
                </div>
              </section>

              <section className="rounded-lg border border-border bg-surface p-1">
                <div className="grid grid-cols-2 gap-1 sm:grid-cols-3 lg:grid-cols-6">
                  {([
                    { tab: "general" as SettingsTab, icon: <Monitor size={15} />, label: "General" },
                    { tab: "runtime" as SettingsTab, icon: <Gauge size={15} />, label: "Runtime" },
                    { tab: "security" as SettingsTab, icon: <ShieldCheck size={15} />, label: "Security" },
                    { tab: "cache" as SettingsTab, icon: <Database size={15} />, label: "Cache" },
                    { tab: "advanced" as SettingsTab, icon: <SlidersHorizontal size={15} />, label: "Advanced" },
                    { tab: "about" as SettingsTab, icon: <Activity size={15} />, label: "About" },
                  ]).map(({ tab, icon, label }) => (
                    <SettingsTabButton
                      key={tab}
                      active={settingsTab === tab}
                      icon={icon}
                      label={label}
                      onClick={() => setSettingsTab(tab)}
                    />
                  ))}
                </div>
              </section>

              <AnimatePresence mode="wait">
                {settingsTab === "general" && (
                  <motion.div key="settings-general" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }} className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
                    <section className="rounded-lg border border-border bg-surface overflow-hidden">
                      <PanelHeader title="Appearance" subtitle="Theme and launcher defaults." />
                      <div className="grid gap-4 p-5">
                        <div>
                          <span className="mb-2 block text-sm font-semibold text-label">Theme</span>
                          <div className="grid grid-cols-2 gap-2 rounded-lg border border-border bg-bg p-1">
                            <button type="button" onClick={() => setDarkMode(false)} className={cn("inline-flex h-10 items-center justify-center gap-2 rounded-md text-sm font-semibold transition-colors", !darkMode ? "bg-primary text-white" : "text-muted hover:bg-surface-hover hover:text-heading")}>
                              <Sun size={15} /> Light
                            </button>
                            <button type="button" onClick={() => setDarkMode(true)} className={cn("inline-flex h-10 items-center justify-center gap-2 rounded-md text-sm font-semibold transition-colors", darkMode ? "bg-primary text-white" : "text-muted hover:bg-surface-hover hover:text-heading")}>
                              <Moon size={15} /> Dark
                            </button>
                          </div>
                        </div>
                        <SelectField label="Default launcher" value={launchTool} options={["codex", "claude", "opencode"]} onChange={(v) => setLaunchTool(v as AITool)} />
                        <SettingsInfoRow icon={<Bell size={15} />} label="Desktop notifications" value="Planned" tone="muted" />
                        <SettingsInfoRow icon={<Wrench size={15} />} label="Auto start ferryllm" value="Planned" tone="muted" />
                      </div>
                    </section>

                    <section className="rounded-lg border border-border bg-surface overflow-hidden">
                      <PanelHeader title="Local Files" subtitle="Executable and generated config paths." />
                      <div className="grid gap-3 p-5">
                        <SettingsInfoRow icon={<Terminal size={15} />} label="Executable" value={executable || "ferryllm"} />
                        <SettingsInfoRow icon={<FileCog size={15} />} label="Config path" value={status?.config_path || "Default app config"} />
                        <SettingsInfoRow icon={<Activity size={15} />} label="Runtime" value={gatewayStatusDetail(status)} tone={gatewayStatusTone(status)} />
                        <SettingsInfoRow icon={<Route size={15} />} label="Routes" value={`${routes.length} configured`} />
                        <SettingsInfoRow icon={<Network size={15} />} label="Providers" value={`${providers.length} configured`} />
                      </div>
                    </section>
                  </motion.div>
                )}

                {settingsTab === "runtime" && (
                  <motion.div key="settings-runtime" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }} className="grid gap-5">
                    <section className="rounded-lg border border-border bg-surface overflow-hidden">
                      <PanelHeader title="Server" subtitle="Listener, limits, request pressure, and default reasoning." />
                      <div className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-3">
                        <TextField label="Listen" value={valueAsString(config.server?.listen)} onChange={(v) => updateSection("server", "listen", v)} />
                        <NumberField label="Timeout seconds" value={valueAsNumber(config.server?.request_timeout_secs)} onChange={(v) => updateSection("server", "request_timeout_secs", v)} />
                        <NumberField label="Body limit MB" value={valueAsNumber(config.server?.body_limit_mb)} onChange={(v) => updateSection("server", "body_limit_mb", v)} />
                        <NumberField label="Max concurrency" optional value={valueAsNumber(config.server?.max_concurrent_requests)} onChange={(v) => updateSection("server", "max_concurrent_requests", v)} />
                        <NumberField label="Rate/minute" optional value={valueAsNumber(config.server?.rate_limit_per_minute)} onChange={(v) => updateSection("server", "rate_limit_per_minute", v)} />
                        <SelectField label="Reasoning policy" value={valueAsString(config.server?.reasoning_policy || "fill_missing")} options={["preserve", "fill_missing", "cap", "force"]} onChange={(v) => updateSection("server", "reasoning_policy", v)} />
                        <SelectField label="Default reasoning" value={valueAsString(config.server?.default_reasoning_effort)} options={REASONING_OPTIONS} onChange={(v) => updateSection("server", "default_reasoning_effort", v)} />
                        <SelectField label="Max reasoning" value={valueAsString(config.server?.max_reasoning_effort)} options={REASONING_OPTIONS} onChange={(v) => updateSection("server", "max_reasoning_effort", v)} />
                      </div>
                    </section>

                    <section className="grid gap-5 lg:grid-cols-2">
                      <div className="rounded-lg border border-border bg-surface overflow-hidden">
                        <PanelHeader title="Reliability" subtitle="Retry and circuit breaker defaults." />
                        <div className="grid gap-4 p-5 sm:grid-cols-2">
                          <NumberField label="Retry attempts" value={valueAsNumber(config.server?.retry_attempts)} onChange={(v) => updateSection("server", "retry_attempts", v)} />
                          <NumberField label="Retry backoff ms" value={valueAsNumber(config.server?.retry_backoff_ms)} onChange={(v) => updateSection("server", "retry_backoff_ms", v)} />
                          <NumberField label="Circuit failures" optional value={valueAsNumber(config.server?.circuit_breaker_failures)} onChange={(v) => updateSection("server", "circuit_breaker_failures", v)} />
                          <NumberField label="Circuit cooldown" optional value={valueAsNumber(config.server?.circuit_breaker_cooldown_secs)} onChange={(v) => updateSection("server", "circuit_breaker_cooldown_secs", v)} />
                        </div>
                      </div>

                      <div className="rounded-lg border border-border bg-surface overflow-hidden">
                        <PanelHeader title="Logging & Metrics" subtitle="Local logs and Prometheus endpoint." />
                        <div className="grid gap-4 p-5 sm:grid-cols-2">
                        <SelectField label="Log level" value={valueAsString(config.logging?.level)} options={["trace", "debug", "info", "warn", "error"]} onChange={(v) => updateSection("logging", "level", v)} />
                        <SelectField label="Log format" value={valueAsString(config.logging?.format)} options={["text", "json"]} onChange={(v) => updateSection("logging", "format", v)} />
                        <BoolField label="ANSI logs" checked={valueAsBool(config.logging?.ansi)} onChange={(v) => updateSection("logging", "ansi", v)} />
                          <BoolField label="Metrics enabled" checked={valueAsBool(config.metrics?.enabled, true)} onChange={(v) => updateSection("metrics", "enabled", v)} />
                          <SettingsInfoRow icon={<BarChart3 size={15} />} label="Dashboard refresh" value="3s" />
                        </div>
                      </div>
                    </section>
                  </motion.div>
                )}

                {settingsTab === "security" && (
                  <motion.div key="settings-security" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }} className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]">
                    <section className="rounded-lg border border-border bg-surface overflow-hidden">
                      <PanelHeader title="Access Control" subtitle="Optional local auth and per-key pressure controls." />
                      <div className="grid gap-4 p-5">
                        <BoolField label="Auth enabled" checked={valueAsBool(config.auth?.enabled)} onChange={(v) => updateSection("auth", "enabled", v)} />
                        <TextField label="API keys env" value={valueAsString(config.auth?.api_keys_env)} onChange={(v) => updateSection("auth", "api_keys_env", v)} />
                        <NumberField label="Per-key rate/minute" optional value={valueAsNumber(config.auth?.per_key_rate_limit_per_minute)} onChange={(v) => updateSection("auth", "per_key_rate_limit_per_minute", v)} />
                        <NumberField label="Per-key concurrency" optional value={valueAsNumber(config.auth?.per_key_max_concurrent_requests)} onChange={(v) => updateSection("auth", "per_key_max_concurrent_requests", v)} />
                      </div>
                    </section>

                    <section className="rounded-lg border border-border bg-surface overflow-hidden">
                      <PanelHeader title="Secrets" subtitle="Key handling status for this desktop session." />
                      <div className="grid gap-3 p-5">
                        <SettingsInfoRow icon={<Lock size={15} />} label="Direct key input" value="Masked by default" tone="success" />
                        <SettingsInfoRow icon={<HardDriveDownload size={15} />} label="Autosave direct keys" value="Excluded" tone="success" />
                        <SettingsInfoRow icon={<Save size={15} />} label="Runnable config" value="Explicit Save/Start only" />
                        <SettingsInfoRow icon={<KeyRound size={15} />} label="Provider key sources" value={`${providers.filter((provider) => providerKeyInfo(provider).ok).length}/${providers.length || 0} ready`} tone={providers.length && providers.every((provider) => providerKeyInfo(provider).ok) ? "success" : "warning"} />
                        <SettingsInfoRow icon={<ShieldCheck size={15} />} label="Credential vault" value="Planned" tone="muted" />
                      </div>
                    </section>
                  </motion.div>
                )}

                {settingsTab === "cache" && (
                  <motion.div key="settings-cache" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }} className="grid gap-5">
                    <section className="rounded-lg border border-border bg-surface overflow-hidden">
                      <PanelHeader title="Prompt Cache" subtitle="Anthropic cache control and OpenAI prompt-cache hints." />
                      <div className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-3">
                        <BoolField label="Anthropic cache control" checked={valueAsBool(config.prompt_cache?.auto_inject_anthropic_cache_control, true)} onChange={(v) => updateSection("prompt_cache", "auto_inject_anthropic_cache_control", v)} />
                        <BoolField label="Cache system" checked={valueAsBool(config.prompt_cache?.cache_system, true)} onChange={(v) => updateSection("prompt_cache", "cache_system", v)} />
                        <BoolField label="Cache tools" checked={valueAsBool(config.prompt_cache?.cache_tools, true)} onChange={(v) => updateSection("prompt_cache", "cache_tools", v)} />
                        <BoolField label="Cache last user" checked={valueAsBool(config.prompt_cache?.cache_last_user_message, true)} onChange={(v) => updateSection("prompt_cache", "cache_last_user_message", v)} />
                        <BoolField label="Debug shape" checked={valueAsBool(config.prompt_cache?.debug_log_request_shape, true)} onChange={(v) => updateSection("prompt_cache", "debug_log_request_shape", v)} />
                        <BoolField label="Log relocated text" checked={valueAsBool(config.prompt_cache?.log_relocated_system_text)} onChange={(v) => updateSection("prompt_cache", "log_relocated_system_text", v)} />
                        <TextField label="OpenAI cache key" value={valueAsString(config.prompt_cache?.openai_prompt_cache_key)} onChange={(v) => updateSection("prompt_cache", "openai_prompt_cache_key", v)} />
                        <TextField label="Retention" value={valueAsString(config.prompt_cache?.openai_prompt_cache_retention)} onChange={(v) => updateSection("prompt_cache", "openai_prompt_cache_retention", v)} />
                        <TextField label="Relocate byte range" value={valueAsString(config.prompt_cache?.relocate_system_prefix_range)} onChange={(v) => updateSection("prompt_cache", "relocate_system_prefix_range", v)} />
                        <div className="sm:col-span-2 lg:col-span-3">
                          <TextField label="Strip prefixes CSV" value={Array.isArray(config.prompt_cache?.strip_system_line_prefixes) ? (config.prompt_cache?.strip_system_line_prefixes as JsonValue[]).join(", ") : ""} onChange={(v) => updateSection("prompt_cache", "strip_system_line_prefixes", splitCsv(v))} />
                        </div>
                      </div>
                    </section>
                  </motion.div>
                )}

                {settingsTab === "advanced" && (
                  <motion.div key="settings-advanced" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }} className="grid gap-5">
                    <section className="grid gap-5 lg:grid-cols-2">
                      <div className="rounded-lg border border-border bg-surface overflow-hidden">
                        <PanelHeader title="Data & Sync" subtitle="Config movement and backup entry points." />
                        <div className="grid gap-3 p-5">
                          <SettingsFeatureRow icon={<Upload size={15} />} title="Import config" status="Planned" />
                          <SettingsFeatureRow icon={<Download size={15} />} title="Export config" status="Planned" />
                          <SettingsFeatureRow icon={<HardDriveDownload size={15} />} title="Backup & rollback" status="Partial" detail="Save uses .bak rollback" />
                          <SettingsFeatureRow icon={<Cloud size={15} />} title="WebDAV sync" status="Planned" />
                        </div>
                      </div>

                      <div className="rounded-lg border border-border bg-surface overflow-hidden">
                        <PanelHeader title="Provider Tooling" subtitle="Operational controls planned around providers." />
                        <div className="grid gap-3 p-5">
                          <SettingsFeatureRow icon={<DollarSign size={15} />} title="Usage adapters" status="Partial" detail="Provider cards can probe common endpoints" />
                          <SettingsFeatureRow icon={<RefreshCw size={15} />} title="Model fetch dropdown" status="Planned" />
                          <SettingsFeatureRow icon={<Route size={15} />} title="Failover queue editor" status="Planned" />
                          <SettingsFeatureRow icon={<Network size={15} />} title="Global outbound proxy" status="Planned" />
                        </div>
                      </div>
                    </section>

                    <section className="rounded-lg border border-border bg-surface overflow-hidden">
                      <div className="flex items-center justify-between border-b border-border px-5 py-4">
                        <div>
                          <h2 className="text-sm font-bold text-heading">Validation</h2>
                          <p className="mt-1 text-xs text-muted">Result from ferryllm config validation.</p>
                        </div>
                        <Button icon={<CheckCircle2 size={14} />} onClick={validateConfig} disabled={busy}>Run</Button>
                      </div>
                      <pre className={cn(
                        "h-36 overflow-auto p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap",
                        validation?.ok ? "text-success" : "text-fg"
                      )}>
                        {validation ? `${validation.stdout}${validation.stderr}`.trim() || "config ok" : "Not validated yet."}
                      </pre>
                    </section>
                  </motion.div>
                )}

                {settingsTab === "about" && (
                  <motion.div key="settings-about" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.15 }} className="grid gap-5">
                    <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                      <SettingsSummaryCard icon={<Network size={18} />} label="Providers" value={String(providers.length)} />
                      <SettingsSummaryCard icon={<Route size={18} />} label="Routes" value={String(routes.length)} />
                      <SettingsSummaryCard icon={<History size={18} />} label="Launches" value={String(recentLaunches.length)} />
                      <SettingsSummaryCard icon={<Activity size={18} />} label="Status" value={status?.running ? "Running" : "Stopped"} tone={status?.running ? "success" : "muted"} />
                    </section>

                    <section className="rounded-lg border border-border bg-surface overflow-hidden">
                      <PanelHeader title="ferryllm Desktop" subtitle="Local control panel for provider routing and launcher workflows." />
                      <div className="grid gap-3 p-5">
                        <SettingsInfoRow icon={<Terminal size={15} />} label="Executable" value={executable || "ferryllm"} />
                        <SettingsInfoRow icon={<FileCog size={15} />} label="Config path" value={status?.config_path || "Default app config"} />
                        <SettingsInfoRow icon={<Gauge size={15} />} label="Listen" value={valueAsString(config.server?.listen) || "127.0.0.1:3000"} />
                        <SettingsInfoRow icon={<BarChart3 size={15} />} label="Metrics" value={valueAsBool(config.metrics?.enabled, true) ? "Enabled" : "Disabled"} tone={valueAsBool(config.metrics?.enabled, true) ? "success" : "muted"} />
                      </div>
                    </section>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}

          {/* ── Launcher View ── */}
          {activeView === "launcher" && (
            <motion.div
              key="launcher"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.2 }}
              className="grid w-full gap-4"
            >
              <LauncherView
                aiSessions={aiSessions}
                favoriteWorkspaces={favoriteWorkspaces}
                filteredWorkspaces={filteredWorkspaces}
                launchTool={launchTool}
                providers={providers}
                recentWorkspaces={recentWorkspaces}
                selectedWorkspace={selectedWorkspace}
                selectedWorkspaceSessions={selectedWorkspaceSessions}
                sessionScanBusy={sessionScanBusy}
                workspaceQuery={workspaceQuery}
                workspaceSearch={workspaceSearch}
                onAddWorkspace={addWorkspaceOnly}
                onCreateWorkspace={() => void createWorkspace()}
                onDeleteLaunch={deleteLaunch}
                onDeleteWorkspace={deleteWorkspace}
                onDeleteWorkspaceDirectory={(path) => void deleteWorkspaceDirectory(path)}
                onDeleteNativeSession={(session) => void deleteNativeSession(session)}
                onLaunchRecent={(item) => void quickLaunch(item)}
                onLaunchWorkspace={(workspace, target, tool) => void launchWorkspace(workspace, target, tool)}
                onRefreshSessions={() => void refreshAiSessions()}
                onRevealWorkspace={(path) => void revealWorkspace(path)}
                onResumeSession={(session) => void resumeAiSession(session)}
                onSelectWorkspace={setSelectedWorkspacePath}
                onSetWorkspaceSearch={setWorkspaceSearch}
                onSetWorkspaceGateway={updateWorkspaceGateway}
                onSetWorkspaceReasoning={updateWorkspaceReasoning}
                onTogglePin={toggleWorkspacePin}
              />
            </motion.div>
          )}
        </AnimatePresence>

      </main>

      <ToastContainer toasts={toasts} onDismiss={(id) => setToasts((p) => p.filter((t) => t.id !== id))} />
    </div>
  );
}

function AppSidebar({
  activeView,
  darkMode,
  provider,
  providerCount,
  logCount,
  status,
  onSetView,
  onToggleTheme,
}: {
  activeView: View;
  darkMode: boolean;
  provider: ProviderConfig | undefined;
  providerCount: number;
  logCount: number;
  status: ProcessStatus | null;
  onSetView: (view: View) => void;
  onToggleTheme: () => void;
}) {
  const navItems = [
    { view: "launcher" as View, icon: <FolderOpen size={15} />, label: "Launcher" },
    { view: "providers" as View, icon: <Network size={15} />, label: "Providers", count: providerCount },
    { view: "dashboard" as View, icon: <BarChart3 size={15} />, label: "Dashboard" },
    { view: "usage-logs" as View, icon: <History size={15} />, label: "Usage Logs", count: logCount },
    { view: "settings" as View, icon: <Settings size={15} />, label: "Settings" },
  ];

  return (
    <aside className="app-sidebar border-r border-border bg-surface">
      <div className="px-5 py-5">
        <button type="button" onClick={() => onSetView("launcher")} className="flex w-full items-center gap-3 text-left">
          <img src="/logo-light.png" alt="ferryllm" className="h-8 w-8 shrink-0 dark:hidden" />
          <img src="/logo-dark.png" alt="ferryllm" className="h-8 w-8 shrink-0 hidden dark:block" />
          <span className="min-w-0">
            <strong className="block text-base font-bold text-heading">ferryllm</strong>
            <span className="block text-xs font-semibold text-muted">LLM gateway console</span>
          </span>
        </button>
      </div>

      <nav className="grid gap-1 px-3">
        {navItems.map((item) => (
          <LauncherNavItem
            key={item.view}
            icon={item.icon}
            label={item.label}
            active={activeView === item.view || (activeView === "provider-detail" && item.view === "providers")}
            count={item.count}
            onClick={() => onSetView(item.view)}
          />
        ))}
      </nav>

      <div className="mt-auto grid gap-2 border-t border-border p-4">
        <SettingsInfoRow icon={<Activity size={15} />} label="Gateway" value={gatewayStatusLabel(status)} tone={gatewayStatusTone(status)} />
        <SettingsInfoRow icon={<Network size={15} />} label="Provider" value={provider?.name || "Not selected"} tone={provider ? "success" : "warning"} />
        <button
          type="button"
          onClick={onToggleTheme}
          className="mt-1 flex h-9 items-center gap-2 rounded-lg px-3 text-sm font-bold text-muted transition-colors hover:bg-muted-soft hover:text-heading"
        >
          {darkMode ? <Sun size={15} /> : <Moon size={15} />}
          <span>{darkMode ? "Light mode" : "Dark mode"}</span>
        </button>
      </div>
    </aside>
  );
}

function LauncherView({
  aiSessions,
  favoriteWorkspaces,
  filteredWorkspaces,
  launchTool,
  providers,
  recentWorkspaces,
  selectedWorkspace,
  selectedWorkspaceSessions,
  sessionScanBusy,
  workspaceQuery,
  workspaceSearch,
  onAddWorkspace,
  onCreateWorkspace,
  onDeleteLaunch,
  onDeleteNativeSession,
  onDeleteWorkspace,
  onDeleteWorkspaceDirectory,
  onLaunchRecent,
  onLaunchWorkspace,
  onRefreshSessions,
  onRevealWorkspace,
  onResumeSession,
  onSelectWorkspace,
  onSetWorkspaceSearch,
  onSetWorkspaceGateway,
  onSetWorkspaceReasoning,
  onTogglePin,
}: {
  aiSessions: AISession[];
  favoriteWorkspaces: WorkspaceEntry[];
  filteredWorkspaces: WorkspaceEntry[];
  launchTool: AITool;
  providers: ProviderConfig[];
  recentWorkspaces: WorkspaceEntry[];
  selectedWorkspace: WorkspaceEntry | undefined;
  selectedWorkspaceSessions: AISession[];
  sessionScanBusy: boolean;
  workspaceQuery: string;
  workspaceSearch: string;
  onAddWorkspace: () => void;
  onCreateWorkspace: () => void;
  onDeleteLaunch: (id: string) => void;
  onDeleteNativeSession: (session: AISession) => void;
  onDeleteWorkspace: (path: string) => void;
  onDeleteWorkspaceDirectory: (path: string) => void;
  onLaunchRecent: (item: RecentLaunch) => void;
  onLaunchWorkspace: (workspace: WorkspaceEntry, target: LaunchTarget, tool: AITool) => void;
  onRefreshSessions: () => void;
  onRevealWorkspace: (path: string) => void;
  onResumeSession: (session: AISession) => void;
  onSelectWorkspace: (path: string) => void;
  onSetWorkspaceSearch: (value: string) => void;
  onSetWorkspaceGateway: (path: string, providerName: string) => void;
  onSetWorkspaceReasoning: (path: string, reasoningEffort: string) => void;
  onTogglePin: (path: string) => void;
}) {
  const hasWorkspaces = filteredWorkspaces.length > 0;
  return (
    <section className="launcher-shell overflow-hidden rounded-lg border border-border bg-surface">
      <main className="min-w-0">
        <div className="launcher-welcome-header">
          <div className="relative min-w-[240px] flex-1">
            <Search size={16} className="provider-search-icon pointer-events-none absolute top-1/2 -translate-y-1/2 text-icon" />
            <input
              value={workspaceSearch}
              onChange={(event) => onSetWorkspaceSearch(event.currentTarget.value)}
              placeholder="Search projects"
              className="provider-search-input h-10"
            />
          </div>
          <div className="flex flex-wrap justify-end gap-2">
            <Button icon={<Plus size={14} />} onClick={onCreateWorkspace}>New project</Button>
            <Button icon={<FolderOpen size={14} />} onClick={onAddWorkspace}>Open</Button>
            <Button icon={<Download size={14} />} onClick={() => undefined} disabled>Clone repo</Button>
          </div>
        </div>

        <div className="launcher-projects">
          {!hasWorkspaces ? (
            <section className="launcher-empty-projects">
              <History size={42} className="text-muted" />
              <div>
                <h2 className="text-xl font-bold text-heading">No projects yet</h2>
                <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">Create or open a workspace, then launch Codex, Claude, VS Code, or resume indexed sessions from the project row.</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="primary" icon={<Plus size={14} />} onClick={onCreateWorkspace}>New project</Button>
                <Button icon={<FolderOpen size={14} />} onClick={onAddWorkspace}>Open project</Button>
              </div>
            </section>
          ) : (
            <div className="launcher-project-list">
              {favoriteWorkspaces.length ? (
                <LauncherWorkspaceSection
                  title="Favorites"
                  workspaces={favoriteWorkspaces}
                  selectedPath={selectedWorkspace?.path}
                  nativeSessions={aiSessions}
                  providers={providers}
                  onSelect={onSelectWorkspace}
                  onLaunch={(workspace) => onLaunchWorkspace(workspace, workspace.defaultLaunchType, workspace.defaultTool)}
                  onLaunchCodex={(workspace) => onLaunchWorkspace(workspace, "cli", "codex")}
                  onLaunchClaude={(workspace) => onLaunchWorkspace(workspace, "cli", "claude")}
                  onLaunchOpenCode={(workspace) => onLaunchWorkspace(workspace, "cli", "opencode")}
                  onLaunchVscode={(workspace) => onLaunchWorkspace(workspace, "vscode", launchTool)}
                  onReveal={onRevealWorkspace}
                  onSetGateway={onSetWorkspaceGateway}
                  onSetReasoning={onSetWorkspaceReasoning}
                  onTogglePin={onTogglePin}
                  onDelete={onDeleteWorkspace}
                  onDeleteDirectory={onDeleteWorkspaceDirectory}
                />
              ) : null}
              <LauncherWorkspaceSection
                title={workspaceQuery ? "Matching projects" : "Recent projects"}
                workspaces={recentWorkspaces}
                selectedPath={selectedWorkspace?.path}
                nativeSessions={aiSessions}
                providers={providers}
                onSelect={onSelectWorkspace}
                onLaunch={(workspace) => onLaunchWorkspace(workspace, workspace.defaultLaunchType, workspace.defaultTool)}
                onLaunchCodex={(workspace) => onLaunchWorkspace(workspace, "cli", "codex")}
                onLaunchClaude={(workspace) => onLaunchWorkspace(workspace, "cli", "claude")}
                onLaunchOpenCode={(workspace) => onLaunchWorkspace(workspace, "cli", "opencode")}
                onLaunchVscode={(workspace) => onLaunchWorkspace(workspace, "vscode", launchTool)}
                onReveal={onRevealWorkspace}
                onSetGateway={onSetWorkspaceGateway}
                onSetReasoning={onSetWorkspaceReasoning}
                onTogglePin={onTogglePin}
                onDelete={onDeleteWorkspace}
                onDeleteDirectory={onDeleteWorkspaceDirectory}
                emptyText={workspaceQuery ? "No project matches this search." : "No recent projects yet."}
              />
            </div>
          )}

          {selectedWorkspace ? (
            <LauncherSessionPanel
              launchHistory={selectedWorkspace.sessions}
              nativeSessions={selectedWorkspaceSessions}
              scanning={sessionScanBusy}
              onDeleteLaunch={onDeleteLaunch}
              onDeleteNativeSession={onDeleteNativeSession}
              onLaunchRecent={onLaunchRecent}
              onRefresh={onRefreshSessions}
              onResumeSession={onResumeSession}
            />
          ) : null}
        </div>
      </main>
    </section>
  );
}

function LauncherNavItem({ icon, label, count, active, onClick }: { icon: ReactNode; label: string; count?: number; active?: boolean; onClick?: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex h-9 items-center gap-2 rounded-lg px-3 text-sm font-bold transition-colors",
        active ? "bg-surface text-heading shadow-sm" : "text-muted hover:bg-muted-soft hover:text-heading"
      )}
    >
      {icon}
      <span className="min-w-0 flex-1 text-left">{label}</span>
      {count !== undefined ? <span className="rounded-full bg-muted-soft px-2 py-0.5 text-[11px] text-muted">{count}</span> : null}
    </button>
  );
}

function toolIcon(tool: AITool, size = 14) {
  if (tool === "claude") return <Zap size={size} />;
  if (tool === "opencode") return <Code2 size={size} />;
  return <Terminal size={size} />;
}

function toolToneClass(tool: AITool) {
  if (tool === "claude") return "bg-success-soft text-success";
  if (tool === "opencode") return "bg-muted-soft text-icon";
  return "bg-info-soft text-primary";
}

function LauncherSessionPanel({ nativeSessions, launchHistory, scanning, onDeleteLaunch, onDeleteNativeSession, onLaunchRecent, onRefresh, onResumeSession }: {
  nativeSessions: AISession[];
  launchHistory: RecentLaunch[];
  scanning: boolean;
  onDeleteLaunch: (id: string) => void;
  onDeleteNativeSession: (session: AISession) => void;
  onLaunchRecent: (item: RecentLaunch) => void;
  onRefresh: () => void;
  onResumeSession: (session: AISession) => void;
}) {
  return (
    <section className="launcher-session-panel rounded-lg border border-border bg-bg">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <h3 className="text-sm font-bold text-heading">Sessions</h3>
          <p className="mt-0.5 text-xs text-muted">{nativeSessions.length} native, {launchHistory.length} launches</p>
        </div>
        <IconAction title="Refresh sessions" icon={<RefreshCw size={14} className={scanning ? "animate-spin" : undefined} />} onClick={onRefresh} />
      </div>
      <div className="launcher-session-list grid gap-1 p-2">
        {nativeSessions.map((session) => (
          <div key={`${session.tool}-${session.id}`} className="flex items-start gap-2 rounded-lg px-2 py-2 hover:bg-surface">
            <span className={cn("grid h-8 w-8 shrink-0 place-items-center rounded-lg", toolToneClass(session.tool))}>
              {toolIcon(session.tool, 14)}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-bold text-heading">{session.tool} session</p>
              <p className="truncate text-xs text-muted">{formatRelativeTime(session.updated_at_ms)} · {session.id.slice(0, 8)} · {session.message_count} msgs</p>
              <p className="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-label">{session.preview || "No preview available."}</p>
            </div>
            <IconAction title="Resume native session" icon={<RotateCw size={13} />} onClick={() => onResumeSession(session)} />
            <IconAction title="Delete session file" danger icon={<Trash2 size={13} />} onClick={() => onDeleteNativeSession(session)} />
          </div>
        ))}
        {nativeSessions.length === 0 ? (
          <div className="px-3 py-5 text-center text-sm text-muted">{scanning ? "Scanning sessions..." : "No native sessions for this project."}</div>
        ) : null}
        {launchHistory.map((item) => (
          <div key={item.id} className="flex items-center gap-2 rounded-lg px-2 py-2 hover:bg-surface">
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-muted-soft text-icon">
              {item.launchType === "vscode" ? <Code2 size={14} /> : <Terminal size={14} />}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-bold text-heading">{item.tool} / {item.launchType === "vscode" ? "VS Code" : "CLI"}</p>
              <p className="truncate text-xs text-muted">{item.providerName} · {formatRelativeTime(item.lastUsed)}</p>
            </div>
            <IconAction title="Launch this target" icon={<Play size={13} />} onClick={() => onLaunchRecent(item)} />
            <IconAction title="Remove" danger icon={<Trash2 size={13} />} onClick={() => onDeleteLaunch(item.id)} />
          </div>
        ))}
      </div>
    </section>
  );
}

function LauncherWorkspaceSection({ title, workspaces, selectedPath, nativeSessions, providers, onSelect, onLaunch, onLaunchCodex, onLaunchClaude, onLaunchOpenCode, onLaunchVscode, onReveal, onSetGateway, onSetReasoning, onTogglePin, onDelete, onDeleteDirectory, emptyText }: {
  title: string;
  workspaces: WorkspaceEntry[];
  selectedPath?: string;
  nativeSessions: AISession[];
  providers: ProviderConfig[];
  onSelect: (path: string) => void;
  onLaunch: (workspace: WorkspaceEntry) => void;
  onLaunchCodex: (workspace: WorkspaceEntry) => void;
  onLaunchClaude: (workspace: WorkspaceEntry) => void;
  onLaunchOpenCode: (workspace: WorkspaceEntry) => void;
  onLaunchVscode: (workspace: WorkspaceEntry) => void;
  onReveal: (path: string) => void;
  onSetGateway: (path: string, providerName: string) => void;
  onSetReasoning: (path: string, reasoningEffort: string) => void;
  onTogglePin: (path: string) => void;
  onDelete: (path: string) => void;
  onDeleteDirectory: (path: string) => void;
  emptyText?: string;
}) {
  const [openMenuPath, setOpenMenuPath] = useState<string | null>(null);
  return (
    <section className="border-b border-border last:border-b-0">
      <div className="flex items-center justify-between px-5 py-3">
        <h2 className="text-xs font-bold uppercase tracking-normal text-muted">{title}</h2>
        <span className="rounded-full bg-muted-soft px-2 py-0.5 text-[11px] font-bold text-muted">{workspaces.length}</span>
      </div>
      {workspaces.length ? (
        <div className="grid gap-1 px-2 pb-3">
          {workspaces.map((workspace) => {
            const nativeCount = nativeSessions.filter((session) => session.cwd.toLowerCase() === workspace.path.toLowerCase()).length;
            const providerReady = providers.some((provider) => provider.name === workspace.defaultProviderName);
            return (
            <div
              key={workspace.path}
              role="button"
              tabIndex={0}
              onClick={() => onSelect(workspace.path)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(workspace.path);
                }
              }}
              className={cn(
                "group grid min-h-[68px] grid-cols-[40px_minmax(0,1fr)_auto] items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
                selectedPath === workspace.path
                  ? "border-primary/50 bg-info-soft"
                  : "border-transparent hover:border-border hover:bg-muted-soft"
              )}
            >
              <span className="grid h-10 w-10 place-items-center rounded-lg border border-border bg-surface text-icon">
                <FolderOpen size={17} />
              </span>
              <span className="min-w-0">
                <span className="flex min-w-0 items-center gap-2">
                  <strong className="truncate text-sm font-bold text-heading">{workspace.name}</strong>
                  {workspace.pinned ? <Star size={12} fill="currentColor" className="shrink-0 text-accent" /> : null}
                </span>
                <span className="mt-0.5 block truncate font-mono text-xs text-muted">{workspace.path}</span>
                <span className="mt-1 flex flex-wrap gap-1.5">
                  <span className="rounded-full bg-surface px-2 py-0.5 text-[11px] font-bold text-muted">{workspace.defaultTool}</span>
                  <span className="rounded-full bg-surface px-2 py-0.5 text-[11px] font-bold text-muted">{workspace.defaultLaunchType === "vscode" ? "VS Code" : "CLI"}</span>
                  {workspace.defaultProviderName ? (
                    <span className={cn("launcher-gateway-chip", providerReady ? "is-ready" : "is-missing")}>{workspace.defaultProviderName}</span>
                  ) : null}
                  {nativeCount ? <span className="rounded-full bg-success-soft px-2 py-0.5 text-[11px] font-bold text-success">{nativeCount} session{nativeCount === 1 ? "" : "s"}</span> : null}
                </span>
              </span>
              <span className="relative flex items-center gap-1">
                <span className="hidden text-[11px] font-semibold text-muted xl:inline">{formatRelativeTime(workspace.lastUsed)}</span>
                <select
                  className="launcher-gateway-select"
                  value={providerReady ? workspace.defaultProviderName : ""}
                  title="Project gateway"
                  onClick={(event) => event.stopPropagation()}
                  onChange={(event) => {
                    event.stopPropagation();
                    onSetGateway(workspace.path, event.currentTarget.value);
                  }}
                >
                  <option value="" disabled>Gateway</option>
                  {providers.map((provider) => (
                    <option key={provider.name} value={provider.name}>{provider.name}</option>
                  ))}
                </select>
                <select
                  className="launcher-gateway-select"
                  value={workspace.defaultReasoningEffort ?? ""}
                  title="Project reasoning"
                  onClick={(event) => event.stopPropagation()}
                  onChange={(event) => {
                    event.stopPropagation();
                    onSetReasoning(workspace.path, event.currentTarget.value);
                  }}
                >
                  <option value="">Reasoning</option>
                  {REASONING_OPTIONS.filter(Boolean).map((effort) => (
                    <option key={effort} value={effort}>{effort}</option>
                  ))}
                </select>
                <IconAction title="Open project" icon={<Play size={13} />} onClick={(event) => { event.stopPropagation(); onLaunch(workspace); }} />
                <IconAction
                  title="Project actions"
                  icon={<MoreVertical size={14} />}
                  onClick={(event) => {
                    event.stopPropagation();
                    setOpenMenuPath((value) => value === workspace.path ? null : workspace.path);
                  }}
                />
                {openMenuPath === workspace.path ? (
                  <div className="launcher-project-menu" onClick={(event) => event.stopPropagation()}>
                    <button type="button" onClick={() => { setOpenMenuPath(null); onLaunch(workspace); }}><Play size={13} /> Open project</button>
                    <button type="button" onClick={() => { setOpenMenuPath(null); onLaunchCodex(workspace); }}><Terminal size={13} /> Open in Codex</button>
                    <button type="button" onClick={() => { setOpenMenuPath(null); onLaunchClaude(workspace); }}><Zap size={13} /> Open in Claude</button>
                    <button type="button" onClick={() => { setOpenMenuPath(null); onLaunchOpenCode(workspace); }}><Code2 size={13} /> Open in OpenCode</button>
                    <button type="button" onClick={() => { setOpenMenuPath(null); onLaunchVscode(workspace); }}><Code2 size={13} /> Open in VS Code</button>
                    <button type="button" onClick={() => { setOpenMenuPath(null); onReveal(workspace.path); }}><FolderOpen size={13} /> Show in Explorer</button>
                    <button type="button" onClick={() => { setOpenMenuPath(null); onTogglePin(workspace.path); }}><Star size={13} fill={workspace.pinned ? "currentColor" : "none"} /> {workspace.pinned ? "Unpin" : "Pin"}</button>
                    <button type="button" className="danger" onClick={() => { setOpenMenuPath(null); onDelete(workspace.path); }}><Trash2 size={13} /> Remove from launcher</button>
                    <button type="button" className="danger" onClick={() => { setOpenMenuPath(null); onDeleteDirectory(workspace.path); }}><Trash2 size={13} /> Delete directory</button>
                  </div>
                ) : null}
              </span>
            </div>
            );
          })}
        </div>
      ) : (
        <div className="px-5 pb-5 text-sm text-muted">{emptyText ?? "No workspaces."}</div>
      )}
    </section>
  );
}

function DashboardView({ snapshot, status, latestSample, dashboard, logs, onRefresh, refreshing }: {
  snapshot: ServerSnapshot | null;
  status: ProcessStatus | null;
  latestSample: MetricSample | undefined;
  dashboard: {
    requests: number;
    ok: number;
    errors: number;
    upstreamErrors: number;
    cacheHit: number;
    successRate: number;
    avgLatency: number;
    requestRatePoints: number[];
    latencyPoints: number[];
  };
  logs: LogEntry[];
  onRefresh: () => void;
  refreshing: boolean;
}) {
  const promptTokens = latestSample?.values.ferryllm_upstream_prompt_tokens_total ?? 0;
  const cachedTokens = latestSample?.values.ferryllm_prompt_cached_tokens_total ?? 0;
  const cachePromptTokens = latestSample?.values.ferryllm_cache_prompt_tokens_total ?? 0;
  const uncachedTokens = Math.max(0, cachePromptTokens - cachedTokens);
  const modelRows = latestSample?.models ?? [];
  const modelCount = (() => {
    try {
      const parsed = snapshot?.models.ok ? JSON.parse(snapshot.models.body) : null;
      return Array.isArray(parsed?.data) ? parsed.data.length : 0;
    } catch {
      return 0;
    }
  })();
  const recentLogs = logs.slice(-10).reverse();
  const hasDashboardData = dashboard.requests > 0 || modelRows.length > 0 || promptTokens > 0 || recentLogs.length > 0;

  return (
    <motion.div
      key="dashboard"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.2 }}
      className="grid gap-5"
    >
      <section className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-heading">Dashboard</h1>
          <p className="mt-1 text-sm text-muted">
            {snapshot ? `Last refresh ${new Date(snapshot.fetched_at_ms).toLocaleTimeString()}` : "Waiting for the first sample"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ProbeBadge label={gatewayStatusLabel(status)} ok={!!status?.running} />
          <ProbeBadge label="Health" ok={snapshot ? !!snapshot.health.ok : null} detail={snapshot?.health.status ? String(snapshot.health.status) : undefined} />
          <ProbeBadge label="Ready" ok={snapshot ? !!snapshot.ready.ok : null} detail={snapshot?.ready.status ? String(snapshot.ready.status) : undefined} />
          <button
            type="button"
            onClick={onRefresh}
            disabled={refreshing}
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-border bg-surface px-3 text-sm font-semibold text-muted hover:text-heading disabled:opacity-50"
          >
            <RefreshCw size={14} className={refreshing ? "animate-spin" : undefined} /> Refresh
          </button>
        </div>
      </section>

      <section className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <StatCard icon={<Activity size={16} />} label="Requests" value={fmtInt(dashboard.requests)} />
        <StatCard icon={<CheckCircle2 size={16} />} label="Success" value={fmtPct(dashboard.successRate)} tone="success" />
        <StatCard icon={<AlertTriangle size={16} />} label="Errors" value={fmtInt(dashboard.errors)} tone={dashboard.errors ? "danger" : undefined} />
        <StatCard icon={<Network size={16} />} label="Upstream" value={fmtInt(dashboard.upstreamErrors)} tone={dashboard.upstreamErrors ? "danger" : undefined} />
        <StatCard icon={<Gauge size={16} />} label="Avg latency" value={fmtLatency(dashboard.avgLatency)} />
        <StatCard icon={<Database size={16} />} label="Cache hit" value={fmtPct(dashboard.cacheHit)} tone={dashboard.cacheHit > 0 ? "success" : undefined} />
      </section>

      <section className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
        <SparklineCard
          title="Requests over time"
          subtitle="Delta per dashboard sample"
          points={dashboard.requestRatePoints}
          value={fmtInt(dashboard.requestRatePoints.length ? dashboard.requestRatePoints[dashboard.requestRatePoints.length - 1] : 0)}
        />
        <SparklineCard
          title="Latency trend"
          subtitle="Average latency per sample"
          points={dashboard.latencyPoints}
          value={fmtLatency(dashboard.latencyPoints.length ? dashboard.latencyPoints[dashboard.latencyPoints.length - 1] : 0)}
        />
      </section>

      {!hasDashboardData ? (
        <DashboardEmptyState
          running={!!status?.running}
          metricsOk={snapshot ? !!snapshot.metrics.ok : null}
          modelsOk={snapshot ? !!snapshot.models.ok : null}
          logs={recentLogs.length}
        />
      ) : (
      <>
      <section className="grid gap-5 lg:grid-cols-[1fr_0.85fr]">
        <div className="rounded-lg border border-border bg-surface">
          <PanelHeader title="Provider / Model" subtitle={`${modelRows.length || modelCount} model entries tracked`} />
          <div className="overflow-auto">
            <table className="w-full min-w-[640px] border-collapse text-sm">
              <thead className="border-b border-border text-left text-xs uppercase text-muted">
                <tr>
                  <th className="px-4 py-3 font-semibold">Provider</th>
                  <th className="px-4 py-3 font-semibold">Model</th>
                  <th className="px-4 py-3 text-right font-semibold">Requests</th>
                  <th className="px-4 py-3 text-right font-semibold">Errors</th>
                  <th className="px-4 py-3 text-right font-semibold">Avg latency</th>
                </tr>
              </thead>
              <tbody>
                {modelRows.length ? modelRows.slice(0, 12).map((row) => (
                  <tr key={`${row.provider}:${row.model}`} className="border-b border-border/70 last:border-0">
                    <td className="px-4 py-3 font-semibold text-heading">{row.provider || "-"}</td>
                    <td className="px-4 py-3 text-muted">{row.model || "-"}</td>
                    <td className="px-4 py-3 text-right tabular-nums">{fmtInt(row.requests)}</td>
                    <td className={cn("px-4 py-3 text-right tabular-nums", row.errors ? "text-danger" : "text-muted")}>{fmtInt(row.errors)}</td>
                    <td className="px-4 py-3 text-right tabular-nums">{fmtLatency(row.requests > 0 ? row.latencyMicros / row.requests : 0)}</td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan={5} className="px-4 py-10 text-center text-sm text-muted">No labeled model metrics yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid gap-5">
          <div className="rounded-lg border border-border bg-surface">
            <PanelHeader title="Prompt Cache" subtitle={`${fmtInt(promptTokens)} upstream prompt tokens`} />
            <div className="space-y-4 p-5">
              <TokenBar cached={cachedTokens} uncached={uncachedTokens} />
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className="text-xs text-muted">Cached</p>
                  <strong className="text-heading">{fmtInt(cachedTokens)}</strong>
                </div>
                <div>
                  <p className="text-xs text-muted">Uncached</p>
                  <strong className="text-heading">{fmtInt(uncachedTokens)}</strong>
                </div>
                <div>
                  <p className="text-xs text-muted">Hit ratio</p>
                  <strong className="text-heading">{fmtPct(dashboard.cacheHit)}</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface">
            <PanelHeader title="Recent Logs" subtitle={gatewayStatusDetail(status)} />
            <div className="max-h-72 overflow-auto p-3">
              {recentLogs.length ? recentLogs.map((entry) => (
                <div key={`${entry.ts_ms}-${entry.stream}-${entry.line}`} className="grid grid-cols-[76px_64px_1fr] gap-2 border-b border-border/60 py-2 text-xs last:border-0">
                  <span className="text-muted">{new Date(entry.ts_ms).toLocaleTimeString()}</span>
                  <span className={cn(
                    "font-semibold",
                    entry.stream === "stderr" ? "text-danger" : entry.stream === "system" ? "text-primary" : "text-muted"
                  )}>{entry.stream}</span>
                  <span className="break-words font-mono text-fg">{entry.line}</span>
                </div>
              )) : (
                <div className="py-10 text-center text-sm text-muted">No logs yet.</div>
              )}
            </div>
          </div>
        </div>
      </section>
      </>
      )}
    </motion.div>
  );
}

function UsageLogsView({ logs, launches, onBack }: { logs: LogEntry[]; launches: RecentLaunch[]; onBack: () => void }) {
  const rows = [
    ...logs.map((entry) => ({
      id: `log-${entry.ts_ms}-${entry.stream}-${entry.line}`,
      ts: entry.ts_ms,
      type: entry.stream,
      provider: "-",
      model: "-",
      status: entry.stream === "stderr" ? "error" : "ok",
      detail: entry.line,
    })),
    ...launches.map((entry) => ({
      id: `launch-${entry.id}`,
      ts: entry.lastUsed,
      type: entry.launchType,
      provider: entry.providerName,
      model: entry.tool,
      status: "launched",
      detail: entry.directory,
    })),
  ].sort((a, b) => b.ts - a.ts).slice(0, 120);

  return (
    <motion.div
      key="usage-logs"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.2 }}
      className="grid w-full gap-4"
    >
      <section className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface px-5 py-4">
        <div className="flex items-start gap-3">
          <Button icon={<ChevronDown size={14} className="rotate-90" />} onClick={onBack}>Back</Button>
          <div>
            <p className="text-xs font-bold uppercase text-muted">Gateway activity</p>
            <h1 className="mt-1 text-xl font-bold text-heading">Usage Logs</h1>
            <p className="mt-1 text-sm text-muted">{rows.length} recent events from runtime logs and launcher history.</p>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard icon={<History size={16} />} label="Events" value={fmtInt(rows.length)} />
        <StatCard icon={<Terminal size={16} />} label="Runtime logs" value={fmtInt(logs.length)} />
        <StatCard icon={<FolderOpen size={16} />} label="Launches" value={fmtInt(launches.length)} />
        <StatCard icon={<AlertTriangle size={16} />} label="Errors" value={fmtInt(rows.filter((row) => row.status === "error").length)} tone={rows.some((row) => row.status === "error") ? "danger" : undefined} />
      </section>

      <section className="rounded-lg border border-border bg-surface">
        <PanelHeader title="Recent Activity" subtitle="Local gateway and launcher events" />
        <div className="overflow-auto">
          <table className="w-full min-w-[860px] border-collapse text-sm">
            <thead className="border-b border-border text-left text-xs uppercase text-muted">
              <tr>
                <th className="px-4 py-3 font-semibold">Time</th>
                <th className="px-4 py-3 font-semibold">Type</th>
                <th className="px-4 py-3 font-semibold">Provider</th>
                <th className="px-4 py-3 font-semibold">Model / Tool</th>
                <th className="px-4 py-3 font-semibold">Status</th>
                <th className="px-4 py-3 font-semibold">Detail</th>
              </tr>
            </thead>
            <tbody>
              {rows.length ? rows.map((row) => (
                <tr key={row.id} className="border-b border-border/70 last:border-0">
                  <td className="whitespace-nowrap px-4 py-3 text-muted">{new Date(row.ts).toLocaleString()}</td>
                  <td className="px-4 py-3 font-semibold text-heading">{row.type}</td>
                  <td className="px-4 py-3 text-muted">{row.provider}</td>
                  <td className="px-4 py-3 text-muted">{row.model}</td>
                  <td className={cn("px-4 py-3 font-semibold", row.status === "error" ? "text-danger" : "text-success")}>{row.status}</td>
                  <td className="max-w-[560px] truncate px-4 py-3 font-mono text-xs text-muted" title={row.detail}>{row.detail}</td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={6} className="px-4 py-16 text-center text-sm text-muted">No usage events yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </motion.div>
  );
}

function DashboardEmptyState({ running, metricsOk, modelsOk, logs }: { running: boolean; metricsOk: boolean | null; modelsOk: boolean | null; logs: number }) {
  const rows = [
    { icon: <Terminal size={15} />, label: "Runtime", value: running ? "Running" : "Stopped", ok: running },
    { icon: <BarChart3 size={15} />, label: "Metrics", value: metricsOk == null ? "Waiting" : metricsOk ? "Reachable" : "Unavailable", ok: metricsOk },
    { icon: <Database size={15} />, label: "Models", value: modelsOk == null ? "Waiting" : modelsOk ? "Reachable" : "Unavailable", ok: modelsOk },
    { icon: <History size={15} />, label: "Logs", value: logs ? `${logs} recent` : "No logs yet", ok: logs > 0 ? true : null },
  ];
  return (
    <section className="rounded-lg border border-border bg-surface p-6">
      <div className="mx-auto max-w-3xl text-center">
        <Activity size={28} className="mx-auto text-icon" />
        <h2 className="mt-3 text-base font-bold text-heading">{running ? "Waiting for traffic" : "Runtime is stopped"}</h2>
        <p className="mt-1 text-sm text-muted">
          {running
            ? "Charts and model rows will appear after the first request sample."
            : "Start ferryllm from Providers or Launcher to collect metrics, models, cache data, and logs."}
        </p>
      </div>
      <div className="mt-5 grid gap-2 md:grid-cols-4">
        {rows.map((row) => (
          <div key={row.label} className="flex items-center gap-3 rounded-lg border border-border bg-bg px-3 py-2.5">
            <span className={cn(
              "inline-grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-muted-soft text-icon",
              row.ok === true && "bg-success-soft text-success",
              row.ok === false && "bg-danger-soft text-danger"
            )}>
              {row.icon}
            </span>
            <div className="min-w-0">
              <p className="text-xs font-semibold text-muted">{row.label}</p>
              <p className="truncate text-sm font-bold text-heading">{row.value}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function ProbeBadge({ label, ok, detail }: { label: string; ok: boolean | null; detail?: string }) {
  return (
    <span className={cn(
      "inline-flex h-8 items-center gap-2 rounded-lg border px-3 text-xs font-bold",
      ok == null && "border-border bg-muted-soft text-muted",
      ok === true && "border-success/30 bg-success-soft text-success",
      ok === false && "border-danger/30 bg-danger-soft text-danger"
    )}>
      <span className={cn("h-2 w-2 rounded-full", ok == null && "bg-muted", ok === true && "bg-success", ok === false && "bg-danger")} />
      {label}{detail ? ` ${detail}` : ""}
    </span>
  );
}

function StatCard({ icon, label, value, tone }: { icon: ReactNode; label: string; value: string; tone?: "success" | "danger" }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className={cn(
        "mb-3 inline-grid h-8 w-8 place-items-center rounded-lg bg-muted-soft text-icon",
        tone === "success" && "bg-success-soft text-success",
        tone === "danger" && "bg-danger-soft text-danger"
      )}>
        {icon}
      </div>
      <p className="text-xs font-semibold uppercase text-muted">{label}</p>
      <strong className="mt-1 block truncate text-xl font-bold text-heading tabular-nums">{value}</strong>
    </div>
  );
}

function SparklineCard({ title, subtitle, points, value }: { title: string; subtitle: string; points: number[]; value: string }) {
  const path = sparklinePath(points);
  const area = sparklineAreaPath(points);
  return (
    <div className="rounded-lg border border-border bg-surface p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-bold text-heading">{title}</h2>
          <p className="mt-1 text-xs text-muted">{subtitle}</p>
        </div>
        <strong className="text-lg font-bold text-heading tabular-nums">{value}</strong>
      </div>
      <svg viewBox="0 0 320 96" className="h-28 w-full overflow-visible">
        <defs>
          <linearGradient id={`spark-${title.replace(/\s+/g, "-")}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--c-primary)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--c-primary)" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {[18, 42, 66, 90].map((y) => (
          <path key={y} d={`M 0 ${y} L 320 ${y}`} stroke="currentColor" className="text-border" strokeWidth="1" opacity="0.55" />
        ))}
        {area ? <path d={area} fill={`url(#spark-${title.replace(/\s+/g, "-")})`} /> : (
          <path d="M 0 76 C 42 58 78 70 116 48 S 190 40 232 56 S 286 74 320 50 L 320 96 L 0 96 Z" fill="var(--c-muted-soft)" opacity="0.75" />
        )}
        {path ? <path d={path} fill="none" stroke="var(--c-primary)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" /> : (
          <path d="M 0 76 C 42 58 78 70 116 48 S 190 40 232 56 S 286 74 320 50" fill="none" stroke="var(--c-border-strong)" strokeWidth="2.5" strokeLinecap="round" />
        )}
      </svg>
    </div>
  );
}

function TokenBar({ cached, uncached }: { cached: number; uncached: number }) {
  const total = cached + uncached;
  const cachedPct = total > 0 ? (cached / total) * 100 : 0;
  return (
    <div>
      <div className="flex h-3 overflow-hidden rounded-full bg-muted-soft">
        {total > 0 ? (
          <>
            <div className="bg-success" style={{ width: `${cachedPct}%` }} />
            <div className="bg-primary" style={{ width: `${100 - cachedPct}%` }} />
          </>
        ) : null}
      </div>
      <div className="mt-2 flex items-center justify-between text-xs text-muted">
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-success" /> cached</span>
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-primary" /> uncached</span>
      </div>
    </div>
  );
}

function ProviderMiniStat({ icon, label, value, detail, ok }: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  ok?: boolean;
}) {
  return (
    <div className="min-w-0 rounded-lg border border-border bg-bg px-3 py-2">
      <div className="mb-1 flex items-center gap-1.5 text-[11px] font-bold uppercase text-muted">
        <span className={cn("text-icon", ok === true && "text-success", ok === false && "text-danger")}>{icon}</span>
        <span className="truncate">{label}</span>
      </div>
      <p className="truncate text-sm font-bold text-heading">{value}</p>
      <p className="mt-0.5 truncate text-[11px] text-muted" title={detail}>{detail}</p>
    </div>
  );
}

/* ── Shared UI atoms ───────────────────────────────────────────────── */

function IconAction({ title, icon, onClick, danger }: {
  title: string;
  icon: ReactNode;
  onClick: (event: MouseEvent<HTMLButtonElement>) => void;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={onClick}
      className={cn(
        "inline-grid h-8 w-8 place-items-center rounded-lg text-icon transition-colors hover:bg-muted-soft hover:text-icon-hover",
        danger && "hover:bg-danger-soft hover:text-danger"
      )}
    >
      {icon}
    </button>
  );
}

function Button({ children, variant, icon, onClick, disabled, title }: {
  children?: ReactNode;
  variant?: "primary" | "success" | "danger";
  icon?: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cn(
        "inline-flex h-9 items-center gap-2 rounded-xl px-4 text-sm font-semibold transition-all duration-150",
        variant === "primary" && "bg-primary text-white hover:bg-primary-hover shadow-sm",
        variant === "success" && "bg-success-soft text-success hover:brightness-95",
        variant === "danger" && "bg-danger-soft text-danger hover:brightness-95",
        !variant && "bg-muted-soft text-fg hover:bg-border",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      {icon} {children}
    </button>
  );
}

function PanelHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="border-b border-border px-6 py-5">
      <h2 className="text-base font-bold text-heading">{title}</h2>
      <p className="mt-1 text-sm text-muted">{subtitle}</p>
    </div>
  );
}

function EmptyState({ title, action, onAction, compact }: { title: string; action: string; onAction: () => void; compact?: boolean }) {
  return (
    <div className={cn(
      "flex flex-col items-center justify-center rounded-2xl border border-dashed border-border-strong bg-muted-soft text-center",
      compact ? "py-8" : "py-16"
    )}>
      <CirclePlus size={compact ? 24 : 36} className="mb-3 text-muted" />
      <strong className="text-base font-bold text-heading">{title}</strong>
      <button
        type="button"
        onClick={onAction}
        className="mt-4 inline-flex h-9 items-center gap-2 rounded-xl bg-primary px-5 text-sm font-semibold text-white hover:bg-primary-hover"
      >
        <Plus size={14} /> {action}
      </button>
    </div>
  );
}

function SettingsTabButton({ active, icon, label, onClick }: {
  active: boolean;
  icon: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex h-11 items-center justify-center gap-2 rounded-md text-sm font-bold transition-colors",
        active ? "bg-primary text-white shadow-sm" : "text-muted hover:bg-muted-soft hover:text-heading"
      )}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}

function SettingsInfoRow({ icon, label, value, tone = "default" }: {
  icon: ReactNode;
  label: string;
  value: string;
  tone?: "default" | "success" | "warning" | "muted";
}) {
  return (
    <div className="flex min-h-12 items-center gap-3 rounded-lg border border-border bg-bg px-3 py-2">
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-muted-soft text-icon">{icon}</span>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-semibold uppercase text-muted">{label}</p>
        <p className={cn(
          "mt-0.5 truncate text-sm font-semibold",
          tone === "success" && "text-success",
          tone === "warning" && "text-accent",
          tone === "muted" && "text-muted",
          tone === "default" && "text-heading"
        )}>
          {value}
        </p>
      </div>
    </div>
  );
}

function SettingsFeatureRow({ icon, title, status, detail }: {
  icon: ReactNode;
  title: string;
  status: "Partial" | "Planned";
  detail?: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-bg px-3 py-3">
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-muted-soft text-icon">{icon}</span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-bold text-heading">{title}</p>
        {detail ? <p className="mt-0.5 truncate text-xs text-muted">{detail}</p> : null}
      </div>
      <span className={cn(
        "shrink-0 rounded-full px-2 py-0.5 text-[11px] font-bold",
        status === "Partial" ? "bg-info-soft text-primary" : "bg-muted-soft text-muted"
      )}>
        {status}
      </span>
    </div>
  );
}

function SettingsSummaryCard({ icon, label, value, tone = "default" }: {
  icon: ReactNode;
  label: string;
  value: string;
  tone?: "default" | "success" | "muted";
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="flex items-center justify-between">
        <span className="grid h-9 w-9 place-items-center rounded-lg bg-muted-soft text-icon">{icon}</span>
        <span className={cn(
          "text-2xl font-bold",
          tone === "success" && "text-success",
          tone === "muted" && "text-muted",
          tone === "default" && "text-heading"
        )}>
          {value}
        </span>
      </div>
      <p className="mt-3 text-xs font-semibold uppercase text-muted">{label}</p>
    </div>
  );
}

function TextField({ label, value, onChange, secret }: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  secret?: boolean;
}) {
  const [revealed, setRevealed] = useState(false);
  const input = (
    <input
      type={secret && !revealed ? "password" : "text"}
      value={value}
      autoComplete={secret ? "off" : undefined}
      spellCheck={secret ? false : undefined}
      className={secret ? "pr-10" : undefined}
      onChange={(e) => onChange(e.currentTarget.value)}
    />
  );

  return (
    <label>
      <span>{label}</span>
      {secret ? (
        <div className="relative">
          {input}
          <button
            type="button"
            title={revealed ? "Hide API key" : "Show API key"}
            aria-label={revealed ? "Hide API key" : "Show API key"}
            className="absolute right-2 top-1/2 grid h-7 w-7 -translate-y-1/2 place-items-center rounded-md text-icon hover:bg-muted-soft hover:text-icon-hover"
            onClick={() => setRevealed((v) => !v)}
          >
            {revealed ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
        </div>
      ) : input}
    </label>
  );
}

function NumberField({ label, value, optional, onChange }: { label: string; value: string; optional?: boolean; onChange: (v: number | undefined) => void }) {
  return (
    <label>
      <span>{label}</span>
      <input
        type="number"
        value={value}
        placeholder={optional ? "off" : undefined}
        onChange={(e) => {
          const raw = e.currentTarget.value;
          onChange(raw === "" ? undefined : Number(raw));
        }}
      />
    </label>
  );
}

function SelectField({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <label>
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.currentTarget.value)}>
        {options.map((o) => <option key={o || "none"} value={o}>{o || "unset"}</option>)}
      </select>
    </label>
  );
}

function BoolField({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex h-10 items-center justify-between gap-3 rounded-lg border border-border bg-bg px-3 text-left transition-colors hover:border-border-strong hover:bg-surface-hover"
    >
      <span className="text-sm font-semibold text-label">{label}</span>
      <span
        className={cn(
          "relative h-5 w-9 shrink-0 rounded-full border transition-colors",
          checked ? "border-primary bg-primary" : "border-border-strong bg-muted-soft"
        )}
      >
        <span
          className={cn(
            "absolute left-0.5 top-1/2 h-4 w-4 -translate-y-1/2 rounded-full bg-white shadow-sm transition-transform",
            checked ? "translate-x-4" : "translate-x-0"
          )}
        />
      </span>
    </button>
  );
}

export default App;

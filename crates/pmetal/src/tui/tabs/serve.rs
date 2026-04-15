//! Serve tab — HTTP inference server configuration and control.
//!
//! Mirrors the `pmetal serve` subcommand. Field navigation, inline edit,
//! and log tailing are delegated to `FormTabState` + `JobLog` so this
//! file only owns the tab-specific shape: the field list, the
//! `ServeStatus` state machine, and the status-panel rendering.

use std::time::{Duration, Instant};

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{
    FieldKind, FormAction, FormField, FormTabState, JobLog, StatusTone, status_line,
};

/// Runtime status of the `pmetal serve` process.
#[derive(Debug, Clone, Default)]
pub enum ServeStatus {
    #[default]
    Idle,
    Starting {
        bind_url: String,
        started_at: Instant,
    },
    Running {
        bind_url: String,
        started_at: Instant,
    },
    Stopped {
        reason: String,
    },
    Failed(String),
}

/// Serve tab state.
pub struct ServeTab {
    pub form: FormTabState,
    pub status: ServeStatus,
    pub log: JobLog,
}

impl ServeTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: ServeStatus::Idle,
            log: JobLog::with_default_cap(),
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            FormField::new("Model", "(not selected)", FieldKind::ModelPicker, "Model"),
            FormField::new("LoRA Adapter", "", FieldKind::Text, "Model"),
            FormField::new("Experts Dir", "", FieldKind::Text, "Model"),
            FormField::new("Host", "0.0.0.0", FieldKind::Text, "Network"),
            FormField::new(
                "Port",
                "8080",
                FieldKind::Integer { min: 1, max: 65_535 },
                "Network",
            ),
            FormField::new(
                "Max Seq Len",
                "4096",
                FieldKind::Integer {
                    min: 256,
                    max: 131_072,
                },
                "Runtime",
            ),
            FormField::new("FP8 Weights", "Disabled", FieldKind::Toggle, "Runtime"),
            FormField::new(
                "KV Cache",
                "auto",
                FieldKind::Enum {
                    options: vec![
                        "auto".into(),
                        "fp16".into(),
                        "q8".into(),
                        "q4".into(),
                        "tq8".into(),
                        "tq4".into(),
                        "tq2_5".into(),
                        "tq3_5".into(),
                    ],
                },
                "KV Cache",
            ),
            FormField::new(
                "KV Group Size",
                "64",
                FieldKind::Integer { min: 8, max: 256 },
                "KV Cache",
            ),
        ]
    }

    // ── Form delegation (all sections always visible) ──────────────────

    pub fn is_editing(&self) -> bool {
        self.form.is_editing()
    }
    pub fn handle_edit_key(&mut self, k: crossterm::event::KeyEvent) {
        self.form.handle_edit_key(k);
    }
    pub fn confirm_edit(&mut self) {
        self.form.confirm_edit();
    }
    pub fn cancel_edit(&mut self) {
        self.form.cancel_edit();
    }
    pub fn next_param(&mut self) {
        self.form.next_param(|_| true);
    }
    pub fn prev_param(&mut self) {
        self.form.prev_param(|_| true);
    }
    pub fn handle_enter(&mut self) -> Option<FormAction> {
        self.form.handle_enter()
    }

    // ── State transitions ──────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(
            self.status,
            ServeStatus::Starting { .. } | ServeStatus::Running { .. }
        )
    }

    pub fn mark_starting(&mut self, bind_url: String) {
        self.log.clear();
        self.status = ServeStatus::Starting {
            bind_url,
            started_at: Instant::now(),
        };
    }

    /// Append a stdout/stderr line. Triggers the Starting → Running
    /// transition when we see the server's "listening"/"ready" banner.
    pub fn append_log(&mut self, line: &str) {
        if let ServeStatus::Starting { bind_url, .. } = &self.status {
            let lower = line.to_lowercase();
            if lower.contains("listening") || lower.contains("ready") {
                let bind_url = bind_url.clone();
                self.status = ServeStatus::Running {
                    bind_url,
                    started_at: Instant::now(),
                };
            }
        }
        self.log.push(line);
    }

    pub fn set_failed(&mut self, message: &str) {
        self.status = ServeStatus::Failed(message.to_string());
    }

    pub fn set_stopped(&mut self, reason: &str) {
        self.status = ServeStatus::Stopped {
            reason: reason.to_string(),
        };
    }

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Model", model_id);
    }

    // ── Config summary / CLI args ──────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        if self.form.value("Model") == "(not selected)" {
            return Err("Model is required.".into());
        }
        if self.form.value("Host").is_empty() {
            return Err("Host is required (e.g. 0.0.0.0 or 127.0.0.1).".into());
        }
        Ok(())
    }

    pub fn bind_url(&self) -> String {
        let host = self.form.value("Host");
        let port = self.form.value("Port");
        let display_host = if host == "0.0.0.0" {
            "localhost".to_string()
        } else {
            host
        };
        format!("http://{display_host}:{port}")
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Model:       {}", self.form.value("Model")),
            format!("Bind:        {}", self.bind_url()),
            format!("Max Seq Len: {}", self.form.value("Max Seq Len")),
            format!("KV Cache:    {}", self.form.value("KV Cache")),
            format!("FP8:         {}", self.form.value("FP8 Weights")),
            String::new(),
            "Start server?".into(),
        ]
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec!["serve".to_string()];

        args.extend(["--model".into(), self.form.value("Model")]);
        let lora = self.form.value("LoRA Adapter");
        if !lora.is_empty() {
            args.extend(["--lora".into(), lora]);
        }
        let experts = self.form.value("Experts Dir");
        if !experts.is_empty() {
            args.extend(["--experts-dir".into(), experts]);
        }
        args.extend(["--host".into(), self.form.value("Host")]);
        args.extend(["--port".into(), self.form.value("Port")]);
        args.extend(["--max-seq-len".into(), self.form.value("Max Seq Len")]);
        args.extend(["--kv-group-size".into(), self.form.value("KV Group Size")]);

        if self.form.value("FP8 Weights") == "Enabled" {
            args.push("--fp8".into());
        }

        // KV cache preset → CLI flag mapping. `auto` emits nothing.
        match self.form.value("KV Cache").as_str() {
            "auto" => {}
            "fp16" => args.push("--no-kv-quant".into()),
            "q8" => args.extend(["--kv-quant".into(), "8".into()]),
            "q4" => args.extend(["--kv-quant".into(), "4".into()]),
            "tq8" => {
                args.push("--kv-turboquant".into());
                args.extend(["--kv-quant".into(), "8".into()]);
            }
            "tq4" => {
                args.push("--kv-turboquant".into());
                args.extend(["--kv-quant".into(), "4".into()]);
            }
            "tq2_5" => args.extend(["--kv-turboquant-preset".into(), "q2_5".into()]),
            "tq3_5" => args.extend(["--kv-turboquant-preset".into(), "q3_5".into()]),
            _ => {}
        }

        args
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl ServeTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "Serve Configuration", |_| true);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(8), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Server Log");
    }

    fn render_status(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Status ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line> = Vec::new();
        match &self.status {
            ServeStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start  [x] Stop",
                    THEME.text_muted,
                )));
            }
            ServeStatus::Starting { bind_url, .. } => {
                lines.push(status_line(StatusTone::Running, "Starting", Some(bind_url)));
            }
            ServeStatus::Running {
                bind_url,
                started_at,
            } => {
                lines.push(status_line(StatusTone::Running, "Serving", Some(bind_url)));
                lines.push(Line::from(vec![
                    Span::styled("  Uptime  ", THEME.kv_key),
                    Span::styled(format_duration(started_at.elapsed()), THEME.kv_value),
                ]));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [x] Stop",
                    THEME.text_muted,
                )));
            }
            ServeStatus::Stopped { reason } => {
                lines.push(status_line(StatusTone::Idle, "Stopped", Some(reason)));
            }
            ServeStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines).wrap(Wrap { trim: false }).render(inner, buf);
    }
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

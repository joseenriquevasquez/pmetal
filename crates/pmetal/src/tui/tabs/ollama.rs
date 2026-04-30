//! Ollama integration tab — wrapper around `pmetal ollama` subcommands.
//!
//! There is no `OllamaSpec`; Ollama is a thin CLI wrapper
//! (`pmetal ollama install/run/list/...`). This tab provides a minimal
//! free-form interface until Ollama gets proper spec coverage.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{JobLog, StatusTone, status_line};

/// Runtime status of the `pmetal ollama` process.
#[derive(Debug, Clone, Default)]
pub enum OllamaStatus {
    #[default]
    Idle,
    Running,
    Completed,
    Failed(String),
}

/// A minimal editable field for the Ollama tab.
#[derive(Debug, Clone)]
pub struct OllamaField {
    pub label: &'static str,
    pub value: String,
    pub editing: bool,
    pub edit_buf: String,
}

impl OllamaField {
    fn new(label: &'static str, default: &str) -> Self {
        Self {
            label,
            value: default.to_string(),
            editing: false,
            edit_buf: String::new(),
        }
    }
}

/// Ollama tab state.
pub struct OllamaTab {
    pub status: OllamaStatus,
    pub log: JobLog,
    /// Simple three-field form: Action / Model / Extra Args.
    fields: [OllamaField; 3],
    selected: usize,
}

impl OllamaTab {
    pub fn new() -> Self {
        Self {
            status: OllamaStatus::Idle,
            log: JobLog::with_default_cap(),
            fields: [
                OllamaField::new("Action", "run"),
                OllamaField::new("Model", ""),
                OllamaField::new("Extra Args", ""),
            ],
            selected: 0,
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    pub fn is_editing(&self) -> bool {
        self.fields.iter().any(|f| f.editing)
    }

    pub fn next_param(&mut self) {
        if !self.is_editing() {
            self.selected = (self.selected + 1) % self.fields.len();
        }
    }

    pub fn prev_param(&mut self) {
        if !self.is_editing() {
            self.selected = (self.selected + self.fields.len() - 1) % self.fields.len();
        }
    }

    pub fn handle_enter(&mut self) {
        let f = &mut self.fields[self.selected];
        if f.editing {
            f.value = f.edit_buf.trim().to_string();
            f.editing = false;
        } else {
            f.edit_buf = f.value.clone();
            f.editing = true;
        }
    }

    pub fn cancel_edit(&mut self) {
        for f in &mut self.fields {
            f.editing = false;
        }
    }

    pub fn handle_edit_key(&mut self, k: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        let f = &mut self.fields[self.selected];
        if !f.editing {
            return;
        }
        match k.code {
            KeyCode::Char(c) => f.edit_buf.push(c),
            KeyCode::Backspace => {
                f.edit_buf.pop();
            }
            _ => {}
        }
    }

    pub fn confirm_edit(&mut self) {
        let f = &mut self.fields[self.selected];
        f.value = f.edit_buf.trim().to_string();
        f.editing = false;
    }

    // ── State transitions ─────────────────────────────────────────────

    pub fn is_running(&self) -> bool {
        matches!(self.status, OllamaStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = OllamaStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = OllamaStatus::Completed;
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = OllamaStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        self.log.push(line);
    }

    // ── Config ───────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        let action = self.fields[0].value.trim().to_string();
        if action.is_empty() {
            return Err("Action is required (e.g. install, run, list).".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Action:  {}", self.fields[0].value),
            format!("Model:   {}", self.fields[1].value),
            format!("Args:    {}", self.fields[2].value),
            String::new(),
            "Run ollama subcommand?".into(),
        ]
    }

    /// Build CLI args: `["ollama", action, model?, extra_args?...]`
    pub fn build_cli_args(&self) -> Vec<String> {
        let action = self.fields[0].value.trim().to_string();
        let model = self.fields[1].value.trim().to_string();
        let extra = self.fields[2].value.trim().to_string();

        let mut args = vec!["ollama".to_string(), action];
        if !model.is_empty() {
            args.push(model);
        }
        if !extra.is_empty() {
            // Split extra args on whitespace
            args.extend(extra.split_whitespace().map(str::to_string));
        }
        args
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl OllamaTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.render_form(config_area, buf);

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Ollama Log");
    }

    fn render_form(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Ollama ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(""));

        for (i, field) in self.fields.iter().enumerate() {
            let is_sel = i == self.selected;
            let label_style = if is_sel {
                THEME.kv_key
            } else {
                THEME.text_muted
            };
            let val_style = if is_sel {
                THEME.table_selected
            } else {
                THEME.kv_value
            };

            let display_val = if field.editing {
                format!("{}_", field.edit_buf)
            } else {
                field.value.clone()
            };

            lines.push(Line::from(vec![
                Span::styled(format!("  {:>12}:  ", field.label), label_style),
                Span::styled(display_val, val_style),
            ]));
            lines.push(Line::from(""));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Actions: install | run | list | pull | rm | ps | serve",
            THEME.text_dim,
        )));
        lines.push(Line::from(Span::styled(
            "  Enter to edit  S to run  x to cancel",
            THEME.text_muted,
        )));

        Paragraph::new(lines).render(inner, buf);
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
            OllamaStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Run  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            OllamaStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Running", None));
            }
            OllamaStatus::Completed => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
            }
            OllamaStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

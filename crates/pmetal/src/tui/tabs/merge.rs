//! Merge tab — blend two (or more) models via SLERP / TIES / DARE /
//! linear / task-arithmetic. Mirrors `pmetal merge`.
//!
//! Form has three model pickers (A, B, optional Base for task-vector
//! methods) plus method-specific knobs (interpolation `t`, weights,
//! density, dtype). Status panel tails the subprocess output.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::theme::THEME;
use crate::tui::widgets::{
    FieldKind, FormAction, FormField, FormTabState, JobLog, StatusTone, status_line,
};

/// Runtime status of the `pmetal merge` subprocess.
#[derive(Debug, Clone, Default)]
pub enum MergeStatus {
    #[default]
    Idle,
    Running,
    Completed {
        output: String,
    },
    Failed(String),
}

/// Merge tab state.
pub struct MergeTab {
    pub form: FormTabState,
    pub status: MergeStatus,
    pub log: JobLog,
}

impl MergeTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: MergeStatus::Idle,
            log: JobLog::with_default_cap(),
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            FormField::new(
                "Model A",
                "(not selected)",
                FieldKind::ModelPicker,
                "Sources",
            ),
            FormField::new(
                "Model B",
                "(not selected)",
                FieldKind::ModelPicker,
                "Sources",
            ),
            FormField::new("Base Model", "", FieldKind::ModelPicker, "Sources"),
            FormField::new(
                "Method",
                "slerp",
                FieldKind::Enum {
                    options: vec![
                        "slerp".into(),
                        "linear".into(),
                        "ties".into(),
                        "dare_ties".into(),
                        "dare_linear".into(),
                        "task_arithmetic".into(),
                        "della".into(),
                        "breadcrumbs".into(),
                        "model_stock".into(),
                        "nearswap".into(),
                        "passthrough".into(),
                    ],
                },
                "Method",
            ),
            FormField::new(
                "Interpolation t",
                "0.5",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "SLERP",
            ),
            FormField::new(
                "Weight A",
                "0.5",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "Linear/TIES",
            ),
            FormField::new(
                "Weight B",
                "0.5",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "Linear/TIES",
            ),
            FormField::new(
                "Density",
                "0.5",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "TIES/DARE",
            ),
            FormField::new(
                "Output Dtype",
                "bfloat16",
                FieldKind::Enum {
                    options: vec!["bfloat16".into(), "float16".into(), "float32".into()],
                },
                "Output",
            ),
            FormField::new("Output Dir", "./output/merged", FieldKind::Text, "Output"),
        ]
    }

    /// Hide method-specific sections that don't apply to the current
    /// choice. Keeps the form focused on the relevant knobs.
    fn field_visible(method: &str, f: &FormField) -> bool {
        let uses_t = matches!(method, "slerp");
        let uses_weights = matches!(method, "linear" | "ties" | "dare_linear" | "dare_ties");
        let uses_density = matches!(
            method,
            "ties" | "dare_ties" | "dare_linear" | "della" | "breadcrumbs"
        );
        let uses_base = matches!(
            method,
            "ties" | "dare_ties" | "dare_linear" | "task_arithmetic" | "della"
        );
        match f.section.as_str() {
            "SLERP" => uses_t,
            "Linear/TIES" => uses_weights,
            "TIES/DARE" => uses_density,
            _ => {
                // Hide the optional Base Model picker when the method
                // doesn't consume it.
                if f.label == "Base Model" {
                    uses_base
                } else {
                    true
                }
            }
        }
    }

    fn method_snapshot(&self) -> String {
        self.form.value("Method")
    }

    // ── Form delegation ─────────────────────────────────────────────────

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
    pub fn handle_enter(&mut self) -> Option<FormAction> {
        self.form.handle_enter()
    }

    pub fn next_param(&mut self) {
        let method = self.method_snapshot();
        self.form
            .next_param(move |f| Self::field_visible(&method, f));
    }

    pub fn prev_param(&mut self) {
        let method = self.method_snapshot();
        self.form
            .prev_param(move |f| Self::field_visible(&method, f));
    }

    // ── Setters ─────────────────────────────────────────────────────────

    pub fn set_model(&mut self, label: &str, model_id: &str) {
        self.form.set_value(label, model_id);
        // Auto-derive output dir from Model A's short name when it's
        // the first slot populated.
        if label == "Model A" {
            let short = super::model_short_name(model_id);
            if self.form.value("Output Dir") == "./output/merged" {
                self.form
                    .set_value("Output Dir", format!("./output/{short}--merged"));
            }
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, MergeStatus::Running)
    }

    pub fn mark_running(&mut self) {
        self.log.clear();
        self.status = MergeStatus::Running;
    }

    pub fn mark_completed(&mut self) {
        self.status = MergeStatus::Completed {
            output: self.form.value("Output Dir"),
        };
    }

    pub fn mark_failed(&mut self, msg: &str) {
        self.status = MergeStatus::Failed(msg.to_string());
    }

    pub fn append_log(&mut self, line: &str) {
        self.log.push(line);
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        if self.form.value("Model A") == "(not selected)" {
            return Err("Model A is required.".into());
        }
        if self.form.value("Model B") == "(not selected)" {
            return Err("Model B is required.".into());
        }
        let method = self.form.value("Method");
        let needs_base = matches!(
            method.as_str(),
            "ties" | "dare_ties" | "dare_linear" | "task_arithmetic" | "della"
        );
        let base = self.form.value("Base Model");
        if needs_base && (base.is_empty() || base == "(not selected)") {
            return Err(format!(
                "Method `{method}` requires a base model for task-vector computation."
            ));
        }
        if self.form.value("Output Dir").is_empty() {
            return Err("Output Dir is required.".into());
        }
        Ok(())
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Dir"))
    }

    pub fn config_summary(&self) -> Vec<String> {
        let method = self.form.value("Method");
        let mut summary = vec![
            format!("Model A: {}", self.form.value("Model A")),
            format!("Model B: {}", self.form.value("Model B")),
            format!("Method:  {method}"),
            format!("Output:  {}", self.form.value("Output Dir")),
        ];
        let base = self.form.value("Base Model");
        if !base.is_empty() && base != "(not selected)" {
            summary.push(format!("Base:    {base}"));
        }
        summary.push(String::new());
        summary.push("Proceed?".into());
        summary
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec!["merge".to_string()];

        args.extend(["--model-a".into(), self.form.value("Model A")]);
        args.extend(["--model-b".into(), self.form.value("Model B")]);
        args.extend(["--output".into(), self.form.value("Output Dir")]);
        args.extend(["--method".into(), self.form.value("Method")]);
        args.extend(["--dtype".into(), self.form.value("Output Dtype")]);

        let base = self.form.value("Base Model");
        if !base.is_empty() && base != "(not selected)" {
            args.extend(["--base".into(), base]);
        }

        let method = self.form.value("Method");
        if method == "slerp" {
            args.extend(["--t".into(), self.form.value("Interpolation t")]);
        }
        if matches!(
            method.as_str(),
            "linear" | "ties" | "dare_linear" | "dare_ties"
        ) {
            args.extend(["--weight-a".into(), self.form.value("Weight A")]);
            args.extend(["--weight-b".into(), self.form.value("Weight B")]);
        }
        if matches!(
            method.as_str(),
            "ties" | "dare_ties" | "dare_linear" | "della" | "breadcrumbs"
        ) {
            args.extend(["--density".into(), self.form.value("Density")]);
        }

        args
    }
}

// ── Rendering ──────────────────────────────────────────────────────────

impl MergeTab {
    pub fn render(&mut self, area: Rect, buf: &mut Buffer) {
        let [config_area, right_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        let method = self.method_snapshot();
        self.form
            .render_list(config_area, buf, "Merge Configuration", move |f| {
                Self::field_visible(&method, f)
            });

        let [status_area, log_area] =
            Layout::vertical([Constraint::Length(8), Constraint::Min(0)]).areas(right_area);
        self.render_status(status_area, buf);
        self.log.render(log_area, buf, "Merge Log");
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
            MergeStatus::Idle => {
                lines.push(status_line(StatusTone::Idle, "Idle", None));
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    "  [S] Start merge  [x] Cancel",
                    THEME.text_muted,
                )));
            }
            MergeStatus::Running => {
                lines.push(status_line(StatusTone::Running, "Merging", None));
            }
            MergeStatus::Completed { output } => {
                lines.push(status_line(StatusTone::Completed, "Completed", None));
                lines.push(Line::from(vec![
                    Span::styled("  Output ", THEME.kv_key),
                    Span::styled(output.clone(), THEME.kv_value),
                ]));
            }
            MergeStatus::Failed(msg) => {
                lines.push(status_line(StatusTone::Failed, "Failed", Some(msg)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

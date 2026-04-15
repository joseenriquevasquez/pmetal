//! Distillation configuration and control tab.
//!
//! Knowledge distillation from a teacher model to a student model.
//! Supports online, offline, and progressive distillation methods.
//!
//! Form navigation, inline edit, and rendering are delegated to
//! `FormTabState`; this module owns only the distillation-specific field
//! list, CLI wiring, and metric-aware status rendering.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::training::{TrainingStatus, render_status_with_metrics};
use crate::tui::widgets::{FieldKind, FormAction, FormField, FormTabState};

/// Distillation tab state.
pub struct DistillationTab {
    pub form: FormTabState,
    pub status: TrainingStatus,
}

impl DistillationTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: TrainingStatus::Idle,
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            FormField::new(
                "Teacher Model",
                "(not selected)",
                FieldKind::ModelPicker,
                "Models",
            ),
            FormField::new(
                "Student Model",
                "(not selected)",
                FieldKind::ModelPicker,
                "Models",
            ),
            FormField::new(
                "Method",
                "online",
                FieldKind::Enum {
                    options: vec!["online".into(), "offline".into(), "progressive".into()],
                },
                "Distillation",
            ),
            FormField::new(
                "Loss Type",
                "kl_divergence",
                FieldKind::Enum {
                    options: vec![
                        "kl_divergence".into(),
                        "jensen_shannon".into(),
                        "soft_cross_entropy".into(),
                        "mse_loss".into(),
                    ],
                },
                "Distillation",
            ),
            FormField::new(
                "Temperature",
                "2.0",
                FieldKind::Number {
                    min: 0.1,
                    max: 100.0,
                },
                "Distillation",
            ),
            FormField::new(
                "Alpha",
                "0.5",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "Distillation",
            ),
            FormField::new("Rationale", "Disabled", FieldKind::Toggle, "Distillation"),
            FormField::new(
                "Rationale Weight",
                "1.0",
                FieldKind::Number {
                    min: 0.0,
                    max: 10.0,
                },
                "Distillation",
            ),
            FormField::new(
                "Learning Rate",
                "2e-5",
                FieldKind::Number {
                    min: 1e-8,
                    max: 1.0,
                },
                "Training",
            ),
            FormField::new(
                "Batch Size",
                "1",
                FieldKind::Integer { min: 1, max: 128 },
                "Training",
            ),
            FormField::new(
                "Epochs",
                "1",
                FieldKind::Integer { min: 1, max: 100 },
                "Training",
            ),
            FormField::new(
                "Max Seq Len",
                "1024",
                FieldKind::Integer {
                    min: 64,
                    max: 131072,
                },
                "Training",
            ),
            FormField::new(
                "LoRA Rank",
                "16",
                FieldKind::Integer { min: 1, max: 256 },
                "LoRA",
            ),
            FormField::new(
                "LoRA Alpha",
                "32",
                FieldKind::Number {
                    min: 1.0,
                    max: 512.0,
                },
                "LoRA",
            ),
            FormField::new(
                "Dataset",
                "(not selected)",
                FieldKind::DatasetPicker,
                "Data",
            ),
            FormField::new(
                "Output Dir",
                "./output/distilled",
                FieldKind::Text,
                "Output",
            ),
        ]
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
        self.form.next_param(|_| true);
    }
    pub fn prev_param(&mut self) {
        self.form.prev_param(|_| true);
    }

    // ── Setters ─────────────────────────────────────────────────────────

    pub fn set_teacher(&mut self, model_id: &str) {
        self.form.set_value("Teacher Model", model_id);
    }

    pub fn set_student(&mut self, model_id: &str) {
        self.form.set_value("Student Model", model_id);
        // Auto-update the output dir with the student name so trained
        // artifacts land somewhere descriptive.
        let short_name = super::model_short_name(model_id);
        self.form
            .set_value("Output Dir", format!("./output/{short_name}--distill"));
    }

    pub fn focus_field(&mut self, label: &str) {
        if let Some(idx) = self.form.fields.iter().position(|f| f.label == label) {
            // Advance by skipping until we hit the target; the shared
            // navigation helper keeps list_state in sync.
            let current = self.form.field_idx();
            let count = self.form.fields.len();
            let forward = (count + idx - current) % count;
            for _ in 0..forward {
                self.form.next_param(|_| true);
            }
        }
    }

    pub fn set_dataset(&mut self, path: &str) {
        self.form.set_value("Dataset", path);
    }

    // ── Config ──────────────────────────────────────────────────────────

    pub fn validate_config(&self) -> Result<(), String> {
        if self.form.value("Teacher Model") == "(not selected)" {
            return Err("Teacher model is required.".into());
        }
        if self.form.value("Student Model") == "(not selected)" {
            return Err("Student model is required.".into());
        }
        if self.form.value("Dataset") == "(not selected)" {
            return Err("Dataset is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Teacher:   {}", self.form.value("Teacher Model")),
            format!("Student:   {}", self.form.value("Student Model")),
            format!("Dataset:   {}", self.form.value("Dataset")),
            format!("Method:    {}", self.form.value("Method")),
            format!("Loss:      {}", self.form.value("Loss Type")),
            format!("Temp:      {}", self.form.value("Temperature")),
            format!("Alpha:     {}", self.form.value("Alpha")),
            String::new(),
            "Proceed?".into(),
        ]
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Dir"))
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        let mut args = vec!["distill".to_string()];

        args.extend(["--teacher".into(), self.form.value("Teacher Model")]);
        args.extend(["--student".into(), self.form.value("Student Model")]);
        args.extend(["--dataset".into(), self.form.value("Dataset")]);
        args.extend(["--output".into(), self.form.value("Output Dir")]);
        args.extend(["--method".into(), self.form.value("Method")]);
        args.extend(["--loss-type".into(), self.form.value("Loss Type")]);
        args.extend(["--temperature".into(), self.form.value("Temperature")]);
        args.extend(["--alpha".into(), self.form.value("Alpha")]);
        args.extend(["--learning-rate".into(), self.form.value("Learning Rate")]);
        args.extend(["--batch-size".into(), self.form.value("Batch Size")]);
        args.extend(["--epochs".into(), self.form.value("Epochs")]);
        args.extend(["--max-seq-len".into(), self.form.value("Max Seq Len")]);
        args.extend(["--lora-r".into(), self.form.value("LoRA Rank")]);
        args.extend(["--lora-alpha".into(), self.form.value("LoRA Alpha")]);

        if self.form.value("Rationale") == "Enabled" {
            args.push("--rationale".into());
            args.extend([
                "--rationale-weight".into(),
                self.form.value("Rationale Weight"),
            ]);
        }

        args
    }
}

impl DistillationTab {
    /// Render the full distillation tab with embedded dashboard metrics.
    pub fn render_with_metrics(
        &mut self,
        area: Rect,
        buf: &mut Buffer,
        samples: &[MetricSample],
        throughput: &[u64],
    ) {
        let [config_area, status_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.form
            .render_list(config_area, buf, "Distillation Configuration", |_| true);
        render_status_with_metrics(&self.status, samples, throughput, status_area, buf);
    }
}

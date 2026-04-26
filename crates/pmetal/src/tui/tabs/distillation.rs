//! Distillation configuration and control tab.
//!
//! Knowledge distillation from a teacher model to a student model.
//! Supports online, offline, and progressive distillation methods.
//!
//! Form navigation, inline edit, and rendering are delegated to
//! `FormTabState`; this module owns only the distillation-specific
//! CLI wiring and metric-aware status rendering.
//! Fields are driven by [`DistillSpec::field_descriptors`].

use std::path::PathBuf;

use pmetal_core::JobFields as _;
use pmetal_core::jobs::DistillSpec;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::training::{TrainingStatus, render_status_with_metrics};
use crate::tui::widgets::{FormAction, FormTabState};

/// Distillation tab state.
pub struct DistillationTab {
    pub form: FormTabState,
    pub status: TrainingStatus,
}

impl DistillationTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::from_spec_default::<DistillSpec>(),
            status: TrainingStatus::Idle,
        }
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

    /// DistillSpec uses "Teacher" as the label (not "Teacher Model").
    pub fn set_teacher(&mut self, model_id: &str) {
        self.form.set_value("Teacher", model_id);
    }

    pub fn set_student(&mut self, model_id: &str) {
        // DistillSpec uses "Student" as the label (not "Student Model").
        self.form.set_value("Student", model_id);
        // Auto-update the output dir with the student name so trained
        // artifacts land somewhere descriptive.
        let short_name = super::model_short_name(model_id);
        self.form
            .set_value("Output Dir", format!("./output/{short_name}--distill"));
    }

    pub fn focus_field(&mut self, label: &str) {
        if let Some(idx) = self.form.fields.iter().position(|f| f.label == label) {
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
        let teacher = self.form.value("Teacher");
        if teacher.is_empty() || teacher == "(not selected)" {
            return Err("Teacher model is required.".into());
        }
        let student = self.form.value("Student");
        if student.is_empty() || student == "(not selected)" {
            return Err("Student model is required.".into());
        }
        let dataset = self.form.value("Dataset");
        if dataset.is_empty() || dataset == "(not selected)" {
            return Err("Dataset is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Teacher:   {}", self.form.value("Teacher")),
            format!("Student:   {}", self.form.value("Student")),
            format!("Dataset:   {}", self.form.value("Dataset")),
            format!("Method:    {}", self.form.value("Method")),
            format!("Loss:      {}", self.form.value("Loss Type")),
            format!("Temp:      {}", self.form.value("Temperature")),
            format!("Alpha:     {}", self.form.value("Alpha (hard/soft)")),
            String::new(),
            "Proceed?".into(),
        ]
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Dir"))
    }

    /// Build CLI args from the form via [`DistillSpec::to_argv`].
    pub fn build_cli_args(&self) -> Vec<String> {
        let spec = self.spec_from_form();
        let mut args = vec!["distill".to_string()];
        args.extend(spec.to_argv());
        args
    }

    fn spec_from_form(&self) -> DistillSpec {
        let mut spec = DistillSpec::default();
        spec.teacher = self.form.value("Teacher");
        spec.student = self.form.value("Student");
        spec.dataset = self.form.value("Dataset");
        spec.output_dir = {
            let v = self.form.value("Output Dir");
            if v.is_empty() { spec.output_dir } else { v }
        };
        spec.method = {
            let v = self.form.value("Method");
            if v.is_empty() { spec.method } else { v }
        };
        spec.loss_type = {
            let v = self.form.value("Loss Type");
            if v.is_empty() { spec.loss_type } else { v }
        };
        spec.temperature = self.form.value("Temperature").parse().unwrap_or(spec.temperature);
        spec.alpha = self.form.value("Alpha (hard/soft)").parse().unwrap_or(spec.alpha);
        spec.rationale = self.form.value("Rationale Distillation") == "Enabled";
        spec.rationale_weight = self.form.value("Rationale Weight").parse().unwrap_or(spec.rationale_weight);
        spec.lora_r = self.form.value("LoRA r").parse().unwrap_or(spec.lora_r);
        spec.lora_alpha = self.form.value("LoRA α").parse().unwrap_or(spec.lora_alpha);
        spec.learning_rate = self.form.value("Learning Rate").parse().unwrap_or(spec.learning_rate);
        spec.batch_size = self.form.value("Batch Size").parse().unwrap_or(spec.batch_size);
        spec.epochs = self.form.value("Epochs").parse().unwrap_or(spec.epochs);
        spec.max_seq_len = self.form.value("Max Seq Len").parse().unwrap_or(spec.max_seq_len);
        spec.seed = self.form.value("Seed").parse().unwrap_or(spec.seed);
        spec
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

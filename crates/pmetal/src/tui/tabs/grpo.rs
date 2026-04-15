//! GRPO (Group Relative Policy Optimization) configuration and control tab.
//!
//! Form navigation, inline edit, and rendering are delegated to
//! `FormTabState`; this module owns only the GRPO-specific field list,
//! RLKD upgrade logic (switches the underlying CLI to `rlkd` when a
//! teacher model is set), and the metric-aware status rendering.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::training::{TrainingStatus, render_status_with_metrics};
use crate::tui::widgets::{FieldKind, FormAction, FormField, FormTabState};

/// GRPO tab state.
pub struct GrpoTab {
    pub form: FormTabState,
    pub status: TrainingStatus,
}

impl GrpoTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: TrainingStatus::Idle,
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            // Model
            FormField::new("Model", "(not selected)", FieldKind::ModelPicker, "Model"),
            // GRPO
            FormField::new(
                "Num Generations",
                "8",
                FieldKind::Integer { min: 2, max: 64 },
                "GRPO",
            ),
            FormField::new(
                "Beta (KL)",
                "0.001",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "GRPO",
            ),
            FormField::new(
                "Max Completion Len",
                "512",
                FieldKind::Integer { min: 32, max: 8192 },
                "GRPO",
            ),
            FormField::new(
                "GRPO Type",
                "bnpo",
                FieldKind::Enum {
                    options: vec![
                        "bnpo".into(),
                        "dr_grpo".into(),
                        "dapo".into(),
                        "reinforce".into(),
                    ],
                },
                "GRPO",
            ),
            FormField::new("Reasoning Rewards", "Disabled", FieldKind::Toggle, "GRPO"),
            FormField::new("VLM Mode", "Disabled", FieldKind::Toggle, "GRPO"),
            FormField::new(
                "Max Image Size",
                "336",
                FieldKind::Integer { min: 64, max: 2048 },
                "GRPO",
            ),
            // Reward Model
            FormField::new("Reward Model", "", FieldKind::Text, "Reward Model"),
            FormField::new(
                "RM Max Length",
                "2048",
                FieldKind::Integer {
                    min: 128,
                    max: 32768,
                },
                "Reward Model",
            ),
            FormField::new(
                "RM Weight",
                "1.0",
                FieldKind::Number {
                    min: 0.0,
                    max: 10.0,
                },
                "Reward Model",
            ),
            FormField::new("RM Template", "", FieldKind::Text, "Reward Model"),
            FormField::new(
                "Async Rewards",
                "Disabled",
                FieldKind::Toggle,
                "Reward Model",
            ),
            // Training
            FormField::new(
                "Learning Rate",
                "5e-6",
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
                "512",
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
            // Data
            FormField::new(
                "Dataset",
                "(not selected)",
                FieldKind::DatasetPicker,
                "Data",
            ),
            // Hardware
            FormField::new("Flash Attention", "Enabled", FieldKind::Toggle, "Hardware"),
            FormField::new(
                "Speculative Decoding",
                "Disabled",
                FieldKind::Toggle,
                "Hardware",
            ),
            FormField::new(
                "Draft Tokens",
                "3",
                FieldKind::Integer { min: 1, max: 16 },
                "Hardware",
            ),
            FormField::new("GRPO KV Cache Bits", "", FieldKind::Text, "Hardware"),
            // RLKD (optional teacher distillation — leave Teacher Model blank for pure GRPO)
            FormField::new("Teacher Model", "", FieldKind::Text, "RLKD"),
            FormField::new(
                "Distill Alpha",
                "0.3",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "RLKD",
            ),
            FormField::new(
                "Distill Temperature",
                "2.0",
                FieldKind::Number {
                    min: 0.5,
                    max: 10.0,
                },
                "RLKD",
            ),
            FormField::new("Anneal Alpha", "Enabled", FieldKind::Toggle, "RLKD"),
            FormField::new(
                "Final Alpha",
                "0.05",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "RLKD",
            ),
            // Output
            FormField::new("Output Dir", "./output/grpo", FieldKind::Text, "Output"),
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

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Model", model_id);
        let short_name = super::model_short_name(model_id);
        self.form
            .set_value("Output Dir", format!("./output/{short_name}--grpo"));
    }

    pub fn set_dataset(&mut self, path: &str) {
        self.form.set_value("Dataset", path);
    }

    // --- Config ---

    pub fn validate_config(&self) -> Result<(), String> {
        if self.form.value("Model") == "(not selected)" {
            return Err("Model is required.".into());
        }
        if self.form.value("Dataset") == "(not selected)" {
            return Err("Dataset is required.".into());
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        let teacher = self.form.value("Teacher Model");
        let mode = if teacher.is_empty() { "GRPO" } else { "RLKD" };
        let mut summary = vec![
            format!("Mode:        {}", mode),
            format!("Model:       {}", self.form.value("Model")),
            format!("Dataset:     {}", self.form.value("Dataset")),
            format!("Generations: {}", self.form.value("Num Generations")),
            format!("Beta:        {}", self.form.value("Beta (KL)")),
            format!("GRPO Type:   {}", self.form.value("GRPO Type")),
            format!("LR:          {}", self.form.value("Learning Rate")),
        ];
        if !teacher.is_empty() {
            summary.push(format!("Teacher:     {}", teacher));
            summary.push(format!(
                "Alpha:       {} → {} (anneal={})",
                self.form.value("Distill Alpha"),
                self.form.value("Final Alpha"),
                self.form.value("Anneal Alpha"),
            ));
            summary.push(format!(
                "Temperature: {}",
                self.form.value("Distill Temperature")
            ));
        }
        if self.form.value("VLM Mode") == "Enabled" {
            summary.push(format!(
                "VLM:         enabled (max_img={}px)",
                self.form.value("Max Image Size")
            ));
        }
        let rm_path = self.form.value("Reward Model");
        if !rm_path.is_empty() {
            summary.push(format!(
                "Reward Model: {} (weight={})",
                rm_path,
                self.form.value("RM Weight")
            ));
        }
        summary.push(String::new());
        summary.push("Proceed?".into());
        summary
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Dir"))
    }

    pub fn build_cli_args(&self) -> Vec<String> {
        // When a teacher model is provided, switch to the RLKD subcommand which
        // adds distillation on top of the GRPO policy gradient objective.
        let teacher = self.form.value("Teacher Model");
        let use_rlkd = !teacher.is_empty();

        let subcommand = if use_rlkd { "rlkd" } else { "grpo" };
        let mut args = vec![subcommand.to_string()];

        args.extend(["--model".into(), self.form.value("Model")]);
        args.extend(["--dataset".into(), self.form.value("Dataset")]);
        args.extend(["--output".into(), self.form.value("Output Dir")]);
        args.extend([
            "--num-generations".into(),
            self.form.value("Num Generations"),
        ]);
        args.extend(["--beta".into(), self.form.value("Beta (KL)")]);
        args.extend(["--learning-rate".into(), self.form.value("Learning Rate")]);
        args.extend(["--batch-size".into(), self.form.value("Batch Size")]);
        args.extend(["--epochs".into(), self.form.value("Epochs")]);
        args.extend(["--max-seq-len".into(), self.form.value("Max Seq Len")]);
        args.extend(["--lora-r".into(), self.form.value("LoRA Rank")]);
        args.extend(["--lora-alpha".into(), self.form.value("LoRA Alpha")]);
        args.extend([
            "--max-completion-length".into(),
            self.form.value("Max Completion Len"),
        ]);

        let grpo_type = self.form.value("GRPO Type");
        if grpo_type == "dapo" {
            args.push("--dapo".into());
        }

        if self.form.value("Reasoning Rewards") == "Enabled" {
            args.push("--reasoning-rewards".into());
        }
        if self.form.value("Flash Attention") == "Disabled" {
            args.push("--no-flash-attention".into());
        }
        if self.form.value("Speculative Decoding") == "Enabled" {
            args.push("--speculative".into());
            args.extend([
                "--speculative-draft-tokens".into(),
                self.form.value("Draft Tokens"),
            ]);
        }
        let grpo_kv_bits = self.form.value("GRPO KV Cache Bits");
        if !grpo_kv_bits.is_empty() {
            args.push("--grpo-kv-bits".to_string());
            args.push(grpo_kv_bits);
        }
        if self.form.value("VLM Mode") == "Enabled" {
            args.push("--vlm".into());
            args.extend(["--max-image-size".into(), self.form.value("Max Image Size")]);
        }

        let rm_path = self.form.value("Reward Model");
        if !rm_path.is_empty() {
            args.extend(["--reward-model".into(), rm_path]);
            args.extend([
                "--reward-model-max-length".into(),
                self.form.value("RM Max Length"),
            ]);
            args.extend(["--reward-model-weight".into(), self.form.value("RM Weight")]);
            let rm_template = self.form.value("RM Template");
            if !rm_template.is_empty() {
                args.extend(["--reward-model-template".into(), rm_template]);
            }
            if self.form.value("Async Rewards") == "Enabled" {
                args.push("--async-rewards".into());
            }
        }

        // RLKD fields — only emitted when Teacher Model is set
        if use_rlkd {
            args.extend(["--teacher-model".into(), teacher]);
            args.extend(["--distill-alpha".into(), self.form.value("Distill Alpha")]);
            args.extend([
                "--distill-temperature".into(),
                self.form.value("Distill Temperature"),
            ]);
            if self.form.value("Anneal Alpha") == "Enabled" {
                args.push("--anneal-alpha".into());
            }
            args.extend(["--final-alpha".into(), self.form.value("Final Alpha")]);
        }

        args
    }
}

impl GrpoTab {
    /// Render the full GRPO tab with embedded dashboard metrics.
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
            .render_list(config_area, buf, "GRPO Configuration", |_| true);
        render_status_with_metrics(&self.status, samples, throughput, status_area, buf);
    }
}

//! Training configuration and control tab.
//!
//! Form navigation, inline edit, and rendering are delegated to
//! `FormTabState`; this module owns only the SFT-specific field list,
//! dataset peek logic, metric-aware status rendering, and the CLI arg
//! builder.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Widget, Wrap};

use crate::tui::tabs::dashboard::MetricSample;
use crate::tui::tabs::model_short_name;
use crate::tui::theme::{THEME, palette};
use crate::tui::widgets::{FieldKind, FormAction, FormField, FormTabState};

/// Training run status.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum TrainingStatus {
    Idle,
    Running {
        step: usize,
        epoch: usize,
        total_epochs: usize,
        total_steps: usize,
        loss: f64,
    },
    Completed {
        final_loss: f64,
        total_steps: usize,
    },
    Failed(String),
}

/// Training tab state.
pub struct TrainingTab {
    pub form: FormTabState,
    pub status: TrainingStatus,
    /// Dataset info message shown below the form (columns, seq len hint).
    pub dataset_info: Option<String>,
    /// Seq len warning/suggestion from dataset peek.
    pub seq_len_warning: Option<String>,
}

impl TrainingTab {
    pub fn new() -> Self {
        Self {
            form: FormTabState::new(Self::default_fields()),
            status: TrainingStatus::Idle,
            dataset_info: None,
            seq_len_warning: None,
        }
    }

    fn default_fields() -> Vec<FormField> {
        vec![
            // Model
            FormField::new("Model", "(not selected)", FieldKind::ModelPicker, "Model"),
            FormField::new("Architecture", "-", FieldKind::ReadOnly, "Model"),
            // Training
            FormField::new(
                "Learning Rate",
                "2e-4",
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
                "2048",
                FieldKind::Integer {
                    min: 0,
                    max: 131072,
                },
                "Training",
            ),
            FormField::new(
                "Grad Accum Steps",
                "4",
                FieldKind::Integer { min: 1, max: 256 },
                "Training",
            ),
            FormField::new(
                "Max Grad Norm",
                "1.0",
                FieldKind::Number {
                    min: 0.0,
                    max: 100.0,
                },
                "Training",
            ),
            FormField::new(
                "Warmup Steps",
                "100",
                FieldKind::Integer {
                    min: 0,
                    max: 100000,
                },
                "Training",
            ),
            FormField::new(
                "Weight Decay",
                "0.01",
                FieldKind::Number { min: 0.0, max: 1.0 },
                "Training",
            ),
            // LoRA
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
                "Quantization",
                "None",
                FieldKind::Enum {
                    options: vec!["None".into(), "NF4".into(), "FP4".into(), "INT8".into()],
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
            FormField::new("Eval Dataset", "(none)", FieldKind::Text, "Data"),
            FormField::new("Sequence Packing", "Enabled", FieldKind::Toggle, "Data"),
            // Hardware
            FormField::new("Flash Attention", "Enabled", FieldKind::Toggle, "Hardware"),
            FormField::new("Fused Optimizer", "Enabled", FieldKind::Toggle, "Hardware"),
            FormField::new("JIT Compilation", "Enabled", FieldKind::Toggle, "Hardware"),
            FormField::new(
                "Cut Cross-Entropy",
                "Disabled",
                FieldKind::Toggle,
                "Hardware",
            ),
            FormField::new(
                "ANE",
                "Disabled",
                FieldKind::Enum {
                    options: vec!["Disabled".into(), "Enabled".into()],
                },
                "Hardware",
            ),
            // Output
            FormField::new("Output Dir", "./output", FieldKind::Text, "Output"),
        ]
    }

    /// Peek at the selected dataset and populate info/warnings.
    /// Called from the app when the dataset field value changes.
    pub fn peek_dataset(&mut self, path: &str) {
        use pmetal_data::{TrainingDataset, peek_columns};
        use std::io::{BufRead, BufReader};

        self.dataset_info = None;
        self.seq_len_warning = None;

        let resolved = TrainingDataset::resolve_dataset_path_pub(std::path::Path::new(path))
            .unwrap_or_else(|_| std::path::PathBuf::from(path));

        // Get columns
        let columns = peek_columns(&resolved).unwrap_or_default();
        if !columns.is_empty() {
            self.dataset_info = Some(format!("Columns: {}", columns.join(", ")));
        }

        // Sample first 100 rows for length estimates
        let mut char_lengths: Vec<usize> = Vec::new();
        if let Ok(file) = std::fs::File::open(&resolved) {
            let reader = BufReader::new(file);
            for line in reader.lines().take(100).flatten() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(obj) =
                    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(trimmed)
                {
                    let total: usize = obj
                        .values()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.len())
                        .sum();
                    char_lengths.push(total / 4); // rough token estimate
                }
            }
        }

        if char_lengths.is_empty() {
            return;
        }

        let avg = char_lengths.iter().sum::<usize>() / char_lengths.len();
        let max = char_lengths.iter().copied().max().unwrap_or(0);
        let mut sorted = char_lengths;
        sorted.sort();
        let p95 = sorted
            .get((sorted.len() as f64 * 0.95) as usize)
            .copied()
            .unwrap_or(avg);
        let suggested = if p95 > 0 { p95.div_ceil(64) * 64 } else { 2048 };

        let max_seq_len: usize = self.form.value("Max Seq Len").parse().unwrap_or(2048);

        let info = format!(
            "~{} samples | avg ~{} tok, max ~{} tok | suggest seq_len {}",
            sorted.len(),
            avg,
            max,
            suggested
        );
        self.dataset_info = Some(info);

        if max_seq_len < avg {
            self.seq_len_warning = Some(format!(
                "WARNING: max_seq_len {} < avg tokens {}. Most samples will be truncated.",
                max_seq_len, avg
            ));
        } else if max_seq_len < suggested && max > max_seq_len {
            self.seq_len_warning = Some(format!(
                "Some samples may be truncated (max ~{} tok). Suggest max_seq_len {}.",
                max, suggested
            ));
        } else if max_seq_len > max * 2 && max > 0 {
            self.seq_len_warning = Some(format!(
                "max_seq_len {} >> data max ~{} tok. Could reduce to {} for speed.",
                max_seq_len, max, suggested
            ));
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

    /// Skip read-only rows (e.g. the auto-detected Architecture field).
    pub fn next_param(&mut self) {
        self.form
            .next_param(|f| !matches!(f.kind, FieldKind::ReadOnly));
    }

    pub fn prev_param(&mut self) {
        self.form
            .prev_param(|f| !matches!(f.kind, FieldKind::ReadOnly));
    }

    // ── Setters ─────────────────────────────────────────────────────────

    pub fn set_model(&mut self, model_id: &str) {
        self.form.set_value("Model", model_id);
        // Architecture is ReadOnly; populated once the model is actually
        // loaded by the trainer. Reset to the placeholder here so the tab
        // doesn't show a stale entry from a previous pick.
        self.form.set_value("Architecture", "(auto-detect)");
        let short_name = model_short_name(model_id);
        self.form
            .set_value("Output Dir", format!("./output/{short_name}--lora"));
    }

    /// Focus a specific field by label, stepping forward through the
    /// nav helper so list-state stays in sync.
    pub fn focus_field(&mut self, label: &str) {
        if let Some(idx) = self.form.fields.iter().position(|f| f.label == label) {
            let current = self.form.field_idx();
            let count = self.form.fields.len();
            let forward = (count + idx - current) % count;
            for _ in 0..forward {
                self.next_param();
            }
        }
    }

    pub fn set_dataset(&mut self, path: &str) {
        self.form.set_value("Dataset", path);
    }

    // --- Status updates ---

    pub fn set_status_running(
        &mut self,
        step: usize,
        epoch: usize,
        total_epochs: usize,
        total_steps: usize,
        loss: f64,
    ) {
        self.status = TrainingStatus::Running {
            step,
            epoch,
            total_epochs,
            total_steps,
            loss,
        };
    }

    pub fn set_status_completed(&mut self, final_loss: f64, total_steps: usize) {
        self.status = TrainingStatus::Completed {
            final_loss,
            total_steps,
        };
    }

    pub fn set_status_failed(&mut self, msg: &str) {
        self.status = TrainingStatus::Failed(msg.to_string());
    }

    // --- Config validation and CLI arg building ---

    pub fn validate_config(&self) -> Result<(), String> {
        let model = self.form.value("Model");
        if model == "(not selected)" || model.is_empty() {
            return Err("Model is required. Press Enter on the Model field to select one.".into());
        }
        let dataset = self.form.value("Dataset");
        if dataset == "(not selected)" || dataset.is_empty() {
            return Err(
                "Dataset is required. Press Enter on the Dataset field to select one.".into(),
            );
        }
        Ok(())
    }

    pub fn config_summary(&self) -> Vec<String> {
        vec![
            format!("Model:     {}", self.form.value("Model")),
            format!("Dataset:   {}", self.form.value("Dataset")),
            format!("LR:        {}", self.form.value("Learning Rate")),
            format!("Batch:     {}", self.form.value("Batch Size")),
            format!("Epochs:    {}", self.form.value("Epochs")),
            format!("LoRA r:    {}", self.form.value("LoRA Rank")),
            format!("Quant:     {}", self.form.value("Quantization")),
            String::new(),
            "Proceed?".into(),
        ]
    }

    pub fn output_dir(&self) -> PathBuf {
        PathBuf::from(self.form.value("Output Dir"))
    }

    pub fn build_cli_args(&self, subcommand: &str) -> Vec<String> {
        let mut args = vec![subcommand.to_string()];

        args.extend(["--model".into(), self.form.value("Model")]);
        args.extend(["--dataset".into(), self.form.value("Dataset")]);
        args.extend(["--output".into(), self.form.value("Output Dir")]);
        args.extend(["--learning-rate".into(), self.form.value("Learning Rate")]);
        args.extend(["--batch-size".into(), self.form.value("Batch Size")]);
        args.extend(["--epochs".into(), self.form.value("Epochs")]);
        args.extend(["--max-seq-len".into(), self.form.value("Max Seq Len")]);
        args.extend([
            "--gradient-accumulation-steps".into(),
            self.form.value("Grad Accum Steps"),
        ]);
        args.extend(["--max-grad-norm".into(), self.form.value("Max Grad Norm")]);
        args.extend(["--warmup-steps".into(), self.form.value("Warmup Steps")]);
        args.extend(["--weight-decay".into(), self.form.value("Weight Decay")]);
        args.extend(["--lora-r".into(), self.form.value("LoRA Rank")]);
        args.extend(["--lora-alpha".into(), self.form.value("LoRA Alpha")]);

        let quant = self.form.value("Quantization").to_lowercase();
        if quant != "none" {
            args.extend(["--quantization".into(), quant]);
        }

        let eval = self.form.value("Eval Dataset");
        if eval != "(none)" && !eval.is_empty() {
            args.extend(["--eval-dataset".into(), eval]);
        }

        if self.form.value("Flash Attention") == "Disabled" {
            args.push("--no-flash-attention".into());
        }
        if self.form.value("Fused Optimizer") == "Disabled" {
            args.push("--no-metal-fused-optimizer".into());
        }
        if self.form.value("JIT Compilation") == "Disabled" {
            args.push("--no-jit-compilation".into());
        }
        if self.form.value("Sequence Packing") == "Disabled" {
            args.push("--no-sequence-packing".into());
        }
        if self.form.value("Cut Cross-Entropy") == "Enabled" {
            args.push("--cut-cross-entropy".into());
        }
        if self.form.value("ANE") == "Enabled" {
            args.push("--ane".into());
        }

        args
    }
}

impl TrainingTab {
    /// Render the full training tab with embedded dashboard metrics.
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

        self.render_config(config_area, buf);
        render_status_with_metrics(&self.status, samples, throughput, status_area, buf);
    }

    fn render_config(&mut self, area: Rect, buf: &mut Buffer) {
        // Reserve space for optional dataset info / seq-len warning lines
        // at the bottom of the config panel. `render_list` paints the
        // form above them.
        let footer_lines =
            self.dataset_info.is_some() as u16 + self.seq_len_warning.is_some() as u16;
        let [form_area, footer_area] = if footer_lines > 0 {
            Layout::vertical([Constraint::Min(0), Constraint::Length(footer_lines)]).areas(area)
        } else {
            [area, Rect::default()]
        };

        self.form
            .render_list(form_area, buf, "Configuration", |_| true);

        if footer_lines > 0 {
            let mut lines: Vec<Line> = Vec::new();
            if let Some(ref info) = self.dataset_info {
                lines.push(Line::from(Span::styled(
                    format!("  {info}"),
                    THEME.text_dim,
                )));
            }
            if let Some(ref warn) = self.seq_len_warning {
                lines.push(Line::from(Span::styled(
                    format!("  {warn}"),
                    THEME.text_warning,
                )));
            }
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(footer_area, buf);
        }
    }
}

/// Shared status panel renderer with embedded dashboard metrics.
/// Used by Training, Distillation, and GRPO tabs.
pub fn render_status_with_metrics(
    status: &TrainingStatus,
    samples: &[MetricSample],
    _throughput: &[u64],
    area: Rect,
    buf: &mut Buffer,
) {
    let block = Block::default()
        .title(match status {
            TrainingStatus::Running { .. } => " Training Monitor ",
            TrainingStatus::Completed { .. } => " Training Complete ",
            TrainingStatus::Failed(_) => " Training Failed ",
            TrainingStatus::Idle => " Status ",
        })
        .title_style(match status {
            TrainingStatus::Running { .. } => THEME.status_running,
            TrainingStatus::Completed { .. } => THEME.status_success,
            TrainingStatus::Failed(_) => THEME.status_error,
            TrainingStatus::Idle => THEME.block_title,
        })
        .borders(Borders::ALL)
        .border_style(match status {
            TrainingStatus::Running { .. } => THEME.block_focused,
            _ => THEME.block,
        });
    let inner = block.inner(area);
    block.render(area, buf);

    match status {
        TrainingStatus::Idle => {
            let lines = vec![
                Line::from(""),
                Line::from(Span::styled("  Status: Idle", THEME.status_idle)),
                Line::from(""),
                Line::from(Span::styled(
                    "  Navigate with j/k, Edit with Enter",
                    THEME.text_dim,
                )),
                Line::from(Span::styled("  Press S to start", THEME.text_dim)),
                Line::from(""),
                Line::from(Span::styled("  Toggles cycle on Enter", THEME.text_muted)),
                Line::from(Span::styled(
                    "  Pickers open a selection dialog",
                    THEME.text_muted,
                )),
            ];
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(inner, buf);
        }
        TrainingStatus::Running {
            step,
            epoch,
            total_epochs,
            total_steps,
            loss: _,
        } => {
            // Split into: stats | sparkline | timing
            let [stats_area, spark_area, timing_area, hint_area] = Layout::vertical([
                Constraint::Length(10),
                Constraint::Length(5),
                Constraint::Fill(1),
                Constraint::Length(1),
            ])
            .areas(inner);

            // --- Stats ---
            let last = samples.last();
            let loss_trend = if samples.len() >= 10 {
                let recent: f64 = samples[samples.len() - 5..]
                    .iter()
                    .map(|s| s.loss)
                    .sum::<f64>()
                    / 5.0;
                let prev: f64 = samples[samples.len() - 10..samples.len() - 5]
                    .iter()
                    .map(|s| s.loss)
                    .sum::<f64>()
                    / 5.0;
                if recent < prev * 0.99 {
                    " (decreasing)"
                } else if recent > prev * 1.01 {
                    " (increasing)"
                } else {
                    " (plateau)"
                }
            } else {
                ""
            };

            // Epoch display
            let epoch_display = if *total_epochs > 0 {
                format!("Epoch {}/{}", epoch + 1, total_epochs)
            } else {
                format!("Epoch {}", epoch + 1)
            };

            let step_display = if *total_steps > 0 {
                let pct = *step as f64 / *total_steps as f64 * 100.0;
                format!("{step}/{total_steps} ({pct:.1}%)")
            } else {
                format!("step {step}")
            };

            let mut stat_lines = vec![
                Line::from(vec![
                    Span::styled("  ", THEME.text),
                    Span::styled(&epoch_display, THEME.status_running),
                    Span::styled("  |  ", THEME.text_dim),
                    Span::styled("Running", THEME.status_running),
                ]),
                Line::from(""),
            ];

            // Progress gauge (only if total_steps known)
            if *total_steps > 0 {
                let gauge_area = Rect {
                    x: stats_area.x + 1,
                    y: stats_area.y + 2,
                    width: stats_area.width.saturating_sub(2),
                    height: 1,
                };
                let ratio = (*step as f64 / *total_steps as f64).clamp(0.0, 1.0);
                Gauge::default()
                    .gauge_style(palette::CHART_2)
                    .ratio(ratio)
                    .label(Span::styled(step_display.clone(), THEME.text))
                    .render(gauge_area, buf);
                stat_lines.push(Line::from("")); // placeholder for gauge row
            } else {
                stat_lines.push(Line::from(vec![
                    Span::styled("  Step:     ", THEME.kv_key),
                    Span::styled(&step_display, THEME.kv_value),
                ]));
            }

            if let Some(s) = last {
                stat_lines.push(Line::from(vec![
                    Span::styled("  Loss:     ", THEME.kv_key),
                    Span::styled(format!("{:.4}", s.loss), THEME.kv_value),
                    Span::styled(loss_trend, THEME.text_dim),
                ]));
                stat_lines.push(Line::from(vec![
                    Span::styled("  LR:       ", THEME.kv_key),
                    Span::styled(format!("{:.2e}", s.lr), THEME.kv_value),
                ]));
                stat_lines.push(Line::from(vec![
                    Span::styled("  Tok/sec:  ", THEME.kv_key),
                    Span::styled(format!("{:.0}", s.tok_sec), THEME.kv_value),
                ]));
                let min_loss = samples
                    .iter()
                    .map(|s| s.loss)
                    .filter(|l| *l > 0.0)
                    .fold(f64::MAX, f64::min);
                if min_loss < f64::MAX {
                    stat_lines.push(Line::from(vec![
                        Span::styled("  Min loss: ", THEME.kv_key),
                        Span::styled(format!("{:.4}", min_loss), THEME.text_success),
                    ]));
                }
            }

            Paragraph::new(stat_lines)
                .wrap(Wrap { trim: false })
                .render(stats_area, buf);

            // --- Loss sparkline ---
            let loss_u64: Vec<u64> = samples
                .iter()
                .rev()
                .take(60)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|s| {
                    // Scale to 0-100 range for sparkline
                    let min = samples.iter().map(|s| s.loss).fold(f64::MAX, f64::min);
                    let max = samples.iter().map(|s| s.loss).fold(f64::MIN, f64::max);
                    let range = (max - min).max(0.001);
                    ((s.loss - min) / range * 100.0) as u64
                })
                .collect();

            if !loss_u64.is_empty() {
                Sparkline::default()
                    .block(
                        Block::default()
                            .title(" Loss Trend ")
                            .title_style(THEME.block_title)
                            .borders(Borders::ALL)
                            .border_style(THEME.block),
                    )
                    .data(&loss_u64)
                    .style(palette::CHART_1)
                    .render(spark_area, buf);
            } else {
                Block::default()
                    .title(" Loss Trend ")
                    .title_style(THEME.block_title)
                    .borders(Borders::ALL)
                    .border_style(THEME.block)
                    .render(spark_area, buf);
            }

            // --- Timing breakdown ---
            if let Some(last) = last {
                let timing_block = Block::default()
                    .title(" Timing ")
                    .title_style(THEME.block_title)
                    .borders(Borders::ALL)
                    .border_style(THEME.block);
                let timing_inner = timing_block.inner(timing_area);
                timing_block.render(timing_area, buf);

                let total = last.total_ms.max(1.0);

                // Detect ANE vs standard training
                let ane_sum = last.ane_fwd_ms
                    + last.ane_bwd_ms
                    + last.rmsnorm_ms
                    + last.cblas_ms
                    + last.adam_ms;
                let is_ane = ane_sum > 0.0;

                if is_ane {
                    // ANE timing breakdown
                    let components = [
                        ("ANE fwd", last.ane_fwd_ms, palette::CHART_1),
                        ("ANE bwd", last.ane_bwd_ms, palette::CHART_2),
                        ("RMSNorm", last.rmsnorm_ms, palette::CHART_3),
                        ("cblas dW", last.cblas_ms, palette::CHART_4),
                        ("Adam", last.adam_ms, palette::CHART_5),
                    ];

                    let constraints: Vec<Constraint> = components
                        .iter()
                        .map(|_| Constraint::Length(1))
                        .chain(std::iter::once(Constraint::Fill(1)))
                        .collect();
                    let rows = Layout::vertical(constraints).split(timing_inner);

                    for (i, (name, ms, color)) in components.iter().enumerate() {
                        if i >= rows.len() - 1 {
                            break;
                        }
                        let ratio = (ms / total).clamp(0.0, 1.0);
                        let label = format!("{:>8}: {:5.1}ms ({:.0}%)", name, ms, ratio * 100.0);
                        Gauge::default()
                            .gauge_style(*color)
                            .ratio(ratio)
                            .label(Span::styled(label, THEME.text))
                            .render(rows[i], buf);
                    }

                    if let Some(&total_row) = rows.last() {
                        // Only show steps/min when total_ms is populated (> 1.0 guards the
                        // max(1.0) floor that prevents divide-by-zero but yields 60000).
                        let steps_per_min_label = if last.total_ms > 1.0 {
                            format!("  ({:.0} steps/min)", 60000.0 / total)
                        } else {
                            String::new()
                        };
                        Line::from(vec![
                            Span::styled("   Total: ", THEME.kv_key),
                            Span::styled(format!("{:.1}ms", total), THEME.kv_value),
                            Span::styled(steps_per_min_label, THEME.text_dim),
                        ])
                        .render(total_row, buf);
                    }
                } else {
                    // Standard MLX training — show step time + throughput
                    let rows = Layout::vertical([
                        Constraint::Length(1),
                        Constraint::Length(1),
                        Constraint::Length(1),
                        Constraint::Fill(1),
                    ])
                    .split(timing_inner);

                    // Step time gauge — only render meaningful values when total_ms is populated.
                    // total = last.total_ms.max(1.0); total_ms == 0 → total == 1.0 (no data).
                    let has_timing = last.total_ms > 1.0;
                    let ratio = if has_timing {
                        (total / 1000.0).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let label = if has_timing {
                        format!("Step time: {:.1}ms", total)
                    } else {
                        "Step time: —".to_string()
                    };
                    Gauge::default()
                        .gauge_style(palette::CHART_2)
                        .ratio(ratio)
                        .label(Span::styled(label, THEME.text))
                        .render(rows[0], buf);

                    let steps_per_min = if has_timing {
                        format!("{:.0}", 60000.0 / total)
                    } else {
                        "—".to_string()
                    };
                    Line::from(vec![
                        Span::styled(" Steps/min: ", THEME.kv_key),
                        Span::styled(steps_per_min, THEME.kv_value),
                    ])
                    .render(rows[1], buf);

                    Line::from(vec![
                        Span::styled(" Tok/sec:   ", THEME.kv_key),
                        Span::styled(format!("{:.0}", last.tok_sec), THEME.kv_value),
                    ])
                    .render(rows[2], buf);
                }
            }

            // Hint
            Line::from(Span::styled("  Press x to stop", THEME.text_dim)).render(hint_area, buf);
        }
        TrainingStatus::Completed {
            final_loss,
            total_steps,
        } => {
            // Show final stats plus the loss sparkline
            let [stats_area, spark_area] =
                Layout::vertical([Constraint::Length(8), Constraint::Fill(1)]).areas(inner);

            let min_loss = samples.iter().map(|s| s.loss).fold(f64::MAX, f64::min);
            let avg_loss = if samples.is_empty() {
                0.0
            } else {
                samples.iter().map(|s| s.loss).sum::<f64>() / samples.len() as f64
            };

            let lines = vec![
                Line::from(Span::styled("  Status: Completed", THEME.status_success)),
                Line::from(""),
                Line::from(vec![
                    Span::styled("  Steps:      ", THEME.kv_key),
                    Span::styled(total_steps.to_string(), THEME.kv_value),
                ]),
                Line::from(vec![
                    Span::styled("  Final Loss: ", THEME.kv_key),
                    Span::styled(format!("{final_loss:.4}"), THEME.text_success),
                ]),
                Line::from(vec![
                    Span::styled("  Min Loss:   ", THEME.kv_key),
                    Span::styled(
                        if min_loss < f64::MAX {
                            format!("{min_loss:.4}")
                        } else {
                            "-".into()
                        },
                        THEME.text_success,
                    ),
                ]),
                Line::from(vec![
                    Span::styled("  Avg Loss:   ", THEME.kv_key),
                    Span::styled(
                        if avg_loss > 0.0 {
                            format!("{avg_loss:.4}")
                        } else {
                            "-".into()
                        },
                        THEME.text_dim,
                    ),
                ]),
            ];
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(stats_area, buf);

            // Final loss curve
            let loss_u64: Vec<u64> = samples
                .iter()
                .map(|s| {
                    let min = min_loss;
                    let max = samples.iter().map(|s| s.loss).fold(f64::MIN, f64::max);
                    let range = (max - min).max(0.001);
                    ((s.loss - min) / range * 100.0) as u64
                })
                .collect();
            if !loss_u64.is_empty() {
                Sparkline::default()
                    .block(
                        Block::default()
                            .title(" Loss Curve ")
                            .title_style(THEME.block_title)
                            .borders(Borders::ALL)
                            .border_style(THEME.block),
                    )
                    .data(&loss_u64)
                    .style(palette::CHART_1)
                    .render(spark_area, buf);
            }
        }
        TrainingStatus::Failed(msg) => {
            let mut lines = vec![
                Line::from(""),
                Line::from(Span::styled("  Status: Failed", THEME.status_error)),
                Line::from(""),
            ];
            for line in msg.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {line}"),
                    THEME.text_error,
                )));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  Check Jobs tab for full output",
                THEME.text_muted,
            )));
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .render(inner, buf);
        }
    }
}

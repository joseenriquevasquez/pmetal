//! Editable form field widget.
//!
//! Supports text, number, integer, enum cycling, and toggle field types.
//! Used by the Training, Distillation, and GRPO tabs for parameter editing.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::text::{Line, Span};

use crate::tui::theme::THEME;

/// The kind of form field, which determines editing behavior.
#[derive(Debug, Clone)]
pub enum FieldKind {
    /// Free-form text input.
    Text,
    /// Floating-point number with optional bounds.
    Number { min: f64, max: f64 },
    /// Integer with optional bounds.
    Integer { min: i64, max: i64 },
    /// Cycle through a fixed set of options.
    Enum { options: Vec<String> },
    /// Toggle between Enabled/Disabled.
    Toggle,
    /// Opens a model picker modal (not inline-editable).
    ModelPicker,
    /// Opens a dataset picker modal (not inline-editable).
    DatasetPicker,
    /// Read-only display field.
    ReadOnly,
    /// Filesystem path (file or directory). Edited inline like `Text`.
    Path,
}

/// A form field with label, value, editing state, and validation.
#[derive(Debug, Clone)]
pub struct FormField {
    pub label: String,
    pub value: String,
    pub kind: FieldKind,
    pub section: String,
    pub editing: bool,
    edit_buffer: String,
    cursor: usize,
}

impl FormField {
    pub fn new(
        label: impl Into<String>,
        value: impl Into<String>,
        kind: FieldKind,
        section: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            kind,
            section: section.into(),
            editing: false,
            edit_buffer: String::new(),
            cursor: 0,
        }
    }

    /// Whether this field can be edited inline (vs. opening a modal).
    pub fn is_inline_editable(&self) -> bool {
        matches!(
            self.kind,
            FieldKind::Text
                | FieldKind::Number { .. }
                | FieldKind::Integer { .. }
                | FieldKind::Path
        )
    }

    /// Whether this field opens a picker modal.
    pub fn is_picker(&self) -> bool {
        matches!(self.kind, FieldKind::ModelPicker | FieldKind::DatasetPicker)
    }

    /// Whether this field can be toggled/cycled with Enter.
    pub fn is_cycleable(&self) -> bool {
        matches!(self.kind, FieldKind::Toggle | FieldKind::Enum { .. })
    }

    /// Start inline editing.
    pub fn start_edit(&mut self) {
        if self.is_inline_editable() {
            self.editing = true;
            self.edit_buffer = self.value.clone();
            self.cursor = self.edit_buffer.len();
        }
    }

    /// Cancel inline editing, discarding changes.
    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_buffer.clear();
        self.cursor = 0;
    }

    /// Confirm inline editing, validating and applying the new value.
    /// Returns `true` if the value was accepted.
    pub fn confirm_edit(&mut self) -> bool {
        if !self.editing {
            return false;
        }
        let new_val = self.edit_buffer.trim().to_string();
        if self.validate(&new_val) {
            self.value = new_val;
            self.editing = false;
            self.edit_buffer.clear();
            self.cursor = 0;
            true
        } else {
            false
        }
    }

    /// Cycle to the next value (for Toggle/Enum fields).
    pub fn cycle(&mut self) {
        match &self.kind {
            FieldKind::Toggle => {
                self.value = if self.value == "Enabled" {
                    "Disabled".to_string()
                } else {
                    "Enabled".to_string()
                };
            }
            FieldKind::Enum { options } => {
                if let Some(idx) = options.iter().position(|o| o == &self.value) {
                    let next = (idx + 1) % options.len();
                    self.value = options[next].clone();
                } else if let Some(first) = options.first() {
                    self.value = first.clone();
                }
            }
            _ => {}
        }
    }

    /// Handle a key event while this field is in editing mode.
    pub fn handle_edit_key(&mut self, key: KeyEvent) {
        if !self.editing {
            return;
        }
        match key.code {
            KeyCode::Char(c) => {
                // For numeric fields, only allow numeric characters
                match &self.kind {
                    FieldKind::Number { .. } => {
                        if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' {
                            self.edit_buffer.insert(self.cursor, c);
                            self.cursor += 1;
                        }
                    }
                    FieldKind::Integer { .. } => {
                        if c.is_ascii_digit() || c == '-' {
                            self.edit_buffer.insert(self.cursor, c);
                            self.cursor += 1;
                        }
                    }
                    // Text, Path, and all other inline-editable kinds accept any character.
                    _ => {
                        self.edit_buffer.insert(self.cursor, c);
                        self.cursor += c.len_utf8();
                    }
                }
            }
            KeyCode::Backspace if self.cursor > 0 => {
                let prev = self.edit_buffer[..self.cursor]
                    .chars()
                    .last()
                    .map(|c| c.len_utf8())
                    .unwrap_or(0);
                self.cursor -= prev;
                self.edit_buffer.remove(self.cursor);
            }
            KeyCode::Left if self.cursor > 0 => {
                let prev = self.edit_buffer[..self.cursor]
                    .chars()
                    .last()
                    .map(|c| c.len_utf8())
                    .unwrap_or(0);
                self.cursor -= prev;
            }
            KeyCode::Right if self.cursor < self.edit_buffer.len() => {
                let next = self.edit_buffer[self.cursor..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(0);
                self.cursor += next;
            }
            _ => {}
        }
    }

    /// Validate a potential new value against the field's constraints.
    fn validate(&self, val: &str) -> bool {
        match &self.kind {
            FieldKind::Text | FieldKind::Path => !val.is_empty(),
            FieldKind::Number { min, max } => {
                if let Ok(n) = val.parse::<f64>() {
                    n >= *min && n <= *max && n.is_finite()
                } else {
                    false
                }
            }
            FieldKind::Integer { min, max } => {
                if let Ok(n) = val.parse::<i64>() {
                    n >= *min && n <= *max
                } else {
                    false
                }
            }
            FieldKind::Enum { options } => options.contains(&val.to_string()),
            FieldKind::Toggle => val == "Enabled" || val == "Disabled",
            FieldKind::ModelPicker | FieldKind::DatasetPicker | FieldKind::ReadOnly => true,
        }
    }

    /// Build a `FormField` from a [`pmetal_core::FieldDescriptor`].
    ///
    /// Maps `pmetal_core::FieldKind` → local `FieldKind` 1:1.  The `current`
    /// parameter, when supplied, overrides the descriptor's default — useful
    /// when the form is being re-built to preserve existing user input.
    pub fn from_descriptor(
        desc: &pmetal_core::FieldDescriptor,
        current: Option<&str>,
    ) -> FormField {
        let tui_kind = core_kind_to_tui(&desc.kind);
        let value = current
            .map(|s| s.to_string())
            .unwrap_or_else(|| desc.default.display());
        // Toggle fields store "Enabled"/"Disabled"; the spec default is "true"/"false".
        let value = if matches!(desc.kind, pmetal_core::FieldKind::Toggle) {
            if value == "true" {
                "Enabled".to_string()
            } else if value == "false" {
                "Disabled".to_string()
            } else {
                value
            }
        } else {
            value
        };
        FormField::new(desc.label, value, tui_kind, desc.group)
    }

    /// Render as a single line in a form list.
    pub fn render_line(&self, key_width: usize, selected: bool) -> Line<'_> {
        let padded_key = format!("  {:>width$}: ", self.label, width = key_width);

        let (value_text, value_style) = if self.editing {
            let display = if self.edit_buffer.is_empty() {
                "_".to_string()
            } else {
                // Show cursor position with a block cursor
                let mut s = self.edit_buffer.clone();
                if self.cursor < s.len() {
                    s.insert(self.cursor, '|');
                } else {
                    s.push('|');
                }
                s
            };
            (display, THEME.text_warning) // Amber for editing
        } else if self.value == "(not selected)" || self.value == "(none)" {
            (self.value.clone(), THEME.text_muted)
        } else {
            (self.value.clone(), THEME.kv_value)
        };

        // Add a hint for picker/toggle fields
        let hint = if selected && !self.editing {
            match &self.kind {
                FieldKind::ModelPicker | FieldKind::DatasetPicker => " [Enter: pick]",
                FieldKind::Toggle | FieldKind::Enum { .. } => " [Enter: cycle]",
                FieldKind::Text
                | FieldKind::Number { .. }
                | FieldKind::Integer { .. }
                | FieldKind::Path => " [Enter: edit]",
                FieldKind::ReadOnly => "",
            }
        } else {
            ""
        };

        Line::from(vec![
            Span::styled(padded_key, THEME.kv_key),
            Span::styled(value_text, value_style),
            Span::styled(hint, THEME.text_muted),
        ])
    }
}

/// Map a `pmetal_core::FieldKind` to its local TUI equivalent.
///
/// The two enums mirror each other 1:1 by design; this function is the
/// only place the mapping lives so any future divergence is caught in one
/// spot.
fn core_kind_to_tui(kind: &pmetal_core::FieldKind) -> FieldKind {
    match kind {
        pmetal_core::FieldKind::Text => FieldKind::Text,
        pmetal_core::FieldKind::Number { min, max } => FieldKind::Number {
            min: *min,
            max: *max,
        },
        pmetal_core::FieldKind::Integer { min, max } => FieldKind::Integer {
            min: *min,
            max: *max,
        },
        pmetal_core::FieldKind::Enum { options } => FieldKind::Enum {
            options: options.iter().map(|s| s.to_string()).collect(),
        },
        pmetal_core::FieldKind::Toggle => FieldKind::Toggle,
        pmetal_core::FieldKind::ModelPicker => FieldKind::ModelPicker,
        pmetal_core::FieldKind::DatasetPicker => FieldKind::DatasetPicker,
        pmetal_core::FieldKind::ReadOnly => FieldKind::ReadOnly,
        pmetal_core::FieldKind::Path => FieldKind::Path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::jobs::TrainSpec;
    use pmetal_core::JobFields;

    /// Collect descriptor-built fields for a spec, filtering out flag-only
    /// fields (Toggle defaults = Disabled/false) to match the hand-built
    /// Training tab which only surfaces the most-used toggles.
    fn fields_from_train_spec() -> Vec<FormField> {
        TrainSpec::field_descriptors()
            .iter()
            .map(|d| FormField::from_descriptor(d, None))
            .collect()
    }

    #[test]
    fn from_descriptor_model_picker_produces_model_picker_kind() {
        let descs = TrainSpec::field_descriptors();
        let model_desc = descs.iter().find(|d| d.name == "model").unwrap();
        let field = FormField::from_descriptor(model_desc, None);
        assert!(
            matches!(field.kind, FieldKind::ModelPicker),
            "model field should be ModelPicker, got {:?}",
            field.kind
        );
        assert_eq!(field.label, "Model");
        assert_eq!(field.section, "Model");
    }

    #[test]
    fn from_descriptor_dataset_picker_kind() {
        let descs = TrainSpec::field_descriptors();
        let ds_desc = descs.iter().find(|d| d.name == "dataset").unwrap();
        let field = FormField::from_descriptor(ds_desc, None);
        assert!(matches!(field.kind, FieldKind::DatasetPicker));
        assert_eq!(field.label, "Dataset");
    }

    #[test]
    fn from_descriptor_number_kind_preserves_bounds() {
        let descs = TrainSpec::field_descriptors();
        let lr_desc = descs.iter().find(|d| d.name == "learning_rate").unwrap();
        let field = FormField::from_descriptor(lr_desc, None);
        if let FieldKind::Number { min, max } = field.kind {
            assert!((min - 1e-8).abs() < 1e-12);
            assert!((max - 1.0).abs() < 1e-9);
        } else {
            panic!("Expected Number kind, got {:?}", field.kind);
        }
        // Default value should be non-empty and parseable
        let v: f64 = field.value.parse().expect("lr default should parse as f64");
        assert!(v > 0.0);
    }

    #[test]
    fn from_descriptor_integer_kind_preserves_bounds() {
        let descs = TrainSpec::field_descriptors();
        let batch_desc = descs.iter().find(|d| d.name == "batch_size").unwrap();
        let field = FormField::from_descriptor(batch_desc, None);
        assert!(matches!(field.kind, FieldKind::Integer { min: 1, .. }));
        let v: i64 = field.value.parse().expect("batch_size default should parse");
        assert!(v >= 1);
    }

    #[test]
    fn from_descriptor_enum_kind_has_options() {
        let descs = TrainSpec::field_descriptors();
        let quant_desc = descs.iter().find(|d| d.name == "quantization").unwrap();
        let field = FormField::from_descriptor(quant_desc, None);
        if let FieldKind::Enum { options } = &field.kind {
            assert!(!options.is_empty(), "quantization enum should have options");
            assert!(options.contains(&"none".to_string()) || options.contains(&"None".to_string()));
        } else {
            panic!("Expected Enum kind, got {:?}", field.kind);
        }
    }

    #[test]
    fn from_descriptor_toggle_values_are_enabled_disabled() {
        let descs = TrainSpec::field_descriptors();
        // cut_cross_entropy is a Toggle / flag field with default false
        let cce_desc = descs.iter().find(|d| d.name == "cut_cross_entropy").unwrap();
        let field = FormField::from_descriptor(cce_desc, None);
        // default false → "Disabled"
        assert_eq!(field.value, "Disabled");
    }

    #[test]
    fn from_descriptor_current_overrides_default() {
        let descs = TrainSpec::field_descriptors();
        let lr_desc = descs.iter().find(|d| d.name == "learning_rate").unwrap();
        let field = FormField::from_descriptor(lr_desc, Some("1e-5"));
        assert_eq!(field.value, "1e-5");
    }

    #[test]
    fn from_descriptor_path_kind() {
        let descs = TrainSpec::field_descriptors();
        let out_desc = descs.iter().find(|d| d.name == "output_dir").unwrap();
        let field = FormField::from_descriptor(out_desc, None);
        assert!(
            matches!(field.kind, FieldKind::Path),
            "output_dir should be Path kind"
        );
        assert!(!field.value.is_empty());
    }

    /// Sanity-check that every descriptor produces a valid (non-panicking) field.
    #[test]
    fn all_train_descriptors_produce_fields() {
        let fields = fields_from_train_spec();
        assert!(
            !fields.is_empty(),
            "should produce at least one field from TrainSpec"
        );
        // Every field should have a non-empty label and section
        for f in &fields {
            assert!(!f.label.is_empty(), "field label must not be empty");
            assert!(!f.section.is_empty(), "field section must not be empty");
        }
    }

    /// Verify the spec path round-trips through `from_descriptor` with a
    /// `current` override and that the value survives.
    #[test]
    fn current_value_survives_round_trip() {
        let descs = TrainSpec::field_descriptors();
        let model_desc = descs.iter().find(|d| d.name == "model").unwrap();
        let field = FormField::from_descriptor(model_desc, Some("my-model/test"));
        assert_eq!(field.value, "my-model/test");
    }
}

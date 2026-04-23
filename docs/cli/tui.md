# pmetal tui

Launch the full-featured terminal control center with 9 tabs.

Launch the terminal UI — a full control center with 9 tabs for monitoring, configuration, and interaction.

## Usage

```bash
pmetal tui
```

## Tabs

| Tab | Description |
|-----|-------------|
| **Dashboard** | Live loss curves (braille), LR schedule, throughput sparklines, timing breakdown gauges |
| **Device** | GPU/ANE info, Metal feature detection, memory gauge, kernel tuning, UltraFusion topology |
| **Models** | Browse cached models, HuggingFace Hub search (`S`), memory fit estimation, download |
| **Datasets** | Scan and preview local datasets (JSONL, Parquet, CSV) with line counts |
| **Training** | Configure and launch SFT/LoRA/QLoRA training runs with sectioned parameter forms |
| **Distillation** | Configure knowledge distillation (online, offline, progressive) |
| **GRPO** | Configure GRPO/DAPO reasoning training with reward functions and sampling params |
| **Inference** | Interactive chat interface with markdown rendering and generation settings sidebar |
| **Jobs** | Training run history with log viewer, status tracking, and metadata |

## Keybindings

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Switch tabs |
| `Alt+1-9` | Jump to tab directly |
| `L` | Adjust learning rate mid-run |
| `S` | Search HuggingFace Hub (Models tab) |
| `q` | Quit |

## See Also

- [pmetal dashboard](/cli/dashboard/) — Standalone dashboard
- [pmetal train](/cli/train/) — CLI training

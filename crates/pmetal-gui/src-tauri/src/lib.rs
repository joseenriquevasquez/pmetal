mod commands;
mod state;

use commands::*;
use state::AppState;
use tauri::{Manager, WindowEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("pmetal_gui=info,pmetal_trainer=info,pmetal_mlx=info,pmetal_lora=info,pmetal_models=info,pmetal=info")),
        )
        .try_init();

    let app_state = AppState::new();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(app_state)
        .setup(|app| {
            #[cfg(desktop)]
            {
                // Auto-updater disabled — requires signing key in CI.
                // app.handle()
                //     .plugin(tauri_plugin_updater::Builder::new().build())?;
                app.handle()
                    .plugin(tauri_plugin_process::init())?;
            }

            let state = app.state::<AppState>();

            // Log startup diagnostics for troubleshooting
            log_startup_diagnostics();

            // Start the broadcast -> Tauri event forwarder
            start_event_forwarder(app.handle().clone(), &state);

            // Load persisted config and refresh model cache on startup.
            // We clone the inner Arcs because AppState itself is not Clone
            // (tokio::process::Child is not Clone).
            let init_state = AppStateInit {
                config: state.config.clone(),
                cached_models: state.cached_models.clone(),
            };

            tauri::async_runtime::spawn(async move {
                init_state.load_config().await;
                init_state.refresh_cached_models().await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // System
            get_config,
            set_config,
            get_system_info,
            get_device_info,
            get_dashboard_stats,
            // Models
            list_models,
            get_model_info,
            get_model_defaults,
            download_model,
            delete_model,
            search_hub_models,
            get_trending_models,
            get_model_fit,
            add_model_directory,
            remove_model_directory,
            list_model_directories,
            // Datasets
            search_hub_datasets,
            get_trending_datasets,
            list_cached_datasets,
            peek_dataset_columns,
            // Training
            start_training,
            get_training_status,
            list_training_runs,
            stop_training,
            // Distillation
            start_distillation,
            get_distillation_status,
            list_distillation_runs,
            stop_distillation,
            // GRPO
            start_grpo,
            get_grpo_status,
            list_grpo_runs,
            stop_grpo,
            // Inference
            start_inference,
            stop_inference,
            // Adapters
            list_trained_adapters,
            // Merge / Fuse / Quantize
            merge_models,
            get_merge_strategies,
            fuse_lora,
            quantize_model,
        ])
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                // Cancel all active training/distillation/GRPO runs so Metal
                // command buffers can drain before the process exits.
                if let Some(state) = window.try_state::<AppState>() {
                    if let Ok(flags) = state.cancel_flags.try_read() {
                        for flag in flags.values() {
                            flag.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                }
                // Brief pause for in-flight GPU work, then force-exit to skip
                // C++ destructor crashes from MLX's Metal device cleanup.
                std::thread::sleep(std::time::Duration::from_millis(200));
                std::process::exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running pmetal-gui");
}

// ---------------------------------------------------------------------------
// Lightweight init handle
//
// AppState owns a `HashMap<String, tokio::process::Child>` which is not Clone.
// Rather than making AppState Clone (which would require Arc<Mutex<Child>>
// everywhere and complicate kill()), we pass just the Arcs we need for
// startup init tasks.
// ---------------------------------------------------------------------------

/// Log startup diagnostics so crash reports have context.
fn log_startup_diagnostics() {
    tracing::info!(
        version = pmetal::version::VERSION,
        arch = std::env::consts::ARCH,
        os = std::env::consts::OS,
        "PMetal GUI starting"
    );

    match pmetal::metal::MetalContext::global() {
        Ok(ctx) => {
            let props = ctx.properties();
            tracing::info!(
                gpu = %props.name,
                gpu_cores = props.gpu_core_count,
                ane_cores = props.ane_core_count,
                bandwidth_gbps = format!("{:.1}", props.memory_bandwidth_gbps),
                "Metal device initialized"
            );
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                "Metal device initialization failed — GPU features will be unavailable"
            );
        }
    }

    let device = pmetal::version::device_info();
    tracing::info!(
        total_memory_gb = format!("{:.1}", device.memory_total_gb),
        available_memory_gb = format!("{:.1}", device.memory_available_gb),
        "System memory"
    );
}

struct AppStateInit {
    config: std::sync::Arc<tokio::sync::RwLock<state::AppConfig>>,
    cached_models: std::sync::Arc<tokio::sync::RwLock<Vec<state::CachedModel>>>,
}

impl AppStateInit {
    async fn load_config(&self) {
        let path = AppState::config_path_pub();
        if let Ok(data) = tokio::fs::read_to_string(&path).await {
            if let Ok(cfg) = serde_json::from_str::<state::AppConfig>(&data) {
                *self.config.write().await = cfg;
                tracing::info!("Loaded config from {}", path.display());
            }
        }
    }

    async fn refresh_cached_models(&self) {
        // Build a temporary AppState-like struct to reuse the full scanning logic
        let app_state = AppState::new();
        *app_state.config.write().await = self.config.read().await.clone();
        app_state.refresh_cached_models().await;

        let models = app_state.cached_models.read().await.clone();
        let count = models.len();
        *self.cached_models.write().await = models;
        tracing::info!("Refreshed model cache: {count} models found");
    }
}

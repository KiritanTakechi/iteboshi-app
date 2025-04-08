use crossbeam_channel::Sender;
use crossbeam_channel::bounded;
use error::AppError;
use parking_lot::Mutex;
use std::{
    fs::{self},
    path::{Path, PathBuf},
};
use tauri::Manager;
use uuid::Uuid;

mod error;
mod multilingual;
mod recorder;
mod transcription;

// --- 应用状态定义 ---
struct AppState {
    is_recording: Mutex<bool>,
    current_recording_path: Mutex<Option<PathBuf>>,
    stop_signal_sender: Mutex<Option<Sender<()>>>,
}

// --- 错误类型别名 ---
type Result<T, E = AppError> = std::result::Result<T, E>;

// --- 辅助函数：文件与路径 ---

// 生成基于 UUID 的文件路径
fn generate_uuid_path(app_handle: &tauri::AppHandle, extension: &str) -> Result<PathBuf> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| AppError::PathResolution(format!("获取应用缓存目录失败: {}", e)))?;
    fs::create_dir_all(&cache_dir)?;
    let unique_id = Uuid::new_v4();
    let filename = format!("{}.{}", unique_id, extension);
    let full_path = cache_dir.join(filename);
    Ok(full_path)
}

// 准备用于处理的音频文件
fn prepare_audio_file(
    app_handle: &tauri::AppHandle,
    original_path: &Path,
) -> Result<(PathBuf, String)> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| AppError::PathResolution(format!("获取应用缓存目录失败: {}", e)))?;

    if original_path.starts_with(&cache_dir) {
        println!("Lib: 文件 '{}' 已在缓存目录中...", original_path.display());
        let display_filename = original_path.file_name().map_or_else(
            || original_path.to_string_lossy().into_owned(),
            |name| name.to_string_lossy().into_owned(),
        );
        Ok((original_path.to_path_buf(), display_filename))
    } else {
        println!(
            "Lib: 文件 '{}' 来自外部，将复制...",
            original_path.display()
        );
        let extension = original_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .unwrap_or_else(|| "bin".to_string());
        let new_uuid_path = generate_uuid_path(app_handle, &extension)?;
        fs::copy(original_path, &new_uuid_path).map_err(|e| {
            let _ = fs::remove_file(&new_uuid_path);
            AppError::Io(format!(
                "无法将文件 '{}' 复制到 '{}': {}",
                original_path.display(),
                new_uuid_path.display(),
                e
            ))
        })?;
        println!("Lib: 文件已复制到 '{}'", new_uuid_path.display());
        let display_filename = new_uuid_path.file_name().map_or_else(
            || format!("未知UUID文件.{}", extension),
            |name| name.to_string_lossy().into_owned(),
        );
        Ok((new_uuid_path, display_filename))
    }
}

// --- Tauri 命令 ---

// `start_recording` 命令
#[tauri::command]
async fn start_recording(
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<()> {
    let mut is_recording_guard = state.is_recording.lock();
    let mut current_path_guard = state.current_recording_path.lock();
    let mut stop_sender_guard = state.stop_signal_sender.lock();

    if *is_recording_guard {
        return Err(AppError::RecordingState("已经在录音中".to_string()));
    }
    if current_path_guard.is_some() || stop_sender_guard.is_some() {
        eprintln!("警告: 状态不一致，将强制重置并开始录音。");
        *current_path_guard = None;
        *stop_sender_guard = None;
    }

    let new_path = generate_uuid_path(&app_handle, "wav")?;
    let path_clone = new_path.clone();
    let (sender, receiver) = bounded(1);

    // 调用 recorder 模块启动线程
    crate::recorder::start(path_clone, receiver); // 使用新的公共函数名 'start'

    println!("Lib: 开始录音指令已发送。将保存到: {}", new_path.display());

    *current_path_guard = Some(new_path);
    *stop_sender_guard = Some(sender);
    *is_recording_guard = true;

    Ok(())
}

// `stop_recording` 命令
#[tauri::command]
async fn stop_recording(state: tauri::State<'_, AppState>) -> Result<String> {
    let mut is_recording_guard = state.is_recording.lock();
    let mut current_path_guard = state.current_recording_path.lock();
    let mut stop_sender_guard = state.stop_signal_sender.lock();

    if !*is_recording_guard {
        return Err(AppError::RecordingState("未在录音中，无法停止".to_string()));
    }

    if let Some(sender) = stop_sender_guard.take() {
        println!("Lib: 发送停止信号...");
        sender.send(()).map_err(AppError::from)?;
    } else {
        eprintln!("错误: 录音状态为 true 但 stop_signal_sender 为空！");
        *is_recording_guard = false;
        *current_path_guard = None;
        return Err(AppError::InternalState(
            "无法找到停止录音的信号通道".to_string(),
        ));
    }

    if let Some(path) = current_path_guard.take() {
        *is_recording_guard = false;
        println!("Lib: 停止录音指令处理完成。文件: {}", path.display());
        let path_str = path.to_string_lossy().into_owned();

        std::thread::sleep(std::time::Duration::from_millis(500));

        Ok(path_str)
    } else {
        *is_recording_guard = false;
        eprintln!("错误: 停止录音时 current_recording_path 为空！");
        Err(AppError::InternalState(
            "停止录音时未找到文件路径".to_string(),
        ))
    }
}

// `transcribe_audio` 命令
#[tauri::command]
async fn transcribe_audio(
    app_handle: tauri::AppHandle,
    file_path: String,
    // --- 扩展：从前端接收参数 (示例) ---
    // #[serde(default)] language: Option<String>, // 使用 serde default 允许前端不传
    // #[serde(default = "default_task")] task: Task, // 需要定义 default_task() -> Task
    // #[serde(default)] timestamps: bool,
    // #[serde(default)] verbose: bool,
) -> Result<String> {
    // 返回 Result<String, AppError>

    println!("Lib: 收到转录请求，原始路径: {}", file_path);
    let original_path = PathBuf::from(&file_path);

    // 1. 准备文件
    let (processing_path, display_filename) = prepare_audio_file(&app_handle, &original_path)?;

    // 2. 设置解码参数 (当前使用默认值)
    let language: Option<String> = Some("zh".to_string()); // None = 自动检测 (如果模型支持)
    let task: Option<crate::transcription::Task> = Some(crate::transcription::Task::Transcribe); // 默认转录
    // let task: Option<crate::transcription::Task> = None; // 或者让 Whisper 决定 (但不推荐)
    let timestamps: bool = false; // 默认不输出时间戳
    let verbose: bool = false; // 默认不打印详细解码日志 (可以用 RUST_LOG=debug 控制)
    let temperature: f64 = 0.0; // 默认温度 (0.0 = 不随机化)

    // --- 扩展：从前端接收参数 ---
    // 如果修改了命令签名以接收前端参数，则可以直接使用这些参数：
    // let language = language; // 直接使用从前端反序列化的 language
    // let task = Some(task); // 直接使用从前端反序列化的 task (需要 Task 实现 Deserialize 或有默认值)
    // let timestamps = timestamps;
    // let verbose = verbose;

    // 3. 调用 transcription 模块处理文件，传递所有参数
    println!(
        "Lib: 转录模块开始处理文件: {} (语言: {:?}, 任务: {:?}, 时间戳: {}, 详细: {}, 温度: {})",
        processing_path.display(),
        language,
        task,
        timestamps,
        verbose,
        temperature
    );

    let transcription_result = crate::transcription::run_whisper(
        app_handle, // 传递 app_handle
        &processing_path,
        &display_filename,
        language,
        task,
        timestamps,
        verbose,
        temperature,
    )
    .await?; // 使用 await 和 ?

    println!("Lib: 转录模块处理完成。");
    Ok(transcription_result)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    // 创建初始的应用状态
    let initial_state = AppState {
        is_recording: Mutex::new(false),
        current_recording_path: Mutex::new(None),
        stop_signal_sender: Mutex::new(None),
    };

    tauri::Builder::default()
        .manage(initial_state)
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            transcribe_audio
        ])
        .run(tauri::generate_context!())
        .expect("运行 Tauri 应用时出错");
}

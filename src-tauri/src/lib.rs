use std::{fs, path::PathBuf, sync::Mutex};

use tauri::Manager;
use uuid::Uuid;

// --- 应用状态定义 ---
struct AppState {
    is_recording: Mutex<bool>,
    current_recording_path: Mutex<Option<PathBuf>>, // 存储当前录音的完整路径
}

// --- 辅助函数：生成基于 UUID 的文件路径 ---
// 在应用缓存目录下创建一个唯一的文件路径
fn generate_uuid_path(
    app_handle: &tauri::AppHandle,
    extension: &str, // 文件扩展名, e.g., "wav"
) -> Result<PathBuf, String> {
    // 1. 获取应用缓存目录
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| format!("获取应用缓存目录失败: {}", e))?;

    // 2. 确保缓存目录存在
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("无法创建缓存目录 '{}': {}", cache_dir.display(), e))?;

    // 3. 生成一个新的 UUID v4
    let unique_id = Uuid::new_v4();

    // 4. 构建文件名和完整路径
    let filename = format!("{}.{}", unique_id, extension);
    let full_path = cache_dir.join(filename);

    Ok(full_path)
}

// --- Tauri 命令 ---

// `start_recording` 命令
#[tauri::command]
async fn start_recording(
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    // 锁定状态
    let mut is_recording_guard = state
        .is_recording
        .lock()
        .map_err(|_| "无法锁定 is_recording 状态".to_string())?;
    let mut current_path_guard = state
        .current_recording_path
        .lock()
        .map_err(|_| "无法锁定 current_recording_path 状态".to_string())?;

    // 检查是否已在录音
    if *is_recording_guard {
        return Err("已经在录音中".to_string());
    }
    // 检查路径是否异常未清理 (理论上不应发生)
    if current_path_guard.is_some() {
        eprintln!("警告: 录音状态为 false，但 current_recording_path 不为空，将覆盖。");
        // 可以选择返回错误或强制清理
        return Err("内部状态异常：路径未清理".to_string());
    }

    // 生成新的录音文件路径
    let new_path = generate_uuid_path(&app_handle, "wav")?; // 假设录音保存为 wav

    println!("后端: 开始录音。将保存到: {}", new_path.display());

    // --- 实际开始录音逻辑占位符 ---
    // 在这里启动录音线程/进程，并将音频流写入 `new_path` 文件
    // 例如: start_audio_capture(&new_path);
    // ---------------------------------

    // 更新应用状态
    *current_path_guard = Some(new_path);
    *is_recording_guard = true;

    Ok(())
}

// `stop_recording` 命令
#[tauri::command]
async fn stop_recording(state: tauri::State<'_, AppState>) -> Result<String, String> {
    // 锁定状态
    let mut is_recording_guard = state
        .is_recording
        .lock()
        .map_err(|_| "无法锁定 is_recording 状态".to_string())?;
    let mut current_path_guard = state
        .current_recording_path
        .lock()
        .map_err(|_| "无法锁定 current_recording_path 状态".to_string())?;

    // 检查是否真的在录音
    if !*is_recording_guard {
        return Err("未在录音中，无法停止".to_string());
    }

    // --- 实际停止录音逻辑占位符 ---
    // 在这里停止录音线程/进程，确保文件写入完成并关闭
    // 例如: stop_audio_capture();
    // ---------------------------------

    // 从状态中取出录音路径
    // .take() 会移除 Option 中的值，留下 None，确保路径只被处理一次
    if let Some(path) = current_path_guard.take() {
        *is_recording_guard = false; // 更新录音状态
        println!("后端: 停止录音。录音文件: {}", path.display());

        // 将 PathBuf 转换为 String 返回给前端
        let path_str = path
            .to_string_lossy() // 处理非 UTF-8 路径的可能性
            .to_string();
        Ok(path_str)
    } else {
        // 这是一个内部错误状态，理论上不应该发生
        *is_recording_guard = false; // 无论如何重置状态
        eprintln!("错误: 录音状态为 true 但 current_recording_path 为空！");
        Err("内部错误：无法找到当前录音文件的路径".to_string())
    }
}

// `transcribe_audio` 命令
#[tauri::command]
async fn transcribe_audio(
    app_handle: tauri::AppHandle, // 用于访问应用路径
    file_path: String,            // 前端传入的原始文件路径 (可能来自录音或用户选择)
) -> Result<String, String> {
    println!("后端: 收到转录请求，原始路径: {}", file_path);

    let input_path = PathBuf::from(&file_path);
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| format!("获取应用缓存目录失败: {}", e))?;

    // 用于最终处理的文件路径 (保证在 cache_dir 内)
    let processing_path: PathBuf;
    // 用于在结果中显示的文件名
    let display_filename: String;

    // --- 安全处理：判断路径来源并确保文件在缓存目录中 ---
    // 检查传入的路径是否 *已经* 在我们的缓存目录中
    if input_path.starts_with(&cache_dir) {
        println!("后端: 检测到文件已在缓存目录中 (可能来自录音)，直接使用。");
        // 认为这个路径是安全的，直接使用
        processing_path = input_path;
        display_filename = processing_path.file_name().map_or_else(
            || file_path.clone(),                       // 如果无法获取文件名，使用原始路径
            |name| name.to_string_lossy().into_owned(), // 获取文件名
        );
    } else {
        // 文件来自缓存目录之外 (用户上传的文件)，需要复制到缓存目录
        println!("后端: 检测到用户上传文件，将复制到缓存目录并使用 UUID 命名。");

        // ** 重要安全提示: **
        // ** 虽然我们在这里复制文件，但 `fs::copy` 仍然需要读取 `input_path`。**
        // ** 这仍然给恶意用户读取系统任意文件的机会 (如果 Tauri Scope 配置不当)。**
        // ** 最安全的方法是在前端读取文件内容 (例如使用 FileReader API)，**
        // ** 将内容 (如 base64 或 ArrayBuffer) 发送到一个专门的 Tauri 命令，**
        // ** 该命令负责将内容写入缓存目录中的新 UUID 文件，然后返回新路径。**
        // ** 本示例为了简化流程，采取了复制策略，请务必了解其局限性。**

        // 1. 获取原始文件扩展名
        let extension = input_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .unwrap_or_else(|| "bin".to_string()); // 如果没有扩展名，默认为 "bin"

        // 2. 生成一个新的、基于 UUID 的目标路径
        let new_uuid_path = generate_uuid_path(&app_handle, &extension)?;
        display_filename = new_uuid_path.file_name().map_or_else(
            || format!("未知文件名.{}", extension), // 备用显示名
            |name| name.to_string_lossy().into_owned(),
        );

        // 3. 尝试将文件从原始路径复制到新路径
        if let Err(e) = fs::copy(&input_path, &new_uuid_path) {
            // 如果复制失败，也尝试删除可能创建的不完整目标文件
            let _ = fs::remove_file(&new_uuid_path); // 忽略删除错误
            return Err(format!(
                "无法将用户文件 '{}' 复制到 '{}': {}",
                input_path.display(),
                new_uuid_path.display(),
                e
            ));
        }

        println!("后端: 用户文件已成功复制到: {}", new_uuid_path.display());
        processing_path = new_uuid_path; // 更新 processing_path 为缓存中的新路径
    }

    // --- 转录逻辑占位符 ---
    // 现在可以安全地使用 `processing_path` 进行 Whisper 处理
    println!(
        "后端: Whisper 处理占位符（使用安全路径: {}），返回硬编码文本。",
        processing_path.display()
    );
    // 在这里调用你的 Whisper 实现，传入 `processing_path`
    // let transcription_result = call_whisper(&processing_path)?;
    // --------------------------

    // 返回硬编码结果，包含处理时使用的文件名
    let hardcoded_transcription = format!(
        "这是来自 Tauri v2 后端的硬编码转录文本。\n处理的文件名: {}",
        display_filename
    );
    Ok(hardcoded_transcription)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // 创建初始的应用状态
    let initial_state = AppState {
        is_recording: Mutex::new(false),
        current_recording_path: Mutex::new(None),
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

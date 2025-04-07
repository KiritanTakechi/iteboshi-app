use std::{
    fs,
    path::{Path, PathBuf},
};

use parking_lot::Mutex;
use tauri::Manager;
use uuid::Uuid;

// --- 应用状态定义 ---
// 定义一个结构体来持有应用共享的状态
struct AppState {
    is_recording: Mutex<bool>,                      // 标记当前是否正在录音
    current_recording_path: Mutex<Option<PathBuf>>, // 存储当前录音文件的完整路径 (如果正在录音)
}

// --- 辅助函数：生成基于 UUID 的文件路径 ---
// 在应用缓存目录下创建一个唯一的文件路径。
// 参数:
//   app_handle: Tauri 应用句柄，用于获取缓存目录路径。
//   extension: 文件扩展名 (例如 "wav", "mp3")。
// 返回:
//   Result<PathBuf, String>: 成功时返回完整的 PathBuf，失败时返回错误信息字符串。
fn generate_uuid_path(app_handle: &tauri::AppHandle, extension: &str) -> Result<PathBuf, String> {
    // 1. 获取应用缓存目录
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| format!("获取应用缓存目录失败: {}", e))?;

    // 2. 确保缓存目录存在 (如果不存在则创建)
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("无法创建缓存目录 '{}': {}", cache_dir.display(), e))?;

    // 3. 生成一个新的 UUID v4 作为唯一标识符
    let unique_id = Uuid::new_v4();

    // 4. 构建带扩展名的文件名
    let filename = format!("{}.{}", unique_id, extension);

    // 5. 组合缓存目录和文件名，得到完整路径
    let full_path = cache_dir.join(filename);

    Ok(full_path) // 返回成功创建的路径
}

// --- 辅助函数：准备用于处理的音频文件 ---
// 接收一个原始路径，确保最终处理的文件位于应用缓存目录内。
// 如果原始文件已在缓存目录中，则直接使用；否则，将其复制到缓存目录并使用 UUID 命名。
// 参数:
//   app_handle: Tauri 应用句柄。
//   original_path: 前端传入的原始文件路径。
// 返回:
//   Result<(PathBuf, String), String>:
//     成功时返回一个元组 (processing_path, display_filename)，其中：
//       - processing_path: 保证在缓存目录内的、用于实际处理的文件路径。
//       - display_filename: 用于在前端显示的文件名 (可能是 UUID 文件名或原始文件名)。
//     失败时返回错误信息字符串。
fn prepare_audio_file(
    app_handle: &tauri::AppHandle,
    original_path: &Path, // 接收 Path 引用
) -> Result<(PathBuf, String), String> {
    // 获取缓存目录，后续判断和复制需要用到
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| format!("获取应用缓存目录失败: {}", e))?;

    // 检查原始路径是否已经在缓存目录内
    if original_path.starts_with(&cache_dir) {
        println!(
            "后端: 文件 '{}' 已在缓存目录中，直接使用。",
            original_path.display()
        );
        // 文件名优先使用路径中的文件名，失败则使用整个路径字符串
        let display_filename = original_path.file_name().map_or_else(
            || original_path.to_string_lossy().into_owned(),
            |name| name.to_string_lossy().into_owned(),
        );
        // 直接返回原始路径和提取的文件名
        Ok((original_path.to_path_buf(), display_filename))
    } else {
        // 文件来自缓存目录之外 (用户上传)，需要安全地复制进来
        println!(
            "后端: 文件 '{}' 来自外部，将复制到缓存目录并使用 UUID 命名。",
            original_path.display()
        );

        // ** 重要的安全提醒 **
        // ** 下面的 `fs::copy` 操作需要读取 `original_path`。**
        // ** 如果 Tauri 的 capabilities (特别是文件系统作用域) 配置不当，**
        // ** 这仍然可能允许前端通过精心构造的路径读取到非预期的文件。**
        // ** 最安全的方法是让前端读取文件内容并发送给后端命令，**
        // ** 由后端命令将内容写入缓存文件。**
        // ** 此处使用 `fs::copy` 是为了流程演示，生产环境请务必评估风险并采用更安全的策略！**

        // 1. 提取原始文件的扩展名 (若无则默认为 "bin")
        let extension = original_path
            .extension()
            .and_then(|ext| ext.to_str()) // 转换为 &str
            .map(|ext| ext.to_lowercase()) // 转为小写
            .unwrap_or_else(|| "bin".to_string()); // 默认值

        // 2. 在缓存目录中生成一个新的、基于 UUID 的目标路径
        let new_uuid_path = generate_uuid_path(app_handle, &extension)?;

        // 3. 尝试将文件从原始路径复制到新的 UUID 路径
        if let Err(e) = fs::copy(original_path, &new_uuid_path) {
            // 如果复制过程中发生错误，尝试删除可能已创建的不完整目标文件，防止残留
            let _ = fs::remove_file(&new_uuid_path); // 忽略删除操作本身可能发生的错误
            // 返回格式化的错误信息
            return Err(format!(
                "无法将用户文件 '{}' 复制到 '{}': {}",
                original_path.display(),
                new_uuid_path.display(),
                e
            ));
        }

        // 复制成功
        println!("后端: 用户文件已成功复制到 '{}'", new_uuid_path.display());

        // 获取新的 UUID 文件名用于显示
        let display_filename = new_uuid_path.file_name().map_or_else(
            || format!("未知UUID文件.{}", extension), // 理论上不会失败
            |name| name.to_string_lossy().into_owned(),
        );

        // 返回新的缓存路径和 UUID 文件名
        Ok((new_uuid_path, display_filename))
    }
}

// --- Tauri 命令 ---

// `start_recording` 命令
#[tauri::command]
async fn start_recording(
    state: tauri::State<'_, AppState>, // 访问共享状态
    app_handle: tauri::AppHandle,      // 访问应用句柄 (用于路径)
) -> Result<(), String> {
    // --- 状态锁定 ---
    // 使用 .lock() 获取 MutexGuard，如果锁被污染则返回错误
    let mut is_recording_guard = state.is_recording.lock();
    let mut current_path_guard = state.current_recording_path.lock();

    // --- 前置条件检查 ---
    if *is_recording_guard {
        return Err("已经在录音中".to_string());
    }
    if current_path_guard.is_some() {
        // 这是一个潜在的逻辑问题，记录警告但继续 (或者选择返回错误)
        eprintln!("警告: 录音状态为 false，但 current_recording_path 不为空，将覆盖。");
        // *current_path_guard = None; // 或者在这里强制清理
    }

    // --- 核心逻辑 ---
    // 1. 生成本次录音的文件路径
    let new_path = generate_uuid_path(&app_handle, "wav")?; // 假设录音文件格式为 wav

    println!("后端: 开始录音。将保存到: {}", new_path.display());

    // 2. **【占位符】** 在此启动实际的录音过程
    //    需要将音频数据流式写入 `new_path` 文件。
    //    这通常涉及到一个独立的线程或异步任务来处理音频捕获。
    //    例如: spawn_recording_thread(new_path.clone());
    //    (注意：如果录音失败，需要处理错误并清理状态)

    // 3. 更新应用状态以反映正在录音
    *current_path_guard = Some(new_path); // 存储录音路径
    *is_recording_guard = true; // 标记为正在录音

    Ok(()) // 表示命令成功完成
}

// `stop_recording` 命令
#[tauri::command]
async fn stop_recording(state: tauri::State<'_, AppState>) -> Result<String, String> {
    // --- 状态锁定 ---
    let mut is_recording_guard = state.is_recording.lock();
    let mut current_path_guard = state.current_recording_path.lock();

    // --- 前置条件检查 ---
    if !*is_recording_guard {
        return Err("未在录音中，无法停止".to_string());
    }

    // --- 核心逻辑 ---
    // 1. **【占位符】** 在此停止实际的录音过程
    //    需要确保所有音频数据已写入文件，并关闭文件句柄。
    //    例如: signal_recording_thread_to_stop();
    //          wait_for_recording_thread_completion();

    // 2. 从状态中取出 (`.take()`) 录音文件路径
    //    `.take()` 会将 Option<PathBuf> 中的 PathBuf 移出，并将 Option 置为 None。
    if let Some(path) = current_path_guard.take() {
        // 成功获取到路径
        *is_recording_guard = false; // 更新状态为不再录音
        println!("后端: 停止录音。录音文件: {}", path.display());

        // 3. 将 PathBuf 转换为 String 以便返回给前端
        let path_str = path
            .to_string_lossy() // 安全地处理可能无效的 UTF-8 路径
            .into_owned(); // 转换为拥有的 String
        Ok(path_str)
    } else {
        // 这是一个内部逻辑错误：状态为正在录音，但路径却为 None
        *is_recording_guard = false; // 无论如何都将状态重置为不再录音
        eprintln!("错误: 录音状态为 true 但 current_recording_path 为空！");
        Err("内部错误：无法找到当前录音文件的路径".to_string())
    }
}

// `transcribe_audio` 命令
#[tauri::command]
async fn transcribe_audio(
    app_handle: tauri::AppHandle, // 用于访问应用路径和调用辅助函数
    file_path: String,            // 前端传入的原始文件路径字符串
) -> Result<String, String> {
    println!("后端: 收到转录请求，原始路径: {}", file_path);

    // --- 文件准备与安全检查 ---
    // 1. 将字符串路径转换为 PathBuf
    let original_path = PathBuf::from(&file_path);

    // 2. 调用辅助函数，确保我们得到一个在缓存目录内的、可安全处理的文件路径
    //    `prepare_audio_file` 会处理复制用户上传文件或直接使用缓存文件的逻辑。
    let (processing_path, display_filename) = prepare_audio_file(&app_handle, &original_path)?;
    // 如果 `prepare_audio_file` 返回错误 (例如复制失败)，错误会在这里 ? 操作符处提前返回。

    // --- 转录逻辑占位符 ---
    // **【占位符】** 在此执行实际的 Whisper 转录逻辑
    //    使用 `processing_path` 作为输入文件。
    println!(
        "后端: Whisper 处理占位符（使用安全路径: {}）",
        processing_path.display()
    );
    // 调用 Whisper 实现，例如:
    // let transcription_result = match call_whisper_service(&processing_path) {
    //     Ok(text) => text,
    //     Err(e) => return Err(format!("Whisper 处理失败: {}", e)),
    // };
    // --------------------------

    // --- 返回结果 ---
    // 目前返回硬编码的文本，包含处理时使用的文件名 (UUID 名或原始缓存文件名)
    let hardcoded_transcription = format!(
        "这是来自 Tauri v2 后端的硬编码转录文本。\n处理的文件名: {}",
        display_filename // 使用 prepare_audio_file 返回的显示文件名
    );
    println!("后端: 返回硬编码的转录结果。");
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

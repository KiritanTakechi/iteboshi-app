use std::sync::Mutex;

struct AppState {
    is_recording: Mutex<bool>,
    // 未来可以添加更多状态，例如录音文件的临时路径等
    // temp_record_path: Mutex<Option<String>>,
}

// `start_recording` 命令
// 使用 tauri::State<T> 来访问共享的应用状态。
#[tauri::command]
async fn start_recording(state: tauri::State<'_, AppState>) -> Result<(), String> {
    // 尝试锁定状态以进行修改
    let mut is_recording = state
        .is_recording
        .lock()
        .map_err(|_| "无法锁定状态".to_string())?;
    if *is_recording {
        // 如果已经在录音，返回错误信息
        return Err("已经在录音中".to_string());
    }
    *is_recording = true; // 更新状态为正在录音

    println!("后端: 开始录音指令已接收。");
    // 在这里添加实际的开始录音逻辑 (例如使用 cpal 库)
    // ...

    Ok(()) // 表示命令成功执行
}

// `stop_recording` 命令
#[tauri::command]
async fn stop_recording(state: tauri::State<'_, AppState>) -> Result<String, String> {
    // 尝试锁定状态
    let mut is_recording = state
        .is_recording
        .lock()
        .map_err(|_| "无法锁定状态".to_string())?;
    if !*is_recording {
        // 如果未在录音，返回错误
        return Err("未在录音中，无法停止".to_string());
    }
    *is_recording = false; // 更新状态为停止录音

    println!("后端: 停止录音指令已接收。");
    // 在这里添加实际的停止录音逻辑，并获取录音文件路径
    // ...

    // 返回一个模拟的文件路径给前端
    // 在真实应用中，这里应该是实际保存的录音文件路径
    let mock_path = "/path/to/mock/recording.wav".to_string();
    println!("后端: 模拟返回录音文件路径: {}", mock_path);

    Ok(mock_path) // 将文件路径作为成功结果返回
}

// `transcribe_audio` 命令
// 接收前端传递的文件路径作为参数。
#[tauri::command]
async fn transcribe_audio(file_path: String) -> Result<String, String> {
    println!("后端: 收到转录请求，文件路径: {}", file_path);

    // --- 转录逻辑占位符 ---
    // 在这里应该添加调用 Whisper 模型进行转录的实际逻辑。
    // 这可能涉及：
    // 1. 使用 tauri-plugin-shell::Command 调用外部 Whisper CLI 程序。
    //    (需要确保在 capabilities 中配置了 shell:allow-execute 权限，并限制范围)
    // 2. 或者，使用 Rust 的 Whisper 绑定库 (如 whisper-rs) 直接在 Rust 中运行模型。
    //    (这通常更高效，但可能增加编译复杂性和二进制大小)

    // 目前，我们只返回一个硬编码的字符串。
    let hardcoded_transcription = format!(
        "这是来自 Tauri v2 后端的硬编码转录文本。\n处理的文件: {}",
        file_path
    );
    println!("后端: 返回硬编码的转录结果。");
    // --- 占位符结束 ---

    Ok(hardcoded_transcription) // 将硬编码的文本作为成功结果返回
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // 创建初始的应用状态
    let initial_state = AppState {
        is_recording: Mutex::new(false),
        // temp_record_path: Mutex::new(None),
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

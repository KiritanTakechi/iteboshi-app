use crossbeam_channel::{Receiver, Sender, bounded};
use hound::{SampleFormat, WavSpec, WavWriter};
use parking_lot::Mutex;
use rodio::{
    cpal::{
        self, traits::{DeviceTrait, HostTrait, StreamTrait}, BuildStreamError, FromSample, Sample as CpalSample, SampleFormat as CpalSampleFormat, SizedSample, Stream, StreamConfig, StreamError, SupportedStreamConfig
    }, Device
};
use std::{
    any::type_name,
    fs::{self, File},
    io::BufWriter,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};
use tauri::Manager;
use uuid::Uuid;

// --- 应用状态定义 ---
struct AppState {
    is_recording: Mutex<bool>,
    current_recording_path: Mutex<Option<PathBuf>>,
    stop_signal_sender: Mutex<Option<Sender<()>>>,
}

// --- 错误类型别名 ---
// 为了简化函数签名，定义一个通用的错误类型
type Result<T, E = String> = std::result::Result<T, E>;

// --- 辅助函数：文件与路径 ---

// 生成基于 UUID 的文件路径
// (保持不变)
fn generate_uuid_path(app_handle: &tauri::AppHandle, extension: &str) -> Result<PathBuf> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| format!("获取应用缓存目录失败: {}", e))?;
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("无法创建缓存目录 '{}': {}", cache_dir.display(), e))?;
    let unique_id = Uuid::new_v4();
    let filename = format!("{}.{}", unique_id, extension);
    let full_path = cache_dir.join(filename);
    Ok(full_path)
}

// 准备用于处理的音频文件
// (保持不变)
fn prepare_audio_file(
    app_handle: &tauri::AppHandle,
    original_path: &Path,
) -> Result<(PathBuf, String)> {
    let cache_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|e| format!("获取应用缓存目录失败: {}", e))?;
    if original_path.starts_with(&cache_dir) {
        println!(
            "后端: 文件 '{}' 已在缓存目录中，直接使用。",
            original_path.display()
        );
        let display_filename = original_path.file_name().map_or_else(
            || original_path.to_string_lossy().into_owned(),
            |name| name.to_string_lossy().into_owned(),
        );
        Ok((original_path.to_path_buf(), display_filename))
    } else {
        println!(
            "后端: 文件 '{}' 来自外部，将复制到缓存目录并使用 UUID 命名。",
            original_path.display()
        );
        // --- 安全警告 --- (保持不变)
        let extension = original_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .unwrap_or_else(|| "bin".to_string());
        let new_uuid_path = generate_uuid_path(app_handle, &extension)?;
        if let Err(e) = fs::copy(original_path, &new_uuid_path) {
            let _ = fs::remove_file(&new_uuid_path);
            return Err(format!(
                "无法将用户文件 '{}' 复制到 '{}': {}",
                original_path.display(),
                new_uuid_path.display(),
                e
            ));
        }
        println!("后端: 用户文件已成功复制到 '{}'", new_uuid_path.display());
        let display_filename = new_uuid_path.file_name().map_or_else(
            || format!("未知UUID文件.{}", extension),
            |name| name.to_string_lossy().into_owned(),
        );
        Ok((new_uuid_path, display_filename))
    }
}

// --- 辅助函数：音频录制核心逻辑 ---

// 初始化 WAV 写入器
fn init_wav_writer(
    path: &Path,
    spec: WavSpec,
) -> Result<Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>> {
    match File::create(path) {
        Ok(file) => match WavWriter::new(BufWriter::new(file), spec) {
            Ok(w) => Ok(Arc::new(Mutex::new(Some(w)))),
            Err(e) => Err(format!("无法创建 WavWriter (spec: {:?}): {}", spec, e)),
        },
        Err(e) => Err(format!("无法创建文件 '{}': {}", path.display(), e)),
    }
}

// 获取默认音频输入设备和配置
fn get_default_input_config() -> Result<(Device, SupportedStreamConfig)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "找不到默认输入设备".to_string())?;
    println!(
        "录音线程: 使用输入设备: {}",
        device.name().unwrap_or_else(|_| "未知名称".into())
    );
    let config = device
        .default_input_config()
        .map_err(|e| format!("无法获取默认输入配置: {}", e))?;
    println!("录音线程: 获取到默认配置: {:?}", config);
    Ok((device, config))
}

// 音频数据写入回调辅助函数 (将样本转换为 i16 并写入)
fn write_input_data<T>(data: &[T], writer_arc: &Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>)
where
    T: CpalSample,      // 输入样本类型
    i16: FromSample<T>, // 确保可以从 T 转换为 i16
{
    if let Some(writer_guard) = writer_arc.lock().as_mut() {
        for sample in data {
            let sample_i16: i16 = sample.to_sample::<i16>();
            if let Err(e) = writer_guard.write_sample(sample_i16) {
                eprintln!(
                    "录音线程错误: 写入样本 (i16 from {}) 失败: {}",
                    type_name::<T>(),
                    e
                );
                // 可以在这里决定是否停止写入，例如设置一个 AtomicBool 标志
                break;
            }
        }
    }
}

// 构建 CPAL 输入流的辅助函数
fn build_input_stream_helper<T>(
    device: &Device,
    config: &StreamConfig,
    writer_arc: Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>,
    err_fn: impl FnMut(StreamError) + Send + 'static,
) -> Result<Stream, BuildStreamError>
where
    T: CpalSample + SizedSample + Send + 'static,
    i16: FromSample<T>,
{
    device.build_input_stream(
        config,
        move |data: &[T], _: &_| {
            // 调用独立的写入函数
            write_input_data::<T>(data, &writer_arc);
        },
        err_fn,
        None, // Timeout
    )
}

// 实际录音线程的主函数
fn record_audio_thread(path: PathBuf, stop_receiver: Receiver<()>) {
    println!(
        "录音线程 (rodio 0.20 - 重构): 开始录音到 {}",
        path.display()
    );

    // 1. 获取设备和配置
    let (device, supported_config) = match get_default_input_config() {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("录音线程错误: {}", e);
            return;
        }
    };
    let stream_config = supported_config.config(); // 获取具体的 StreamConfig

    // 2. 准备 WAV 写入器 (使用获取到的配置)
    let wav_spec = WavSpec {
        channels: supported_config.channels(),
        sample_rate: supported_config.sample_rate().0,
        bits_per_sample: 16, // 固定写入 16位 Int
        sample_format: SampleFormat::Int,
    };
    let writer_arc = match init_wav_writer(&path, wav_spec) {
        Ok(arc) => arc,
        Err(e) => {
            eprintln!("录音线程错误: {}", e);
            // 尝试清理可能创建的文件
            let _ = fs::remove_file(&path);
            return;
        }
    };

    // 3. 构建 CPAL 输入流
    let writer_clone = Arc::clone(&writer_arc);
    let err_fn = |err: StreamError| {
        eprintln!("录音线程错误: 音频流错误: {}", err);
    };

    let stream_result = match supported_config.sample_format() {
        CpalSampleFormat::I16 => {
            build_input_stream_helper::<i16>(&device, &stream_config, writer_clone, err_fn)
        }
        CpalSampleFormat::F32 => {
            build_input_stream_helper::<f32>(&device, &stream_config, writer_clone, err_fn)
        }
        CpalSampleFormat::I8 => {
            build_input_stream_helper::<i8>(&device, &stream_config, writer_clone, err_fn)
        }
        CpalSampleFormat::I32 => {
            build_input_stream_helper::<i32>(&device, &stream_config, writer_clone, err_fn)
        }
        CpalSampleFormat::U16 => {
            build_input_stream_helper::<u16>(&device, &stream_config, writer_clone, err_fn)
        }
        other_format => {
            eprintln!("录音线程错误: 不支持的输入样本格式: {:?}", other_format);
            let _ = fs::remove_file(&path);
            return;
        }
    };

    let stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            eprintln!("录音线程错误: 无法构建音频输入流: {}", e);
            let _ = fs::remove_file(&path);
            return;
        }
    };

    // 4. 启动流、等待停止、完成写入
    if let Err(e) = stream.play() {
        eprintln!("录音线程错误: 无法启动音频流: {}", e);
        let _ = fs::remove_file(&path);
        return;
    }

    println!(
        "录音线程: 音频流已启动 (配置: {:?}), 等待停止信号...",
        stream_config
    );

    match stop_receiver.recv() {
        Ok(_) => println!("录音线程: 收到停止信号。"),
        Err(e) => eprintln!("录音线程错误: 接收停止信号时出错: {}", e),
    }

    println!("录音线程: 正在停止音频流并完成文件写入...");
    drop(stream); // 停止流

    // 5. Finalize WAV 文件
    if let Some(writer_instance) = writer_arc.lock().take() {
        if let Err(e) = writer_instance.finalize() {
            eprintln!("录音线程错误: finalize WavWriter 失败: {}", e);
        } else {
            println!("录音线程: 录音文件 {} 写入完成。", path.display());
        }
    } else {
        // Writer 可能在回调中因错误被置为 None，或者初始化时就失败了
        eprintln!("录音线程警告: WavWriter 实例在 finalize 前已变为 None 或未成功初始化");
        // 尝试删除不完整的文件
        let _ = fs::remove_file(&path);
    }

    println!("录音线程: 退出。");
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
        return Err("已经在录音中".to_string());
    }
    if current_path_guard.is_some() || stop_sender_guard.is_some() {
        eprintln!("警告: 状态不一致，将强制重置并开始录音。");
        *current_path_guard = None;
        *stop_sender_guard = None;
    }

    let new_path = generate_uuid_path(&app_handle, "wav")?;
    let path_clone = new_path.clone();

    let (sender, receiver) = bounded(1);

    // 启动录音线程
    thread::spawn(move || {
        record_audio_thread(path_clone, receiver);
    });

    println!("后端: 开始录音指令已发送。将保存到: {}", new_path.display());

    // 更新状态
    *current_path_guard = Some(new_path);
    *stop_sender_guard = Some(sender);
    *is_recording_guard = true;

    Ok(())
}

// `stop_recording` 命令
// (逻辑保持不变)
#[tauri::command]
async fn stop_recording(state: tauri::State<'_, AppState>) -> Result<String> {
    let mut is_recording_guard = state.is_recording.lock();
    let mut current_path_guard = state.current_recording_path.lock();
    let mut stop_sender_guard = state.stop_signal_sender.lock();

    if !*is_recording_guard {
        return Err("未在录音中，无法停止".to_string());
    }

    if let Some(sender) = stop_sender_guard.take() {
        println!("后端: 发送停止信号给录音线程...");
        let _ = sender.send(()); // 忽略发送错误
    } else {
        eprintln!("错误: 录音状态为 true 但 stop_signal_sender 为空！");
        *is_recording_guard = false;
        *current_path_guard = None;
        return Err("内部错误：无法找到停止录音的信号通道".to_string());
    }

    if let Some(path) = current_path_guard.take() {
        *is_recording_guard = false;
        println!(
            "后端: 停止录音指令处理完成。录音文件应为: {}",
            path.display()
        );
        let path_str = path.to_string_lossy().into_owned();
        Ok(path_str)
    } else {
        *is_recording_guard = false;
        eprintln!("错误: 停止录音时 current_recording_path 为空！");
        Err("内部错误：停止录音时未找到文件路径".to_string())
    }
}

// `transcribe_audio` 命令
// (逻辑保持不变)
#[tauri::command]
async fn transcribe_audio(app_handle: tauri::AppHandle, file_path: String) -> Result<String> {
    println!("后端: 收到转录请求，原始路径: {}", file_path);
    let original_path = PathBuf::from(&file_path);

    let (processing_path, display_filename) = prepare_audio_file(&app_handle, &original_path)?;

    println!(
        "后端: Whisper 处理占位符（使用安全路径: {}）",
        processing_path.display()
    );
    // --- 实际 Whisper 调用占位符 ---

    let hardcoded_transcription = format!(
        "这是来自 Tauri v2 后端的硬编码转录文本。\n处理的文件名: {}",
        display_filename
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

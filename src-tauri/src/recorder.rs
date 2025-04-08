// src-tauri/src/recorder.rs

//! 负责音频录制的核心逻辑

use crate::error::AppError; // 引入统一错误类型
use crossbeam_channel::Receiver;
use hound::{SampleFormat, WavSpec, WavWriter};
use parking_lot::Mutex;
use rodio::cpal::{
    self, BuildStreamError, Device, FromSample, Sample as CpalSample,
    SampleFormat as CpalSampleFormat, SizedSample, Stream, StreamConfig, StreamError,
    SupportedStreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use std::{
    fs::{self, File},
    io::BufWriter,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};

// --- 内部辅助函数 ---

// 初始化 WAV 写入器
fn init_wav_writer(
    path: &Path,
    spec: WavSpec,
) -> Result<Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>, AppError> {
    let file = File::create(path)?;
    let writer = WavWriter::new(BufWriter::new(file), spec)?;
    Ok(Arc::new(Mutex::new(Some(writer))))
}

// 获取默认音频输入设备和配置
fn get_default_input_config() -> Result<(Device, SupportedStreamConfig), AppError> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| AppError::AudioDevice("找不到默认输入设备".to_string()))?;
    println!(
        "Recorder: 使用输入设备: {}",
        device.name().unwrap_or_else(|_| "未知名称".into())
    );
    let config = device.default_input_config()?;
    println!("Recorder: 获取到默认配置: {:?}", config);
    Ok((device, config))
}

// 音频数据写入回调辅助函数
fn write_input_data<T>(data: &[T], writer_arc: &Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>)
where
    T: CpalSample,
    i16: FromSample<T>,
{
    if let Some(writer_guard) = writer_arc.lock().as_mut() {
        for sample in data {
            let sample_i16: i16 = sample.to_sample::<i16>();
            if let Err(e) = writer_guard.write_sample(sample_i16) {
                eprintln!(
                    "Recorder Error: 写入样本失败: {}",
                    AppError::from(e) // 使用统一错误类型打印
                );
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
            write_input_data::<T>(data, &writer_arc);
        },
        err_fn,
        None,
    )
}

// 实际录音线程的主函数
fn record_audio_thread_internal(path: PathBuf, stop_receiver: Receiver<()>) {
    println!(
        "录音线程 (rodio 0.20 - 更正2): 开始录音到 {}",
        path.display()
    );

    // --- 1. 初始化准备工作 ---
    // 将可能失败的操作放在前面，并将结果存储起来
    let setup_result = (|| -> Result<_, AppError> {
        let (device, supported_config) = get_default_input_config()?;
        let wav_spec = WavSpec {
            channels: supported_config.channels(),
            sample_rate: supported_config.sample_rate().0,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let writer_arc = init_wav_writer(&path, wav_spec)?;
        Ok((device, supported_config, writer_arc)) // 返回成功获取的资源
    })();

    // 处理初始化结果
    let (device, supported_config, writer_arc) = match setup_result {
        Ok(tuple) => tuple,
        Err(e) => {
            eprintln!("Recorder Error [初始化失败]: {}", e);
            let _ = fs::remove_file(&path); // 清理可能创建的文件
            return;
        }
    };

    // --- 2. 构建并启动音频流 ---
    // 现在我们确定 device, config, writer_arc 都有效
    let stream_config = supported_config.config(); // 获取 StreamConfig
    let writer_clone = Arc::clone(&writer_arc);
    let err_fn = |err: StreamError| {
        eprintln!("Recorder Error [音频流回调]: {}", AppError::from(err));
    };

    // 构建流
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
        _ => Err(BuildStreamError::StreamConfigNotSupported),
    };

    // 处理流构建结果
    let stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            eprintln!("录音线程错误 [构建流]: {}", AppError::from(e)); // 使用 AppError 包装
            let _ = fs::remove_file(&path);
            return;
        }
    };

    // --- 3. 播放流并等待停止信号 ---
    if let Err(e) = stream.play() {
        eprintln!("录音线程错误: 无法启动音频流: {}", AppError::from(e)); // 使用 AppError 包装
        let _ = fs::remove_file(&path);
        return;
    }

    println!(
        "Recorder: 音频流已启动 (配置: {:?}), 等待停止信号...",
        stream_config
    );

    match stop_receiver.recv() {
        Ok(_) => println!("Recorder: 收到停止信号。"),
        Err(e) => eprintln!(
            "Recorder Error: 接收停止信号时出错: {}",
            AppError::StopSignalSend(e.to_string())
        ),
    }

    println!("Recorder: 正在停止音频流并完成文件写入...");
    // --- 4. 停止流并 Finalize ---
    drop(stream); // 停止流

    // Finalize WAV 文件 (保持不变)
    if let Some(writer_instance) = writer_arc.lock().take() {
        match writer_instance.finalize() {
            Ok(_) => {
                println!("Recorder: 录音文件 {} 写入完成。", path.display());
            }
            Err(e) => {
                eprintln!(
                    "Recorder Error: finalize WavWriter 失败: {}",
                    AppError::from(e)
                );
            }
        }
    } else {
        eprintln!("Recorder Warning: WavWriter 实例在 finalize 前已变为 None");
        let _ = fs::remove_file(&path);
    }

    println!("Recorder: 录音线程退出。");
}

// --- 公共接口：启动录音线程 ---
pub fn start(path: PathBuf, stop_receiver: Receiver<()>) {
    println!(
        "Recorder Module: Spawning recording thread for {}",
        path.display()
    );
    thread::spawn(move || {
        record_audio_thread_internal(path, stop_receiver);
    });
}

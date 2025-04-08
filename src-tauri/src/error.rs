use crossbeam_channel::SendError;
use hound;
use rodio::cpal::{
    BuildStreamError, DefaultStreamConfigError, DevicesError, PlayStreamError, StreamError,
};
use serde::Serialize;
use std::io;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("文件系统错误: {0}")]
    Io(String), // 包装 IO 错误信息

    #[error("音频设备错误: {0}")]
    AudioDevice(String), // 包装设备查找/配置错误

    #[error("音频流构建错误: {0}")]
    AudioStreamBuild(String), // 包装流构建错误

    #[error("音频流播放错误: {0}")]
    AudioStreamPlay(String), // 包装流播放错误 (stream.play())

    #[error("音频流错误: {0}")]
    AudioStream(String), // 通用音频流运行时错误

    #[error("WAV 文件处理错误: {0}")]
    WavProcessing(String), // 包装 hound 错误

    #[error("Tauri 路径解析错误: {0}")]
    PathResolution(String), // 包装 Tauri 路径错误

    #[error("录音状态错误: {0}")]
    RecordingState(String), // 应用逻辑状态错误

    #[error("内部状态不一致: {0}")]
    InternalState(String), // 内部逻辑断言失败

    #[error("无法发送停止信号: {0}")]
    StopSignalSend(String), // 包装通道发送错误

    #[error("Whisper 转录错误: {0}")]
    Transcription(String),

    #[error("模型/配置加载错误: {0}")]
    ModelLoad(String), // 用于包装 hf-hub 或手动加载错误

    #[error("Candle 核心错误: {0}")]
    CandleCore(#[from] candle_core::Error), // 使用 #[from]

    #[error("Tokenizer 错误: {0}")]
    Tokenizer(String), // Tokenizer 错误类型比较复杂，暂时转为 String

    #[error("音频重采样错误: {0}")]
    Resampling(#[from] rubato::ResampleError), // 使用 #[from]

    #[error("音频预处理错误: {0}")]
    AudioPreprocessing(String),

    #[error("JSON 解析错误: {0}")]
    JsonParse(#[from] serde_json::Error),

    #[error("Hugging Face Hub API 错误: {0}")]
    HfHubApi(#[from] hf_hub::api::tokio::ApiError),

    #[error("模型文件下载失败: {0}")]
    DownloadFailed(String),
}

impl From<io::Error> for AppError {
    fn from(e: io::Error) -> Self {
        AppError::Io(e.to_string())
    }
}

impl From<hound::Error> for AppError {
    fn from(e: hound::Error) -> Self {
        AppError::WavProcessing(e.to_string())
    }
}

impl From<DevicesError> for AppError {
    fn from(e: DevicesError) -> Self {
        AppError::AudioDevice(format!("设备枚举错误: {}", e))
    }
}

impl From<DefaultStreamConfigError> for AppError {
    fn from(e: DefaultStreamConfigError) -> Self {
        AppError::AudioDevice(format!("获取默认配置错误: {}", e))
    }
}

impl From<BuildStreamError> for AppError {
    fn from(e: BuildStreamError) -> Self {
        AppError::AudioStreamBuild(e.to_string())
    }
}

impl From<PlayStreamError> for AppError {
    fn from(e: PlayStreamError) -> Self {
        AppError::AudioStreamPlay(e.to_string())
    }
}

impl From<StreamError> for AppError {
    fn from(e: StreamError) -> Self {
        AppError::AudioStream(e.to_string())
    }
}

impl From<SendError<()>> for AppError {
    fn from(e: SendError<()>) -> Self {
        AppError::StopSignalSend(e.to_string())
    }
}

impl From<tokenizers::Error> for AppError {
    fn from(e: tokenizers::Error) -> Self {
        AppError::Tokenizer(e.to_string())
    }
}

impl serde::Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

#[macro_export]
macro_rules! app_err {
    (Io, $msg:expr) => { $crate::error::AppError::Io(std::io::Error::new(std::io::ErrorKind::Other, $msg)) };
    ($variant:ident, $msg:expr) => { $crate::error::AppError::$variant($msg.to_string()) };
    ($variant:ident, $fmt:expr, $($arg:tt)*) => {
        $crate::error::AppError::$variant(format!($fmt, $($arg)*))
    };
}

use crossbeam_channel::SendError;
use hound;
use rodio::cpal::{
    BuildStreamError, DefaultStreamConfigError, DevicesError, PlayStreamError, StreamError,
};
use serde::Serialize;
use std::io;
use thiserror::Error;

#[derive(Debug, Error, Serialize)]
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
    Transcription(String), // 为未来的 Whisper 实现预留
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

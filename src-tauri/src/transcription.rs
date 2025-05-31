// src-tauri/src/transcription.rs

use crate::app_err;
use crate::error::AppError;
use crate::multilingual::LANGUAGES;
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::Config;
use candle_transformers::models::whisper::{
    self, EOT_TOKEN, NO_SPEECH_TOKENS, NO_TIMESTAMPS_TOKEN, SAMPLE_RATE, SOT_TOKEN,
    TRANSCRIBE_TOKEN, TRANSLATE_TOKEN, audio, model::Whisper,
};
use hf_hub::RepoType;
use hf_hub::api::tokio::{Api, ApiRepo};
use hound::{self, SampleFormat, WavReader};
use rand::distr::weighted::WeightedIndex;
use rand::{SeedableRng, distr::Distribution};
use rubato::{FftFixedIn, Resampler};
use std::cmp::min;
use std::fs::{self};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::OnceCell;

// --- 类型别名和常量 ---
type Result<T, E = AppError> = std::result::Result<T, E>; // 使用 AppError

const WHISPER_SAMPLING_RATE: u32 = SAMPLE_RATE as u32; // 16000
const MAX_AUDIO_LENGTH_SECS: u32 = 30;
const SAMPLES_PER_CHUNK: usize = (WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH_SECS) as usize;
const MODEL_ID: &str = "kiritan/iteboshi-medium";
const REVISION: &str = "main";

// --- 全局状态 ---
static MODEL_LOADER: OnceCell<TokioMutex<WhisperComponents>> = OnceCell::const_new();

// 存放需要共享的、加载一次的资源
struct WhisperComponents {
    model: Model, // 使用 Model enum 包装普通和量化模型 (如果支持)
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>, // 加载预计算的 Mel 滤波器
    device: Device,
    config: Config, // 存储加载的配置
}

// --- Model Enum (来自官方示例，用于支持未来可能的量化模型) ---
pub enum Model {
    Normal(whisper::model::Whisper),
    // Quantized(whisper::quantized_model::Whisper), // 暂时只实现 Normal
}

impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            // Self::Quantized(m) => &m.config,
        }
    }
    // 将模型操作委托给内部模型
    fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            // Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }
    fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            // Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }
    fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            // Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
    fn reset_kv_cache(&mut self) {
        match self {
            Model::Normal(m) => {
                m.decoder.reset_kv_cache();
            } // Model::Quantized(m) => { ... }
        }
        log::trace!("Model KV Cache 已重置"); // 添加日志确认
    }
}

// --- Helper 函数 ---

/// 异步下载文件如果不存在
async fn download_if_not_exists(
    repo_api: &ApiRepo,
    filename: &str, // 要下载的文件名
) -> Result<PathBuf, AppError> {
    log::debug!("检查并下载文件: {}", filename);
    // 直接调用 download，它会处理缓存和下载
    let path = repo_api.get(filename).await?; // 使用 get 获取本地路径
    // let path = repo_api.download(filename).await?; // download 也可以，但 get 可能更通用
    if path.exists() {
        log::debug!("文件 '{}' 已存在于缓存: {}", filename, path.display());
    } else {
        // hf_hub::get 通常会确保文件存在，如果走到这里可能意味着缓存问题？
        log::warn!(
            "文件 '{}' 在 get 调用后仍不存在于 {}",
            filename,
            path.display()
        );
        // 可以在这里尝试强制下载，但这不应该是常规流程
        // repo_api.download(filename, &path).await?; // 强制下载到指定路径？API 不支持
        return Err(app_err!(DownloadFailed, "文件 '{}' 下载后未找到", filename));
    }
    Ok(path)
}

/// 加载模型、分词器和配置 (带自动下载)
async fn load_whisper_components(
    _app_handle: &tauri::AppHandle,
) -> Result<&'static TokioMutex<WhisperComponents>, AppError> {
    MODEL_LOADER
        .get_or_try_init(|| async {
            // ... (加载 device, api, repo, 下载文件不变) ...
            log::info!("首次加载 Whisper 组件...");
            let device = select_device()?;
            log::info!("使用设备: {:?}", device);
            let api = Api::new().map_err(|e| app_err!(HfHubApi, "创建 HF Hub Api 失败: {}", e))?;
            let repo = hf_hub::Repo::with_revision(
                MODEL_ID.to_string(),
                RepoType::Model,
                REVISION.to_string(),
            );
            let repo_api = api.repo(repo);
            log::info!("使用模型仓库: {}@{}", MODEL_ID, REVISION);
            log::info!("准备下载/查找文件...");
            let config_path = download_if_not_exists(&repo_api, "config.json").await?;
            let tokenizer_path = download_if_not_exists(&repo_api, "tokenizer.json").await?;
            let model_path = download_if_not_exists(&repo_api, "model.safetensors").await?;

            log::info!("加载模型配置...");
            // 1. 加载原始 config
            let config: Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
            // 2. 克隆一份用于最后存储
            let config_to_store = config.clone();
            log::info!("模型配置加载完成。Num Mel Bins: {}", config.num_mel_bins); // 使用原始 config 读取

            // --- 加载 Mel 滤波器 ---
            log::info!("加载预计算的 Mel 滤波器...");
            // 3. 使用原始 config 读取 num_mel_bins
            let mel_bytes: &[u8] = match config.num_mel_bins {
                80 => include_bytes!("melfilters/melfilters.bytes"),
                128 => include_bytes!("melfilters/melfilters128.bytes"),
                nmel => return Err(app_err!(ModelLoad, "不支持的 num_mel_bins: {}", nmel)),
            };
            // ... (读取 mel_bytes 到 mel_filters 的逻辑不变) ...
            if mel_bytes.len() % 4 != 0 {
                return Err(app_err!(ModelLoad, "Mel 滤波器文件大小无效"));
            }
            let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
            let mut cursor = Cursor::new(mel_bytes);
            cursor
                .read_f32_into::<LittleEndian>(&mut mel_filters)
                .map_err(|e| app_err!(ModelLoad, "读取 Mel 滤波器失败: {}", e))?;
            log::info!("Mel 滤波器加载完成。");

            log::info!("加载 Tokenizer...");
            let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
            log::info!("Tokenizer 加载完成。");

            log::info!("加载模型权重...");
            let weights = candle_core::safetensors::load(&model_path, &device)?;
            let vb = VarBuilder::from_tensors(weights, DType::F32, &device);

            // 4. 使用原始的 config (移动所有权)
            let model = Whisper::load(&vb, config)?;
            log::info!("模型加载完成。");

            Ok(TokioMutex::new(WhisperComponents {
                model: Model::Normal(model),
                tokenizer,
                mel_filters,
                device,
                config: config_to_store,
            }))
        })
        .await
}

/// 选择计算设备 (Metal > CPU)
fn select_device() -> Result<Device, AppError> {
    #[cfg(feature = "metal")]
    {
        if candle_core::utils::metal_is_available() {
            match Device::new_metal(0) {
                Ok(dev) => {
                    log::info!("使用 Metal 设备。");
                    return Ok(dev);
                }
                Err(e) => log::warn!("Metal 初始化失败: {}, 回退到 CPU", e),
            }
        } else {
            log::warn!("Metal feature 已启用但未检测到可用 Metal 设备。");
        }
    }
    log::info!("使用 CPU 设备。");
    Ok(Device::Cpu)
}

/// 加载并预处理音频文件 (期望 16kHz)
fn preprocess_audio(audio_path: &Path) -> Result<Vec<f32>, AppError> {
    log::info!("预处理音频文件: {}", audio_path.display());

    match fs::metadata(audio_path) {
        Ok(metadata) => {
            log::debug!("文件元数据: 大小 = {} bytes", metadata.len());
            if metadata.len() < 44 {
                // WAV 文件头至少 44 字节
                log::warn!("文件大小异常，可能为空或损坏。");
                // 可以选择在这里返回错误
                // return Err(app_err!(AudioPreprocessing, "读取的 WAV 文件大小异常"));
            }
        }
        Err(e) => {
            log::error!("无法获取文件元数据: {}", e);
            // 也可能在这里返回错误
            // return Err(AppError::Io(format!("无法获取文件元数据: {}", e)));
        }
    }

    let mut reader = WavReader::open(audio_path)?;

    let spec = reader.spec();
    log::info!("读取 WAV 规范: {:?}", spec);

    // --- 1. 读取所有样本并转换为 f32 单声道 ---
    let samples_f32: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => {
            log::debug!("读取 f32 样本...");
            // 直接 collect f32 samples
            reader.samples::<f32>().collect::<Result<_, _>>()? // ? 转换 hound::Error -> AppError
        }
        SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => {
                    log::debug!("读取 i16 样本并归一化...");
                    // 读取所有 i16 样本，然后 map 转换
                    reader
                        .samples::<i16>()
                        .collect::<Result<Vec<_>, _>>()? // 先 collect Vec<i16>
                        .into_iter() // 转换成迭代器
                        .map(|s| s as f32 / i16::MAX as f32) // 归一化
                        .collect() // 收集为 Vec<f32>
                }
                32 => {
                    log::debug!("读取 i32 样本并归一化...");
                    reader
                        .samples::<i32>()
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .map(|s| s as f32 / i32::MAX as f32)
                        .collect()
                }
                // 可以添加对 i8, i24 (如果 hound 支持) 的处理
                // 8 => reader.samples::<i8>()...
                _ => {
                    return Err(app_err!(
                        AudioPreprocessing,
                        "不支持的整数位深: {}",
                        spec.bits_per_sample
                    ));
                }
            }
        }
    };
    log::info!("读取并转换为 f32 样本完成，数量: {}", samples_f32.len());

    let samples_mono: Vec<f32> = if spec.channels == 1 {
        samples_f32
    } else if spec.channels > 1 {
        log::info!("将 {} 声道转为单声道...", spec.channels);
        samples_f32
            .chunks_exact(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        return Err(app_err!(
            AudioPreprocessing,
            "不支持的声道数: {}",
            spec.channels
        ));
    };
    log::info!(
        "读取并转换为单声道 f32 完成，样本数: {}",
        samples_mono.len()
    );

    // --- 2. 如果需要，执行重采样 ---
    let final_samples = if spec.sample_rate != WHISPER_SAMPLING_RATE {
        log::info!(
            "重采样音频从 {} Hz 到 {} Hz...",
            spec.sample_rate,
            WHISPER_SAMPLING_RATE
        );
        let original_rate = spec.sample_rate as usize;
        let target_rate = WHISPER_SAMPLING_RATE as usize;
        let original_length = samples_mono.len();

        // a) 创建重采样器
        let chunk_size = 1024; // 或其他合适的值
        // 可以使用更详细的参数创建，如此处或之前的简单方式
        // let params = InterpolationParameters { sinc_len: 256, f_cutoff: 0.95, interpolation: InterpolationType::Linear, oversampling_factor: 256, window: WindowFunction::BlackmanHarris2 };
        // let mut resampler = FftFixedIn::<f32>::new_with_params(original_rate, target_rate, params, chunk_size, 1, 1)?;
        let mut resampler = FftFixedIn::<f32>::new(original_rate, target_rate, chunk_size, 1, 1)?;

        // b) 获取延迟并计算预期输出长度
        let delay = resampler.output_delay();
        let expected_output_len =
            (original_length as f64 * target_rate as f64 / original_rate as f64).ceil() as usize;
        log::debug!(
            "重采样延迟: {}, 预期输出长度: {}",
            delay,
            expected_output_len
        );

        // c) 创建临时输出缓冲区
        // 容量稍大于预期长度 + 延迟，以防万一
        let mut output_buffer_temp = Vec::with_capacity(expected_output_len + delay + chunk_size);

        // d) 处理主要数据块 (循环)
        let mut processed_input_frames = 0;
        loop {
            let needed_input = resampler.input_frames_next();
            let available_input = original_length - processed_input_frames;

            // 检查是否还有足够的输入来处理一个完整的块
            if available_input < needed_input {
                log::debug!(
                    "输入不足一个完整块 (需要 {}, 剩余 {}), 跳出主循环。",
                    needed_input,
                    available_input
                );
                break; // 跳出循环去处理剩余部分
            }

            // 准备输入块
            let input_chunk = vec![
                samples_mono[processed_input_frames..processed_input_frames + needed_input]
                    .to_vec(),
            ];
            log::trace!(
                "处理输入块: [{}..{}] ({} 帧)",
                processed_input_frames,
                processed_input_frames + needed_input,
                needed_input
            );

            // 处理并追加到临时缓冲区
            let resampled_chunk = resampler.process(&input_chunk, None)?;
            if let Some(channel_data) = resampled_chunk.into_iter().next() {
                log::trace!("生成输出块: {} 帧", channel_data.len());
                output_buffer_temp.extend_from_slice(&channel_data);
            }

            processed_input_frames += needed_input; // 更新已处理的帧数
        }

        // e) 处理剩余的输入帧 (使用 process_partial)
        if processed_input_frames < original_length {
            // remaining_input 需要是 &[Vec<f32>]
            let remaining_slice = &samples_mono[processed_input_frames..];
            let remaining_input_vec = vec![remaining_slice.to_vec()]; // 包装为 Vec<Vec<f32>>
            log::debug!(
                "处理剩余输入块: [{}..{}] ({} 帧)",
                processed_input_frames,
                original_length,
                remaining_slice.len()
            );

            // --- 修正点：将 &remaining_input_vec 包装在 Some() 中 ---
            let resampled_chunk = resampler.process_partial(Some(&remaining_input_vec), None)?; // 使用 Some()

            if let Some(channel_data) = resampled_chunk.into_iter().next() {
                log::debug!("生成剩余输出块: {} 帧", channel_data.len());
                output_buffer_temp.extend_from_slice(&channel_data);
            }
        }

        // f) 处理内部延迟 (冲刷缓冲区)
        // 持续调用 process_partial(None) 直到输出足够长
        log::debug!(
            "处理内部延迟，当前输出: {}, 目标: {}",
            output_buffer_temp.len(),
            expected_output_len + delay
        );
        while output_buffer_temp.len() < expected_output_len + delay {
            log::trace!("缓冲区不足，调用 process_partial(None)...");

            // --- 修正点：传递 None 给 process_partial ---
            // process_partial 的第一个参数是 Option<&[V]>，所以可以直接传 None
            let resampled_chunk = resampler.process_partial::<&[f32]>(None, None)?; // 显式指定 V 为 &[f32] (或省略让编译器推断)

            if let Some(channel_data) = resampled_chunk.into_iter().next() {
                if channel_data.is_empty() {
                    log::debug!("process_partial(None) 返回空块，停止冲刷。");
                    break;
                }
                log::trace!("冲刷得到输出块: {} 帧", channel_data.len());
                output_buffer_temp.extend_from_slice(&channel_data);
            } else {
                log::debug!("process_partial(None) 返回空 Vec<Vec>, 停止冲刷。");
                break;
            }
            log::trace!("冲刷后缓冲区长度: {}", output_buffer_temp.len());
        }

        // g) 提取最终结果 (跳过延迟部分)
        if output_buffer_temp.len() <= delay {
            log::warn!(
                "重采样输出长度 ({}) 小于或等于延迟 ({})，结果为空！",
                output_buffer_temp.len(),
                delay
            );
            vec![] // 返回空向量
        } else {
            // 截取从 `delay` 开始到 `delay + expected_output_len` 的部分
            let end_index = min(output_buffer_temp.len(), delay + expected_output_len);
            log::info!("重采样完成，最终提取输出样本数: {}", end_index - delay);
            output_buffer_temp[delay..end_index].to_vec() // 提取所需部分
        }
    } else {
        log::info!("音频采样率已经是 16kHz，无需重采样。");
        samples_mono // 直接返回原始单声道样本
    };

    // --- 4. 截断和最终检查 ---
    let target_samples = SAMPLES_PER_CHUNK;
    let mut final_samples_mut = final_samples; // 创建可变绑定
    if final_samples_mut.len() > target_samples {
        log::info!("音频长度超过 {} 秒，将截断。", MAX_AUDIO_LENGTH_SECS);
        final_samples_mut.truncate(target_samples);
    }
    if final_samples_mut.is_empty() {
        log::warn!("预处理后的最终音频样本为空！");
        // return Err(...) // 如果需要失败
    } else {
        log::info!(
            "预处理最终输出长度: {} 秒 ({} 样本)",
            final_samples_mut.len() as f32 / WHISPER_SAMPLING_RATE as f32,
            final_samples_mut.len()
        );
    }

    Ok(final_samples_mut)
}

// --- Whisper Decoder (改编自官方示例) ---

#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    // compression_ratio: f64, // 官方示例有，但计算复杂，暂略
}

struct Decoder<'a> {
    model: &'a mut Model, // 可变借用模型
    rng: rand::rngs::StdRng,
    task: Option<Task>,       // 任务类型
    timestamps: bool,         // 是否输出时间戳
    verbose: bool,            // 是否打印详细日志
    tokenizer: &'a Tokenizer, // 借用 tokenizer
    suppress_tokens: Tensor,  // 预计算的抑制 token 张量
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>, // 检测或指定的语言 token
    device: &'a Device,          // 借用设备
}

// 任务类型枚举 (来自官方示例)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Transcribe,
    Translate,
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, AppError> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| app_err!(ModelLoad, "Tokenizer 中找不到 token '{}'", token))
}

pub fn detect_language(
    model: &mut Model, // 使用我们的 Model enum
    tokenizer: &Tokenizer,
    mel: &Tensor,
) -> Result<u32, AppError> {
    // 返回 Result<_, AppError>
    let (_bsize, _, seq_len) = mel.dims3()?;
    // 限制输入长度以匹配模型预期
    let mel = mel.narrow(
        2,
        0,
        usize::min(seq_len, model.config().max_source_positions),
    )?;
    let device = mel.device();

    // 获取所有支持语言的 token IDs
    let language_token_ids = LANGUAGES // 使用导入的常量
        .iter()
        .map(|(t, _)| token_id(tokenizer, &format!("<|{}|>", t))) // 使用我们的 token_id helper
        .collect::<Result<Vec<_>, _>>()?; // 收集到 Result<Vec<u32>, AppError>

    let sot_token = token_id(tokenizer, SOT_TOKEN)?; // 使用常量
    let audio_features = model.encoder_forward(&mel, true)?; // 使用 Model::encoder_forward
    let tokens = Tensor::new(&[[sot_token]], device)?; // 创建初始 token 张量
    let language_token_ids_tensor = Tensor::new(language_token_ids.as_slice(), device)?; // 语言 tokens 转张量

    // 运行一次 Decoder 获取 logits
    let ys = model.decoder_forward(&tokens, &audio_features, true)?;
    let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?; // (vocab_size)

    // 在 logits 中只选择语言相关的 token
    let logits = logits.index_select(&language_token_ids_tensor, 0)?;
    let probs = candle_nn::ops::softmax(&logits, candle_core::D::Minus1)?; // 在最后一个维度 softmax
    let probs_vec = probs.to_vec1::<f32>()?; // 转换为 Vec<f32>

    // 找出概率最高的语言
    let mut probs_sorted = LANGUAGES.iter().zip(probs_vec.iter()).collect::<Vec<_>>();
    probs_sorted.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1)); // 按概率降序排序

    // 打印前 5 种语言及其概率 (用于调试)
    log::debug!("语言检测概率 (Top 5):");
    for ((code, language_name), p) in probs_sorted.iter().take(5) {
        log::debug!("  - {}({}): {:.4}", language_name, code, p);
    }

    // 获取概率最高的语言代码，并查找其 token id
    let best_lang_code = probs_sorted[0].0.0;
    let language_token = token_id(tokenizer, &format!("<|{}|>", best_lang_code))?;

    log::info!(
        "检测到最可能的语言: {} (token: {})",
        best_lang_code,
        language_token
    );
    Ok(language_token)
}

impl<'a> Decoder<'a> {
    // --- new 方法修正 ---
    fn new(
        model_components: &'a mut WhisperComponents,
        seed: u64,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
        // 不再接收 language 参数，它将在 run 中确定或使用
    ) -> Result<Self> {
        let model = &mut model_components.model;
        let tokenizer = &model_components.tokenizer;
        let device = &model_components.device;
        let config = &model_components.config;

        let no_timestamps_token = token_id(tokenizer, NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| {
                /* ... (逻辑不变) ... */
                if config.suppress_tokens.contains(&i) || (timestamps && i == no_timestamps_token) {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;

        let sot_token = token_id(tokenizer, SOT_TOKEN)?;
        let transcribe_token = token_id(tokenizer, TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(tokenizer, TRANSLATE_TOKEN)?;
        let eot_token = token_id(tokenizer, EOT_TOKEN)?;
        let no_speech_token = NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(tokenizer, token).ok())
            .ok_or_else(|| app_err!(ModelLoad, "找不到任何 non-speech token"))?;

        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token: None, // 初始化为 None，将在 run 中设置
            no_timestamps_token,
            device,
        })
    }

    fn predict_next_token(
        &mut self,
        tokens: &[u32],          // 当前序列
        audio_features: &Tensor, // Encoder 输出
        is_first_step: bool,     // 是否是第一步
        temperature: f64,        // 解码温度
    ) -> Result<(u32, f64), AppError> {
        // 返回 (next_token, logprob)
        let tokens_tensor = Tensor::new(tokens, self.device)?.unsqueeze(0)?;
        let decoder_output =
            self.model
                .decoder_forward(&tokens_tensor, audio_features, is_first_step)?;
        let (_, seq_len, _) = decoder_output.dims3()?;
        let logits = self
            .model
            .decoder_final_linear(
                &decoder_output.i((..1, seq_len - 1..))?, // 取最后一个时间步
            )?
            .i(0)?
            .i(0)?;
        let logits = logits.broadcast_add(&self.suppress_tokens)?;

        let next_token = if temperature > 0.0 {
            let prs = softmax(&(&logits / temperature)?, 0)?;
            let logits_v: Vec<f32> = prs.to_vec1()?;
            let distr = WeightedIndex::new(&logits_v)
                .map_err(|e| app_err!(Transcription, "创建采样分布失败: {}", e))?;
            distr.sample(&mut self.rng) as u32
        } else {
            logits.argmax(D::Minus1)?.to_scalar::<u32>()?
        };

        let probs = softmax(&logits, D::Minus1)?;
        let logprob = match probs.i(next_token as usize)?.to_scalar::<f32>()? as f64 {
            p if p > 0.0 => p.ln(),
            _ => f64::NEG_INFINITY,
        };

        Ok((next_token, logprob))
    }

    // --- run 方法 (处理分块和调用 decode_segment) ---
    // 改编自官方示例的 run 方法
    fn run(
        &mut self,
        audio_features: &Tensor, // 接收完整的 Encoder 输出
        language: Option<String>,
        temperature: f64, // 接收温度参数
    ) -> Result<String> {
        // --- 1. 确定语言 Token (只执行一次) ---
        self.language_token = match language {
            Some(lang) => Some(token_id(
                self.tokenizer,
                &format!("<|{}|>", lang.to_lowercase()),
            )?),
            None => {
                log::info!("未指定语言，进行自动检测...");
                // 需要 Encoder 输出进行检测
                let detected_lang_token =
                    detect_language(self.model, self.tokenizer, audio_features)?;
                log::info!("检测到语言 token: {}", detected_lang_token);
                Some(detected_lang_token)
            }
        };

        // --- 2. 初始化解码序列 ---
        let mut tokens = vec![self.sot_token];
        if let Some(lang_token) = self.language_token {
            tokens.push(lang_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }

        // --- 3. 解码主循环 ---
        let mut all_tokens = tokens.clone(); // 存储完整序列，包括初始 tokens
        let sample_len = self.model.config().max_target_positions; // 最大解码步数

        for i in 0..sample_len {
            // --- 关键修改：调用 predict_next_token ---
            let (next_token, _logprob) = self.predict_next_token(
                &tokens, // 传递当前已生成的序列 tokens
                audio_features,
                i == 0,      // 第一次迭代 flush=true
                temperature, // 传递温度
            )?;
            // --- 修改结束 ---

            tokens.push(next_token); // 添加到当前序列 (用于下一步预测)
            all_tokens.push(next_token); // 添加到完整序列

            if next_token == self.eot_token {
                log::debug!("解码在第 {} 步遇到 EOT token。", i + 1);
                break;
            }
        }
        if tokens.len() >= sample_len + (all_tokens.len() - tokens.len()) {
            // 稍微修正长度检查
            log::warn!("解码达到最大长度 {}，可能未正常结束。", sample_len);
        }

        // --- 4. 最终解码完整文本 ---
        let final_text = self
            .tokenizer
            .decode(&all_tokens, true)
            .map_err(AppError::from)?;

        if self.verbose {
            log::debug!("最终解码 Tokens: {:?}", all_tokens);
        }

        Ok(final_text)
    }
}

/// 公共接口：执行转录
pub async fn run_whisper(
    app_handle: tauri::AppHandle,
    audio_path: &Path,
    display_filename: &str,
    language: Option<String>,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    temperature: f64,
) -> Result<String, AppError> {
    log::info!(
        "开始处理文件 '{}', task: {:?}, language: {:?}, timestamps: {}",
        display_filename,
        task,
        language,
        timestamps
    );

    let components_mutex = load_whisper_components(&app_handle).await?;

    let audio_path_owned = audio_path.to_path_buf();
    log::info!("预处理音频...");
    let samples_f32 = tokio::task::spawn_blocking(move || preprocess_audio(&audio_path_owned))
        .await
        .map_err(|e| app_err!(Transcription, "音频预处理任务 join 失败: {}", e))??;
    log::info!("预处理完成，样本数: {}", samples_f32.len());

    if samples_f32.is_empty() {
        log::warn!("预处理后的音频样本为空，无法进行转录。");
        return Ok("".to_string());
    }

    log::info!("执行 Whisper 推理...");
    let transcription = tokio::task::spawn_blocking(move || {
        log::debug!("获取模型锁...");
        let mut components = components_mutex.blocking_lock();

        let whisper_components = &mut *components;
        log::debug!("模型锁已获取。");

        log::info!("重置模型 Decoder KV Cache...");
        whisper_components.model.reset_kv_cache();
        log::info!("模型状态已重置。");

        // --- 计算 Mel 特征 ---
        log::info!("计算 Mel 频谱特征...");
        let mel_vec: Vec<f32> = audio::pcm_to_mel(
            &whisper_components.config,
            &samples_f32,
            &whisper_components.mel_filters,
        );
        let n_mels = whisper_components.config.num_mel_bins;

        if mel_vec.is_empty() {
            return Err(app_err!(AudioPreprocessing, "Mel 特征向量为空"));
        }
        if mel_vec.len() % n_mels != 0 {
            return Err(app_err!(
                AudioPreprocessing,
                "Mel 特征向量长度 ({}) 不能被 num_mel_bins ({}) 整除",
                mel_vec.len(),
                n_mels
            ));
        }

        let num_frames = mel_vec.len() / n_mels;
        log::info!("计算得到帧数: {}", num_frames); // <-- 打印帧数

        let mel_shape = (1, n_mels, num_frames);

        let mel_tensor_cpu = Tensor::from_vec(mel_vec, mel_shape, &Device::Cpu)?;
        let mel_tensor = mel_tensor_cpu.to_device(&whisper_components.device)?;
        log::info!("特征 Tensor 创建完成，形状: {:?}", mel_tensor.shape());

        log::info!("运行 Encoder...");
        // 需要对 whisper_components.model (类型 Model) 进行可变借用
        let audio_features = whisper_components
            .model
            .encoder_forward(&mel_tensor, true)?; // flush=true
        log::info!(
            "Encoder 输出 (audio_features) 形状: {:?}",
            audio_features.shape()
        );

        // --- 解码 ---
        log::info!("创建解码器并运行...");
        let mut decoder = Decoder::new(
            whisper_components, // 传递整个可变结构体给 new
            42,
            task,
            timestamps,
            verbose,
        )?;

        // --- 关键修正：调用 run 时传入 audio_features ---
        log::info!("调用 decoder.run 处理 audio_features...");
        let final_text = decoder.run(&audio_features, language, temperature)?;

        log::info!("解码完成。最终文本长度: {}", final_text.len());
        log::debug!("最终拼接文本: '{}'", final_text);

        Ok::<_, AppError>(final_text)
    })
    .await
    .map_err(|e| app_err!(Transcription, "推理任务 join 失败: {}", e))??;

    log::info!("转录处理完成。");
    Ok(transcription)
}

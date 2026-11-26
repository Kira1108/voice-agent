# 帧（`app/frame/`）

组件之间只靠帧通信。帧分两类，走两条完全不同的路：

| | 数据帧（`base.py`） | 事件帧（`events.py`） |
|---|---|---|
| 基类 | `Frame` | `EventFrame` |
| 怎么走 | 沿 pipeline 一级一级往下传：上游 `push()`，进下游的队列 | 不走 pipeline：谁产生谁调 `controller.receive()`，controller 立刻广播 |
| 谁收到 | 只有紧挨着的下一个组件 | 所有在 `control_frames` 里声明了这个类型的组件（`isinstance` 匹配，子类也算） |
| 顺序 | 严格按顺序，排队处理 | 不排队，抢占式：打断、停止这类事要立刻生效 |
| 内容 | 音频、文本，要被处理的东西 | 状态变化：谁开口了、谁停了、该打断了 |

```
transport 输入 → [VAD] → STT → [轮次管理] → LLM → TTS → transport 输出      ← 数据帧，一条线往下
        ↑            ↑      ↑        ↑         ↑     ↑         ↑
        └────────────┴──────┴── SessionController（广播事件帧）──┴──→ 页面 / 客户端
```

## 贯穿全程的两个字段

所有数据帧都有这两个字段，从输入一直带到输出。事件帧大多也带 `utterance_id`。

| 字段 | 含义 | 谁设置 | 用在哪 |
|---|---|---|---|
| `utterance_id` | 用户一句话的编号。这句话引出的识别结果、回复文本、回复音频、各种事件都用它 | VAD（`VADAccumulator`）或流式 STT（`TencentSTTComponent`），从 1 开始递增 | 轮次管理判断事件是不是当前这一轮的；打断时指明打断哪一轮；对话历史按它对上回复；延迟统计按它记时间点 |
| `origin_ts` | 这句话说完的墙上时间（`time.time()`，秒） | 同上 | 日志里的"说完多久出第一个字"；回复的文本、音频沿用用户那句的值 |

几个特殊约定：

- **合并**：轮次管理把两句合成一轮时（`TextAudioFrame.merge`），取后一句的 `utterance_id` 和 `origin_ts`，因为用户是在后一句才真正说完。
- **同传**：`SimulTranslatorComponent` 攒了几个片段才翻译时，译文用最后一个片段的 id。
- **快答慢想**：主动报告结果的"通知轮"不是用户说的话，用负数 id（-1、-2…）。目前没有任何代码比较 id 大小，所以安全；以后如果加了"id 必须递增"的逻辑，要考虑这种情况。

## 数据帧

### 总览

| 帧 | 是什么 | 谁发出 | 谁处理 |
|---|---|---|---|
| `AudioFrame` | 一段 16 bit 单声道 PCM。三种用法：用户音频小块、VAD 攒好的一整句、bot 回复的音频块 | 用户音频：`WebRTCInput`、`TransportInput`（ws）<br>整句：`VADAccumulator`<br>回复：`Qwen3TTSComponent` / `VolTTSComponent` / `VolTTS2Component` | 小块：`VADAccumulator`、`TencentSTTComponent`<br>整句：`Qwen3STTComponent`<br>回复：`WebRTCOutput`、`TransportOutput`（ws） |
| `TextFrame` | 一段完整的文本 | 轮次管理（把 `TextAudioFrame` 去掉音频后往下传）；旧版 `TranslatorComponent` | LLM 组件当作用户的一句话；TTS 组件当作一整个回复 |
| `TextStreamFrame` | 流式回复的一段（通常是一句），用 `is_first` / `is_end` 标出回复的开头和结尾 | `NaiveAgentComponent`、`SimulTranslatorComponent`、`STFRComponent` | TTS 组件 |
| `InterimTextFrame` | 流式识别还没定稿的中间结果，后来的覆盖前面的 | `TencentSTTComponent` | 轮次管理交给 `on_interim_text`（`BargeInAggregator` 用它判断插话），不往下传 |
| `TextAudioFrame` | 定稿的识别结果，同时带着它对应的那段音频 | `Qwen3STTComponent`、`TencentSTTComponent` | 轮次管理（smart turn 要听音频判断说完没有）；翻译管线里没有轮次管理，直接进 `SimulTranslatorComponent`（它是 `TextFrame` 的子类） |

继承关系很关键，因为组件都用 `isinstance` 判断：

```
Frame
├── AudioFrame
├── TextFrame
│   ├── TextStreamFrame        所有收 TextFrame 的组件都会收它
│   └── TextAudioFrame         同上
└── InterimTextFrame           故意不是 TextFrame：中间结果不能被当成一句完整的话
```

### `AudioFrame`

| 字段 | 类型 | 含义 |
|---|---|---|
| `audio_data` | `bytes` | 16 bit 单声道 PCM。回复的结束帧可以是空的（见 `is_end`）。不打进日志 |
| `sample_rate` | `int \| None` | 采样率。WebSocket 输入不填（约定 16 kHz）；VAD、TTS 产出的都会填。transport 输出时，没有采样率的回复音频会被丢弃 |
| `is_first` | `bool` | 回复的第一块音频。transport 据此发 `BotStartedSpeakingFrame`、记首音延迟 |
| `is_end` | `bool` | 回复结束。TTS 合成完最后一句后**单独补一个空的结束帧**，因为合成时不知道哪一块是最后一块。transport 据此发 `ReplyEndFrame`、安排 `BotStoppedSpeakingFrame` |
| `audio_start` / `audio_end` | `float \| None` | 只用于用户语音：这段音频在本次会话输入流里的位置，单位是"音频秒数"（已收到的样本数 / 采样率），不是墙上时间。回复音频为 None |
| `text` | `str \| None` | 只用于回复音频：TTS 在每一句的**第一块**音频上带上这句的文字。transport 据此记录用户听到了哪些句子（`app/transport/spoken.py`），被打断时才能知道听到了多少 |

### `TextFrame` / `TextStreamFrame`

| 字段 | 类型 | 含义 |
|---|---|---|
| `text` | `str` | 文本 |
| `is_first` | `bool` | 仅 `TextStreamFrame`：回复的第一段 |
| `is_end` | `bool` | 仅 `TextStreamFrame`：回复的最后一段。LLM 生成完会补一个 `text=""`、`is_end=True` 的结束帧 |

TTS 组件读这两个标记时用 `getattr(frame, "is_first", True)`：普通 `TextFrame` 没有这两个字段，就被当成"一整个回复，既是开头也是结尾"。

### `InterimTextFrame`

| 字段 | 类型 | 含义 |
|---|---|---|
| `text` | `str` | 当前的中间识别结果 |
| `audio_start` / `audio_end` | `float \| None` | 和 `AudioFrame` 同一个音频时钟 |

### `TextAudioFrame`

| 字段 | 类型 | 含义 |
|---|---|---|
| `text` | `str` | 定稿的识别文本 |
| `audio_data` | `bytes` | 这句话的音频，16 kHz int16 PCM。合并后是两句音频直接拼起来，中间的停顿不在里面 |
| `sample_rate` | `int \| None` | 采样率 |
| `audio_start` / `audio_end` | `float \| None` | 这句话在输入流里的位置。整句识别时来自 VAD；流式识别时按服务端返回的时间戳算 |

| 方法 | 作用 |
|---|---|
| `duration` | `audio_data` 的秒数。合并过的帧会比 `audio_end - audio_start` 短，因为停顿没保留 |
| `from_audio_and_text(audio_frame, text)` | 整句识别：用 VAD 的整句 `AudioFrame` 加识别结果构造 |
| `merge(later)` | 把后一句接上来（smart turn 判断用户还没说完时用）。文本按语言拼接（中文直接接，英文中间加空格），id 和 `origin_ts` 取后一句 |
| `to_text_frame()` | 去掉音频，变成普通 `TextFrame` 交给 LLM |

## 事件帧

### 公共字段（`EventFrame`）

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | `str` | 随机 UUID |
| `event_type` | `str` | 事件类型，每个子类固定一个值。发给页面 / 客户端时，它们按这个字段分发 |
| `timestamp` | `int` | 创建时间，毫秒 |

### 总览

"订阅方"指在 `control_frames` 里声明了它的组件。所有 `PipelineComponent` 默认都订阅 `StopFrame`。

| 帧（`event_type`） | 什么时候发 | 谁发出 | 订阅方和效果 | 发给页面 / 客户端 |
|---|---|---|---|---|
| `StopFrame`（`stop`） | 会话结束：用户挂断、连接断开，或某个组件意外退出 | transport；`PipelineComponent` 的运行循环异常退出时自己报 | **所有组件**：取消正在处理的帧、丢掉积压、执行 `cleanup`，再把它推给下游；controller 标记会话结束 | 是（服务端结束时页面自动挂断） |
| `UserStartedSpeakingFrame`（`user_started_speaking`） | 听到疑似说话，比识别结果早得多 | `VADAccumulator`、`TencentSTTComponent` | 轮次管理（`SmartTurnKeywordInterruptAggregator`、`BargeInAggregator`）：用户还在说，暂停"等说完"的计时<br>`SimulTranslatorComponent`：取消停顿兜底计时<br>`STFRComponent`：有人在说话，不主动报告 | WebRTC 页面 |
| `UserStoppedSpeakingFrame`（`user_stopped_speaking`） | 用户停下了。`utterance_id=None` 表示误触发（没确认成一句话） | 同上 | 轮次管理：恢复计时<br>`SimulTranslatorComponent`：有积压的片段就开始停顿兜底计时<br>`STFRComponent`：误触发时重新看有没有结果要报告 | WebRTC 页面 |
| `BotStartedSpeakingFrame`（`bot_started_speaking`） | 回复的第一块音频开始播（WebRTC）或已发出（WebSocket） | transport | 轮次管理：bot 进入忙碌状态（这期间用户的话按插话处理） | WebRTC 页面 |
| `BotStoppedSpeakingFrame`（`bot_stopped_speaking`） | 回复播完（WebRTC 精确；WebSocket 按播放时长估算） | transport | 轮次管理：bot 空闲，可以处理下一轮<br>`STFRComponent`：自己的回复播完了，看有没有结果要主动报告 | WebRTC 页面 |
| `BotSpokeFrame`（`bot_spoke`） | 一轮回复结束（播完或被打断），**先于** `BotStoppedSpeakingFrame` | transport（由 `SpokenLedger` 记账） | `ConversationContext`：把用户实际听到的部分写进对话历史 | WebRTC 页面 |
| `InterruptFrame`（`interrupt`） | 用户插话打断，或说了"闭嘴" | 轮次管理（`KeywordInterruptAggregator` 及子类）；`STFRComponent`（用户开口时，打断自己正在播的通知轮） | LLM 组件、TTS 组件：取消当前这一轮、丢掉积压（`VolTTS2Component` 还会取消合成会话）<br>transport：清空没播的音频，按已播部分发 `BotSpokeFrame` | 是：客户端据此清空播放缓冲 |
| `ReplyEndFrame`（`reply_end`） | 回复的最后一块音频发出（WebSocket）或播完（WebRTC） | transport | 无（不经过 controller，只发给客户端） | 是 |
| `ConversationFrame`（`conversation`） | 对话历史每次变化：新的一轮、回复生成完、播放结果回来 | `ConversationContext` | 无组件订阅，只用于展示 | WebRTC 页面（"对话历史"栏） |
| `TasksFrame`（`tasks`） | 后台任务每次变化：委派、调工具、工具返回、查完、告诉用户、取消 | `STFRComponent` | 无组件订阅，只用于展示 | WebRTC 页面（"后台 THINKER"栏） |

WebSocket transport 只把 `StopFrame`、`InterruptFrame`、`ReplyEndFrame` 和回复音频发给客户端，不转发 `ConversationFrame`、`TasksFrame` 和用户开口 / 停口事件。

### 各事件的专有字段

| 帧 | 字段 | 类型 | 含义 |
|---|---|---|---|
| `ReplyEndFrame`、`BotStartedSpeakingFrame`、`BotStoppedSpeakingFrame` | `utterance_id` | `int \| None` | 这个回复回答的是哪一句 |
| `InterruptFrame` | `utterance_id` | `int \| None` | 被打断的是哪一句的回复 |
| `UserStartedSpeakingFrame` | `utterance_id` | `int \| None` | 如果被确认成一句话，它将得到的 id（VAD 是先报开口、后确认） |
| `UserStoppedSpeakingFrame` | `utterance_id` | `int \| None` | 结束的那句话；`None` 表示误触发 |
| `BotSpokeFrame` | `utterance_id` | `int \| None` | 这个回复回答的是哪一句 |
| | `sentences` | `list[(str, float)] \| None` | 回复的每一句和它被播出的比例，1.0 是整句播完，0.0 是一点没播。`None`：transport 不知道播的是哪段文字（TTS 没在音频上标句子，比如 `VolTTS2`）；`[]`：一点都没播 |
| | `interrupted` | `bool` | 是否被打断 |
| `ConversationFrame` | `turns` | `list[dict]` | 完整对话历史，旧的在前。每轮的字段见 `ConversationContext.snapshot()`：`utterance_id`、`user`、`said`（下一轮发给 LLM 的回复）、`heard`、`generated`、`reported`、`interrupted` |
| `TasksFrame` | `tasks` | `list[dict]` | 本次会话所有后台任务，旧的在前。每个任务的字段见 `Task.snapshot()`：`id`、`instruction`、`status`、`result`、`error`、`reported`、`started`、`finished`、`steps` |

## 时序约定

改动任何发帧或收帧的代码前，先确认没破坏这几条：

1. **开口事件先于文本**：`UserStartedSpeakingFrame` 远早于这句话的识别结果；中间可能只有一个 `UserStoppedSpeakingFrame(utterance_id=None)`，没有任何文本（误触发）。
2. **一轮回复的边界靠标记，不靠计时**：文本层是 `TextStreamFrame` 的 `is_first` / `is_end`，音频层是 `AudioFrame` 的 `is_first` / `is_end`。结束标记一定会到，哪怕没有任何可读的文字（TTS 跳过纯标点时也会补一个空帧）。
3. **`BotSpokeFrame` 先于 `BotStoppedSpeakingFrame`**：对话历史要先知道用户听到了什么，bot 才算空闲，下一轮的 LLM 才能读到正确的历史。
4. **打断只作用于一轮**：`InterruptFrame` 让各组件取消当前这一轮，组件本身继续运行；`StopFrame` 才是整个会话结束。`STFRComponent` 的后台任务不受打断影响。
5. **句子文字只在每句第一块音频上**：transport 靠它把音频和文字对上，TTS 不能把它放到别的块上，也不能省略。

## 不是帧的东西

- **延迟报告**：`app/metrics/latency.py` 发给 WebRTC 页面的 `{"event_type": "latency", ...}` 是普通 dict，不是 `EventFrame`，不经过 controller 广播。
- **组件内部消息**：`SimulTranslatorComponent` 的 `_Pause`、`STFRComponent` 的 `_Report`，是组件往**自己的**队列里塞的提醒（"停顿到时间了""有结果可以报告了"），不会出组件。

## 新增或修改帧时

- **新数据帧**：继承 `TextFrame` 就会被所有收文本的组件（LLM、TTS）当成文本处理。不想这样，就像 `InterimTextFrame` 一样直接继承 `Frame`，再让需要它的组件显式处理。
- **新事件帧**：订阅它的组件要把它加进 `control_frames`。要显示在 WebRTC 页面上，还要加进 `WebRTCOutput` 的 `control_frames` 和转发列表，页面的 `EVENTS` 里也要加对应的处理函数。
- **改字段**：事件帧会被序列化成 JSON 发给页面和客户端（`model_dump()`），改字段名要同步改 `app/static/webrtc.html` 和 WebSocket 客户端。

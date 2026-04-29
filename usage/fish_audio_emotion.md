# Fish Audio TTS 接入与 S2-Pro 情绪标注

## 1. 参考文档

| 文档 | URL |
|------|-----|
| TTS API（请求格式、参数说明） | https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech |
| 情绪标签完整列表（S1 / S2-Pro 语法对比） | https://docs.fish.audio/api-reference/emotion-reference |
| 模型概览（S1 vs S2-Pro 能力矩阵） | https://docs.fish.audio/developer-guide/models-pricing/models-overview |
| 核心功能：情绪控制 | https://docs.fish.audio/developer-guide/core-features/emotions |

---

## 2. 整体思路

### 2.1 Fish Audio 接入

Fish Audio 是一个 REST TTS 服务，接口极简：

```
POST https://api.fish.audio/v1/tts
Header: Authorization: Bearer <FISH_AUDIO_API_KEY>
Header: model: s2-pro          ← 模型选择在 Header 里，不在 Body
Body (JSON): { "text": "...", "reference_id": "<voice_id>", ... }
```

几个关键点：

- **模型选择在 HTTP Header**，字段名是 `model`，不在请求 body 里
- **说话人声音**通过 `reference_id` 指定（在 fish.audio 平台上传/克隆的声音模型 ID）
- **语速**不是顶层字段，而是嵌套在 `prosody` 对象里：`"prosody": {"speed": 1.2}`
  > ⚠️ 这是一个容易犯错的地方，写成顶层 `"speed": 1.2` 不报 4xx，但会导致服务端 reset 连接（ConnectionResetError）

### 2.2 情绪标注的设计决策

**问题**：Fish Audio S2-Pro 支持在文本中用 `[bracket]` 语法嵌入自然语言情绪描述，但情绪是 TTS provider 专属的，不同 provider 语法不同。

**两种可选方案**：

| 方案 | 做法 | 缺点 |
|------|------|------|
| A：生成 transcript 时同步标注 | LLM 生成文案时就写入 `[bracket]` | transcript 被污染，换 provider 必须剥离；需要提前锁定 TTS 引擎 |
| **B：TTS 层独立做一次标注 pass** | 在将文本送往 TTS 之前，对完整 transcript 做一次 LLM 调用插入情绪标签 | 增加一次 LLM 调用延迟（可接受） |

**选方案 B**，原因：
1. transcript 保持 TTS 无关，可复用
2. 标注逻辑完全封装在 FishAudio provider 内部，其他 provider 不受影响
3. 完整 transcript 上下文更好，LLM 能看到整段对话的情绪弧线，标注更连贯

### 2.3 S1 vs S2-Pro 语法差异

| 模型 | 语法 | 说明 |
|------|------|------|
| S1 | `(happy)` `(sad)` `(whispering)` | 64+ 固定标签，括号语法，只能用预定义情绪 |
| **S2-Pro** | `[feel happy and excited]` `[calm, thoughtful]` | **自由自然语言**，无固定列表限制，对 LLM 更友好 |

本文档只涉及 S2-Pro。

---

## 3. 核心代码

### 3.1 API 请求结构（`fishaudio.py`）

```python
headers = {
    "Authorization": f"Bearer {self.api_key}",
    "Content-Type": "application/json",
    "model": effective_model,          # 模型在 Header 里
}
payload = {
    "text": text,
    "format": "mp3",
    "mp3_bitrate": 128,
    "normalize": True,
    "latency": "normal",
    "reference_id": voice,             # 声音模型 ID
    "prosody": {"speed": 1.2},         # ⚠️ speed 必须在 prosody 内
}
```

### 3.2 重试逻辑

Fish Audio 偶尔会出现连接 reset，加了指数退避重试：

```python
for attempt in range(1, 4):
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.content
    except requests.HTTPError:
        raise   # 4xx/5xx 不重试，直接暴露 API 错误
    except (requests.ConnectionError, requests.Timeout) as e:
        if attempt < 3:
            time.sleep(2 ** attempt)   # 2s → 4s
```

### 3.3 情绪标注 Prompt

```python
_ANNOTATION_SYSTEM_PROMPT = """\
You are an emotion annotation assistant for a podcast TTS engine using Fish Audio S2-Pro.

S2-Pro supports natural-language emotion cues in square brackets, e.g.:
  [feel excited and energetic] Welcome to today's episode!
  [calm and thoughtful] That is a really interesting point.

Rules:
1. Add one [emotion bracket] at the START of each Person's turn.
2. For long turns where tone shifts mid-way, you may insert one more bracket inline — sparingly.
3. Bracket content must be ≤8 words.
4. Do NOT change any dialogue text — only insert brackets.
5. Keep ALL <Person1>...</Person1> and <Person2>...</Person2> tags exactly as-is.
6. Return ONLY the annotated transcript, no explanation, no code fences.
"""
```

### 3.4 `preprocess_transcript()` 方法

```python
def preprocess_transcript(self, text: str) -> str:
    if not self._annotation_enabled():
        return text

    model_id = f"openai/{annotation_model}" if llm_base_url else annotation_model

    response = litellm.completion(
        model=model_id,
        messages=[
            {"role": "system", "content": _ANNOTATION_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Annotate:\n\n{text}"},
        ],
        temperature=0.4,
        api_base=llm_base_url,   # 走聚合站时传入
        api_key=llm_api_key,
    )
    annotated = response.choices[0].message.content.strip()

    # Sanity-check：标注后必须还包含 Person 标签
    if "<Person1>" not in annotated or "<Person2>" not in annotated:
        return text   # 回退原文

    return annotated
```

---

## 4. 情绪标注的完整流程

```
用户请求 /generate
        │
        ▼
[LLM] content_generator.py
  生成干净的 transcript（只含 <Person1>/<Person2> 标签，无情绪）
  保存到 data/transcripts/
        │
        ▼
text_to_speech.py: convert_to_speech(text)
  cleaned_text = self.provider.preprocess_transcript(text)
        │
        │  ← FishAudioTTS.preprocess_transcript()
        │     1. 读取配置：emotion_annotation / annotation_model
        │     2. 用 litellm 发起一次 LLM 调用
        │     3. 输入：完整 transcript
        │     4. 输出：每个 Person turn 前加了 [emotion bracket] 的版本
        │     5. Sanity-check，失败则回退原文
        │
        ▼
  provider.split_qa(cleaned_text)
  → 按 <Person1>/<Person2> 拆分成 (question, answer) 对
  → 每个 turn 的文本头部已含情绪标签，如：
      "[feel excited and enthusiastic] So today we're talking about..."
        │
        ▼
  FishAudioTTS.generate_audio(turn_text, voice, model)
  → POST /v1/tts，S2-Pro 识别 [bracket] 并以对应情绪合成语音
        │
        ▼
  合并所有 segment → 输出 MP3
```

### 4.1 配置项（`conversation_config.yaml`）

```yaml
text_to_speech:
  fishaudio:
    model: "s2-pro"                      # 必须是 s2-pro，S1 不支持自由语法
    emotion_annotation: true             # false 可关闭标注
    annotation_model: "gemini-3-flash-preview"   # 标注用的 LLM
    default_voices:
      question: "<voice_model_id>"
      answer:   "<voice_model_id>"
    voice_speeds:
      <voice_model_id>: 1.2              # 可选，per-voice 语速
```

### 4.2 环境变量

```bash
FISH_AUDIO_API_KEY=<your_key>   # fish.audio 平台获取
LLM_BASE_URL=https://...        # 可选，走 OpenAI-compatible 聚合站
LLM_API_KEY=sk-...              # 聚合站 key
```

### 4.3 迁移到其他项目的 Checklist

- [ ] 实现 `preprocess_transcript(text: str) -> str` 方法（或等价的预处理函数）
- [ ] 在实际调用 TTS API **之前**触发该方法，传入完整对话文本
- [ ] 确保 `model` 参数设为 `s2-pro`（Header，不是 Body）
- [ ] `speed` 放在 `prosody.speed`，不要放顶层
- [ ] Sanity-check 标注结果（防止 LLM 删改原文或丢失结构标签）
- [ ] 标注失败时 fallback 原文，不阻断 TTS 流程

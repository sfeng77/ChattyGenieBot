# Agent Benchmark Harness

对 ChattyGenieBot 的 agent 层做确定性评测:mock 工具 + 固定任务集 + 多次重复。

## 安装位置
把整个 `bench/` 文件夹放到 repo 根目录(和 `app/` 同级)。无额外依赖。

## 运行
```bash
# 用 .env 里的默认模型(本地 gpt-oss:20b)
python -m bench.runner --tasks bench/tasks.json --repeats 5

# 换强模型跑同一套任务,回答"模型还是系统"的问题
python -m bench.runner --tasks bench/tasks.json --repeats 5 --model <strong-model>

# 只跑某一类
python -m bench.runner --filter negative

# gpt-oss 的 reasoning(think)开关 A/B,不改 .env
python -m bench.runner --repeats 5 --think on
python -m bench.runner --repeats 5 --think off
```

## 工作原理
- 复用你真实的 `.env` 配置(同模型、同工具开关、同 system prompt),
  但把 sessions/history 两个 SQLite 重定向到临时目录,并关闭 history pruning。
- 每个 (task, repeat) 用独立 chat_id => 干净 session,任务间零污染。
- Mock 层不重新定义工具,而是 `dataclasses.replace()` 克隆真实 FunctionTool、
  只替换执行体:模型看到的 name/description/schema 与生产完全一致。
- `set_reminder` 在 bot.py 里创建(依赖 Telegram JobQueue),裸 runtime 没有,
  harness 按原 schema 重建了一个 mock 版。

## 评分维度(分层报告)
- tool_selection:该调的都调了,禁止的没调
- params:参数子集匹配(字符串大小写不敏感,值可给候选列表如 `[14, "14"]`)
- no_extra_calls:没有多余调用(search_memory 默认豁免,可用 allowed_tools 覆盖)
- responded:最终有非空回复
- content:`response_must_contain` / `response_must_not_contain` 断言(见下)
- 另报 flaky tasks(时过时不过 => 方差)和 always-failing tasks(稳定失败 => 系统问题优先查)

## 任务 schema
```json
{
  "id": "select_01", "category": "tool_selection",
  "input": "用户消息",
  "expected_tools": ["stock_trend"],
  "forbidden_tools": ["web_search"],
  "allowed_tools": ["search_memory"],
  "expected_params": {"stock_trend": {"symbol": "NVDA", "days": [14, "14"]}},
  "mock_responses": {"stock_trend": {"error": "..."} },
  "response_must_contain": ["some substring", ["alt A", "alt B"]],
  "response_must_not_contain": ["stale phrase"],
  "seed_history": {"generate": {"topics": ["旅行"], "pairs_per_topic": 10}}
}
```
`mock_responses` 里给 `error` 注入失败,给 `response` 覆盖默认返回。

### response_must_contain / response_must_not_contain
两者都做大小写不敏感的子串匹配,作用在最终回复上:
- `response_must_contain`:列表内所有元素都必须命中(AND)。元素本身也可以是
  一个列表,表示"任一命中即可"(OR)——例如
  `[["7月2日", "2026-07-02"]]` 表示回复至少要包含这两种日期格式之一。
- `response_must_not_contain`:列表里的任何字符串都不能出现,命中即判定
  `content_ok=False`,详情写入 `detail.content_violations`。
- 没有这两个字段的任务(旧任务)行为完全不变,`content_ok` 恒为 `True`。

### 动态日期占位符
`tasks.json` 是静态文件,但"今天"会变。`runner.py` 在加载任务后、运行前会
对每个任务的所有字符串字段(含嵌套的 list/dict,比如 `mock_responses` 和
`response_must_contain` 里的 OR 列表)做占位符替换,时区取自
`AGENT_TIMEZONE`(与 agent 注入当前时间用的是同一个设置):
- `{{TODAY}}`:`YYYY-MM-DD`
- `{{TODAY_CN}}`:`M月D日`
- `{{NEXT_WEDNESDAY}}` / `{{NEXT_WEDNESDAY_CN}}`:下周三(计算方式是"下一个自然周"的周三,而不是最近的周三——如果今天恰好是周三或更早,也不会返回本周的周三)

### temporal_grounding 分类
验证 agent 是否真的用上了系统提示里注入的当前时间(而不是猜测或用搜索结果里
的旧日期):
- `temporal_01_weather_stale_vs_fresh`:mock 的 `web_search` 同时返回一条
  过期结果(3月6日)和一条当前日期的结果,断言回复里不出现过期日期。
- `temporal_02_what_day_is_it`:问"今天几号",要求不调用 `web_search`,直接
  从注入的当前时间回答。
- `temporal_03_relative_date_math`:问"下周三是几号",要求算对相对当前日期
  的日期。

预期结果(不在代码里做硬断言,靠人读报告判断):在加上"注入当前时间"这个
修复之前,这三个任务应该失败;修复之后,足够强的模型应该能通过。本地小模型
的结果可能会有波动——这种波动本身就是这个 benchmark 想暴露的问题。

### --think 开关
`OPENAI_THINK_ENABLED`(默认 `False`,对应之前硬编码的 `extra_body={"think": False}`)
控制 gpt-oss 的 reasoning。`--think on` / `--think off` 临时覆盖当前运行的这个设置,
不动 `.env`,机制和 `--model` 一样(`Settings.model_copy(update=...)`)。报告表头和
结果 JSON 的 `summary.think_enabled` / `summary.model` 都会显示这次跑的实际取值,
`summary.overall.mean_elapsed_seconds` 报告平均单次耗时(think=on 预期会更慢,用这个
数字量化)。

### seed_history / long_context 分类
用于衡量"上下文很长时,agent 还能不能正确选工具、不跑题"——在跑任务输入
之前,先往这个 chat 的 session 里塞一堆无关的历史消息。两种写法:
- Form A(手写对话):`{"turns": [["用户说的话", "助手回的话"], ...]}`,
  按顺序原样注入为交替的 user/assistant items。
- Form B(本地生成,不调用模型):
  `{"generate": {"topics": ["电子游戏", "旅行", "摄影"], "pairs_per_topic": 14}}`,
  为每个 topic 用固定模板生成 `pairs_per_topic` 轮问答填充,达到目标轮数
  (3 个 topic × 14 轮 ≈ 40 轮真实用户不会关心的历史)。

`long_context` 分类里的任务是 `tool_selection` / `negative` / `param_accuracy`
里 4 个已有任务的"加长版"(id 加 `_seeded` 后缀,期望值完全不变),只是多了
`seed_history`。种子消息在分配好 chat_id、跑真正的任务输入之前,通过
`runtime._get_session(chat_id)`(私有访问,和 `runtime._agent.tools` 同样的
约定)直接 `add_items` 进去——harness 本来就禁用了 pruning,所以这里测的是
"上下文本身很长"这件事,不是测 pruning 触发后的行为。把 `long_context` 的
结果和它们的未加长版对照,就能量化"上下文变长导致的选工具/参数准确率下降"。

## 已知取舍
- runner 直接访问 `runtime._agent.tools`(私有属性)。想干净一点,给
  `AgentRuntime.__init__` 加个 `tools_override: list | None = None` 参数即可。
- error_recovery 类任务的"优雅恢复"目前只检查 responded;更严格可以加
  关键词断言或 LLM judge,建议先看几个真实输出再定标准。

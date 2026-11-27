# Rolling Buffer 问答试用版

GitHub Pages 保持静态，Cloudflare Worker 代表博客调用 DeepSeek Chat Completions。第一版要求独立的随机试用码，不是读者登录，也不是公开匿名聊天。没有设置后端 URL 时，页面不生成入口。文章内容不会自动上传；只在访客提交时，按当时的勾选状态发送问题和最多 4,500 字符的本页公开摘录。不扫描审稿库，不保存对话，不提供全站检索或多轮记忆。不启用 Worker 请求体日志。

## 启用

1. 在你自己的 Cloudflare 账号部署此目录，保留配置中的 SQLite Durable Object 迁移。安装并运行 Wrangler 前请查阅 [Cloudflare 官方入门](https://developers.cloudflare.com/workers/get-started/guide/)。运行 `npx wrangler deploy`；开通合约、费用或账户授权由站点所有者决定。
2. 在 Cloudflare 仪表盘的 Worker Secrets 内，分别配置 `DEEPSEEK_API_KEY` 和 `ASSISTANT_ACCESS_CODE`。试用码使用独立且不可猜的 32–128 字符（例如本地运行 `openssl rand -hex 24`），只分发给获准试用的人。不要把任何密钥或试用码写到 GitHub、站点配置、前端或聊天。也可用 Wrangler 的交互式 `secret put`，不要将值作为命令行参数。
3. 在博客 `_config.yml` 加 **公开** 地址：`rolling_buffer_assistant_endpoint: "https://<worker>.workers.dev/ask"`。如果站点换域名，在 Worker 把 `BLOG_ORIGIN` 改为对应 HTTPS origin。先验证正确试用码、拒绝错误试用码和配额，再将地址合入站点。删除该设置即可隐藏入口，删除 Worker 的 DeepSeek secret 则停止真实调用。

默认每 UTC 日最多 30 次上游请求尝试（含超时/失败），每次最多输出 700 token；输入、请求正文和超时另有上限。请求配额是共享硬限制，不是账单金额承诺；部署平台自身也有用量限制。拟开放匿名使用时，先建立反滥用验证和独立的预算控制，不要仅删除试用码：Origin 检查不是鉴权。密钥只被服务端固定发送至 DeepSeek 官方 HTTPS API；前端以文本显示回答，不解释为 HTML。

## 无凭据测试

Node.js 24 或更新版本：`node --test _assistant_backend/*.test.mjs`。测试使用内存 SQLite 和替代上游，没有发送真实模型请求。Cloudflare 部署和实际 DeepSeek 调用需另行验证。

资料：[DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/) · [Cloudflare Secrets](https://developers.cloudflare.com/workers/configuration/secrets/) · [SQLite Durable Objects](https://developers.cloudflare.com/durable-objects/platform/storage-options/)

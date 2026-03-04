# skills

本仓库用于维护可迁移的本地 skills 源码。

## Repository Layout

- `skills/`：所有 skill 源码目录
- `skills/sora-watermark-lite/`：本地视频去水印（LAMA + 批处理）
- `skills/nano-banana-pro/`：Gemini 图像生成/编辑（支持自定义 API 网关）

## Install Skills

在仓库目录内安装单个 skill：

```bash
npx skills add . --skill sora-watermark-lite -g -y
npx skills add . --skill nano-banana-pro -g -y
```

从 GitHub 安装：

```bash
npx skills add "git@github.com:gaowei-space/skills.git" --skill sora-watermark-lite -g -y
npx skills add "git@github.com:gaowei-space/skills.git" --skill nano-banana-pro -g -y
```

## Runtime Notes

- `nano-banana-pro` 依赖 `uv` 与 `GEMINI_API_KEY`
- 可选三方网关环境变量：`GEMINI_BASE_URL`、`GEMINI_API_VERSION`
- `sora-watermark-lite` 建议在 Apple Silicon 上使用 `PYTORCH_ENABLE_MPS_FALLBACK=1`

## Conventions

- 本仓库仅保存源码，不提交 `.skill` 打包产物
- 每个 skill 独立目录管理，便于后续扩展更多 skills

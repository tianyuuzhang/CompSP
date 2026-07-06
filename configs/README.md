# configs

这里存放可复现的命令示例和非敏感配置。

Secrets are intentionally excluded. Set API credentials with environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

For non-OpenAI-compatible providers, set separate env vars and pass them to `run_ofa.py` using the `--*-api-key-env` and `--*-base-url-env` options.

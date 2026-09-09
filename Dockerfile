# CloudBase 云托管 专用镜像（跑 uvicorn 服务）
# 知识库在启动时从 COS 拉取（RUN_ENV=scf 触发 cos_sync）
FROM python:3.10-slim

WORKDIR /app

# 编译 crcmod(COS SDK 依赖) 需要 build-essential；curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# 应用代码（app/ 即可，SCF 的 scf_handler.py 不需要）
COPY app ./app

ENV RUN_ENV=scf

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

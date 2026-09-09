# -*- coding: utf-8 -*-
"""冷启动数据准备：SCF 无持久盘，代码目录只读。

- 从 COS 把知识库 chunks 下载到 KNOWLEDGE_DIR（/tmp 下）
- 确保 sqlite 所在目录存在
仅在 RUN_ENV=scf 时由 main.on_startup 调用；本地开发自动跳过。
"""
import os
import re
import logging
from pathlib import Path

logger = logging.getLogger("cos_sync")

_KB_PREFIX_DEFAULT = "chunks/"


def _cos_client():
    from qcloud_cos import CosConfig, CosS3Client

    cfg = CosConfig(
        Region=os.environ["COS_REGION"],
        SecretId=os.environ["COS_SECRET_ID"],
        SecretKey=os.environ["COS_SECRET_KEY"],
        Token=os.environ.get("COS_TOKEN") or None,
    )
    return CosS3Client(cfg)


def ensure_dirs() -> None:
    """确保知识库与 sqlite 的父目录存在（SCF /tmp 下）"""
    from app.config import settings

    Path(settings.KNOWLEDGE_DIR).mkdir(parents=True, exist_ok=True)

    m = re.match(r"^sqlite:///(.+)$", settings.DATABASE_URL)
    if m:
        db_path = m.group(1)
        if db_path and not db_path.startswith(":memory:"):
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def ensure_knowledge_dir() -> Path:
    """KNOWLEDGE_DIR 为空时从 COS 拉取知识库，返回其路径"""
    from app.config import settings

    dst = Path(settings.KNOWLEDGE_DIR)
    if any(dst.rglob("*.md")):
        return dst

    bucket = os.environ.get("COS_BUCKET")
    if not bucket:
        # 本地开发：无 COS 配置且目录为空 -> 直接返回，由上层自行处理空知识库
        logger.info("未配置 COS_BUCKET，跳过知识库同步（本地模式）")
        return dst

    prefix = os.environ.get("COS_KNOWLEDGE_PREFIX", _KB_PREFIX_DEFAULT).strip("/") + "/"
    client = _cos_client()
    dst.mkdir(parents=True, exist_ok=True)

    marker, total = "", 0
    while True:
        resp = client.list_objects(Bucket=bucket, Prefix=prefix, Marker=marker)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix):]
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(Bucket=bucket, Key=key, DestFilePath=str(target))
            total += 1
        if str(resp.get("IsTruncated")) != "true":
            break
        marker = resp.get("NextMarker")

    logger.info("已从 COS(%s/%s) 同步 %s 个文件到 %s", bucket, prefix, total, dst)
    return dst

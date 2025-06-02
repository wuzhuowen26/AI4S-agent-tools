#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

TOOL_NAME = sys.argv[1] if len(sys.argv) > 1 else None
BASE_TAG = ""          # 可设置为 "local/base:0.1"
SAVE_TAR = False       # 若为 True，则导出 .tar

if TOOL_NAME is None:
    print("❌ 用法: python build_one_tool.py <tool_name>")
    sys.exit(1)

TOOL_DIR = Path("servers") / TOOL_NAME
TEMPLATE_PATH = Path("TEMPLATE.Dockerfile")
DOCKERFILE_PATH = TOOL_DIR / "Dockerfile"


if not TOOL_DIR.is_dir():
    print(f"❌ 目录不存在: {TOOL_DIR}")
    sys.exit(1)

if not TEMPLATE_PATH.exists():
    print(f"❌ 缺少模板: {TEMPLATE_PATH}")
    sys.exit(1)

TPL = TEMPLATE_PATH.read_text()

def sh(cmd, **kw):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True, **kw)

if not DOCKERFILE_PATH.exists():
    print(f"   ✨ 自动生成 Dockerfile: {DOCKERFILE_PATH}")
    dockerfile = TPL.replace("ARG BASE_IMG=python:3.12-slim",
                             f"ARG BASE_IMG={BASE_TAG or 'python:3.12-slim'}")
    DOCKERFILE_PATH.write_text(dockerfile)

img_tag = f"{TOOL_NAME.lower().replace('_', '-')}:latest"

build_arg = f"--build-arg BASE_IMG={BASE_TAG}" if BASE_TAG else ""
sh(f"docker build -f {DOCKERFILE_PATH} {build_arg} -t {img_tag} {TOOL_DIR}")

if SAVE_TAR:
    tar_path = f"{TOOL_NAME}.tar"
    sh(f"docker save {img_tag} -o {tar_path}")
    print(f"    💾 镜像已保存到: {tar_path}")

print(f"✅ 工具 {TOOL_NAME} 镜像构建完成")
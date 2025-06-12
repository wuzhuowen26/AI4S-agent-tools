#!/usr/bin/env python3
import sys
import subprocess
import time
import argparse
from pathlib import Path

TOOL_NAME = sys.argv[1] if len(sys.argv) > 1 else None
BASE_TAG = ""        
SAVE_TAR = False     
IMAGE_PREFIX = "registry.dp.tech/deepmodeling/mcp"  

if TOOL_NAME is None:
    print("❌ 用法: python build_one_tool.py <tool_name>")
    sys.exit(1)

TOOL_DIR = Path("servers") / TOOL_NAME
TEMPLATE_PATH = Path("Dockerfile.tmpl")
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
    dockerfile = TPL.replace("ARG BASE_IMG=registry.dp.tech/deepmodeling/python:3.12-slim-bullseye",
                             f"ARG BASE_IMG={BASE_TAG or 'registry.dp.tech/deepmodeling/python:3.12-slim-bullseye'}")
    DOCKERFILE_PATH.write_text(dockerfile)

img_tag = f"{IMAGE_PREFIX}/{TOOL_NAME.lower().replace('_', '-')}:latest"

def parse_args():
    parser = argparse.ArgumentParser(description='构建并测试Docker镜像')
    parser.add_argument('tool_name', help='工具名称')
    parser.add_argument('--run-test', action='store_true', help='构建后运行测试')
    parser.add_argument('--push', action='store_true', help='构建后推送镜像')
    parser.add_argument('--port', type=int, default=50001, help='服务端口号')
    parser.add_argument('--health-path', default='/health', help='健康检查路径')
    return parser.parse_args()

def wait_for_container(container_name, port, health_path, max_retries=30):
    """等待容器启动并尝试访问健康检查接口"""
    for i in range(max_retries):
        try:
            # 首先检查容器是否在运行
            status = subprocess.check_output(
                f"docker inspect -f '{{{{.State.Status}}}}' {container_name}",
                shell=True, text=True
            ).strip()
            
            if status != "running":
                print(f"容器状态: {status}")
                time.sleep(1)
                continue

            # 尝试访问健康检查接口
            sh(f"curl -s http://localhost:{port}{health_path}", capture_output=True)
            return True
        except subprocess.CalledProcessError:
            if i < max_retries - 1:
                time.sleep(1)
            continue
    return False

def run_container_test(img_tag, port, health_path):
    """运行容器测试"""
    container_name = f"test-{TOOL_NAME.lower()}"
    try:
        print(f"🚀 启动测试容器: {container_name}")
        sh(f"docker run -d --name {container_name} -p {port}:{port} {img_tag}")
        
        if wait_for_container(container_name, port, health_path):
            print("✅ 健康检查通过")
        else:
            print("❌ 健康检查失败")
            # 显示容器日志以帮助诊断
            print("\n📝 容器日志:")
            sh(f"docker logs {container_name}")
            
    finally:
        print(f"🧹 清理测试容器: {container_name}")
        sh(f"docker rm -f {container_name}", capture_output=True)

def main():
    args = parse_args()
    global TOOL_NAME
    TOOL_NAME = args.tool_name

    build_arg = f"--build-arg BASE_IMG={BASE_TAG}" if BASE_TAG else ""
    
def main():
    args = parse_args()
    global TOOL_NAME
    TOOL_NAME = args.tool_name

    build_arg = f"--build-arg BASE_IMG={BASE_TAG}" if BASE_TAG else ""
    
    if args.push:
        sh(f"docker buildx build --no-cache --platform=linux/amd64 -f {DOCKERFILE_PATH} {build_arg} -t {img_tag} --push {TOOL_DIR}")
    else:
        sh(f"docker buildx build --no-cache --platform=linux/amd64 -f {DOCKERFILE_PATH} {build_arg} -t {img_tag} {TOOL_DIR}")

    if SAVE_TAR:
        tar_path = f"{TOOL_NAME}.tar"
        sh(f"docker save {img_tag} -o {tar_path}")
        print(f"    💾 镜像已保存到: {tar_path}")

    print(f"✅ 工具 {TOOL_NAME} 镜像构建完成")

    if args.run_test:
        run_container_test(img_tag, args.port, args.health_path)

if __name__ == "__main__":
    main()
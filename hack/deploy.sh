#!/bin/bash

# AI4S Tools 部署脚本
# 用法: ./hack/deploy.sh <tool_name> [port]

set -e

if [ $# -eq 0 ]; then
    echo "❌ 用法: $0 <tool_name> [port]"
    echo "例如: $0 Paper_Search 50001"
    exit 1
fi

TOOL_NAME="$1"
PORT="${2:-50001}"

echo "🚀 开始部署工具: $TOOL_NAME"
echo "📊 使用端口: $PORT"

# 步骤1: 构建Docker镜像
echo ""
echo "🔨 步骤1: 构建Docker镜像..."
python hack/build_docker.py "$TOOL_NAME"

# 步骤2: 生成k8s配置
echo ""
echo "⚙️  步骤2: 生成k8s配置..."
python hack/gen_k8s.py "$TOOL_NAME" "$PORT"

# 步骤3: 部署到k8s
echo ""
echo "☸️  步骤3: 部署到Kubernetes..."
TOOL_NAME_LOWER=$(echo "$TOOL_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
kubectl apply -f "infra/k8s/$TOOL_NAME_LOWER/"

echo ""
echo "✅ 部署完成！"
echo "🌐 访问地址: https://$TOOL_NAME_LOWER-mcp.mlops.dp.tech" 
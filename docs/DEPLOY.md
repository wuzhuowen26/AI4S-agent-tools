# AI4S Tools 部署指南

## 概述

本项目使用自动化的CI/CD流程来构建Docker镜像、生成Kubernetes配置并部署工具。


## 自动化CI/CD

### 触发条件

CI/CD流程会在以下情况自动触发：

1. **添加了** `servers/` 新工具
2. **代码推送**到 `main` 或 `master` 分支
3. **Pull Request**目标为 `main` 或 `master` 分支



## 工具开发

### 添加新工具

1. 在 `servers/` 目录下创建新的工具文件夹
2. 确保工具目录包含必要的依赖文件
3. 推送代码，CI/CD会自动检测并构建


### 工具目录结构要求

工具名（小写英文,使用-代替_）/
    - server.py(主要的MCP_server.py)
    - pyproject.toml,
    - uv.lock

### 公网访问

部署后，可以通过公网访问mcp工具
https://<tool-name>-mcp.mlops.dp.tech/sse


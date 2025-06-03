import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

if len(sys.argv) < 3:
    print("用法: python hack/gen_k8s.py <tool_name> <port>")
    sys.exit(1)

tool = sys.argv[1].lower().replace('_', '-')
port = int(sys.argv[2])
image = f"registry.dp.tech/deepmodeling/mcp/{tool}:latest"          
host  = f"{tool}-mcp.mlops.dp.tech"

TEMPLATE_DIR = Path("infra/k8s_templates")
OUT_DIR      = Path("infra/k8s") / tool  
OUT_DIR.mkdir(parents=True, exist_ok=True)

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=True)

for kind in ["deployment", "service", "ingress", "secret"]:
    tpl = env.get_template(f"{kind}.yaml.j2")
    rendered = tpl.render(name=tool, image=image, port=port, host=host)
    (OUT_DIR / f"{kind}.yaml").write_text(rendered)
    print(f"✅ 生成 {OUT_DIR/kind}")

print(f"\n🎉 完成！ 工具 {tool} 的 k8s 已生成\n")
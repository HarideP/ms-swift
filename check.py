# check_env.py

import pkg_resources
import sys
import subprocess

# 要求的最小版本 (只用于检查)
requirements = {
    'python': (3, 9),
    'torch': (2, 0),
    'transformers': (4, 33),
    'modelscope': (1, 19),
    'peft': (0, 11),
    'trl': (0, 13),
    'deepspeed': (0, 14),
    'vllm': (0, 5, 1),
    'lmdeploy': (0, 5),
    'evalscope': (0, 11),
}

# 推荐安装的版本 (用于给出安装建议)
recommend_versions = {
    'python': '3.10',
    'torch': '',
    'transformers': '4.51',
    'modelscope': '',
    'peft': '',
    'trl': '0.16',
    'deepspeed': '0.14.5',
    'vllm': '0.7.3 or 0.8.4',
    'lmdeploy': '0.7.2.post1',
    'evalscope': '',
}

def version_tuple(v):
    return tuple(map(int, (v.split("."))))

def check_python_version():
    py_version = sys.version_info
    required = requirements['python']
    print("===== 检查Python版本 =====")
    if py_version >= required:
        print(f"[✔] python {'.'.join(map(str, py_version[:3]))} OK")
    else:
        print(f"[X] python {'.'.join(map(str, py_version[:3]))} 不符合要求 >= {'.'.join(map(str, required))}")
        print(f"→ 推荐升级Python到 {recommend_versions['python']}")

def check_packages():
    print("\n===== 检查Python包版本 =====")
    for pkg, min_ver in requirements.items():
        if pkg == 'python':
            continue
        try:
            installed_ver = pkg_resources.get_distribution(pkg).version
            if version_tuple(installed_ver) >= min_ver:
                print(f"[✔] {pkg} {installed_ver} OK")
            else:
                print(f"[X] {pkg} {installed_ver} 不符合要求 >= {'.'.join(map(str, min_ver))}")
                suggest_install(pkg)
        except pkg_resources.DistributionNotFound:
            print(f"[X] {pkg} 未安装")
            suggest_install(pkg)

def suggest_install(pkg):
    recommended = recommend_versions.get(pkg, '')
    if recommended:
        if ' or ' in recommended:
            print(f"→ 推荐安装命令: pip install {pkg}=={recommended.split(' or ')[0]}（或{recommended.split(' or ')[1]}）")
        else:
            print(f"→ 推荐安装命令: pip install {pkg}=={recommended}")
    else:
        print(f"→ 推荐安装命令: pip install {pkg}")

def check_cuda_version():
    print("\n===== 检查CUDA版本 =====")
    try:
        output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        for line in output.split('\n'):
            if "CUDA Version" in line:
                print(f"[✔] {line.strip()}")
                return
        print("[X] 未检测到CUDA版本信息")
        print("→ 推荐安装CUDA 12")
    except Exception as e:
        print(f"[X] 无法执行nvidia-smi，可能未安装CUDA或未正确配置驱动: {e}")
        print("→ 推荐安装CUDA 12")

if __name__ == "__main__":
    check_python_version()
    check_packages()
    check_cuda_version()

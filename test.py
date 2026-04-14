import os
import re
from typing import List, Tuple, Dict, Optional
from packaging import version
from dataclasses import dataclass


@dataclass
class FileInfo:
    """文件信息类"""
    original_name: str
    mod_name: str
    version_str: str
    minecraft_version: Optional[str]
    full_path: str


class ModFileComparator:
    def __init__(self):
        # 常见版本分隔符模式
        self.version_patterns = [
            # 主要模式: modname-mcversion-modversion
            r'^(.+?)-(\d+\.\d+\.\d+(?:\.\d+)?)-([\d\.]+(?:[a-zA-Z]\d*)?.*)$',
            # 次要模式: modname-modversion-mcversion
            r'^(.+?)-([\d\.]+(?:[a-zA-Z]\d*)?.*)-(\d+\.\d+\.\d+(?:\.\d+)?)$',
            # 简单模式: modname-version
            r'^(.+?)-([\d\.]+(?:[a-zA-Z]\d*)?.*)$',
            # 带forge/fabric的模式
            r'^(.+?)-(\d+\.\d+\.\d+(?:\.\d+)?)-([\d\.]+(?:[a-zA-Z]\d*)?.*)-(?:forge|fabric)$',
            r'^(.+?)-(?:forge|fabric)-(\d+\.\d+\.\d+(?:\.\d+)?)-([\d\.]+(?:[a-zA-Z]\d*)?.*)$',
        ]

    def parse_filename(self, filename: str) -> Optional[FileInfo]:
        """解析模组文件名"""
        base_name = filename.replace('.jar', '')

        for pattern in self.version_patterns:
            match = re.match(pattern, base_name)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    # 尝试判断哪个是MC版本，哪个是模组版本
                    parts = list(groups)
                    mod_name = parts[0]

                    # 识别MC版本（通常是x.x.x格式，如1.20.1）
                    mc_version = None
                    mod_version = None

                    for i, part in enumerate(parts[1:], 1):
                        if re.match(r'^\d+\.\d+(?:\.\d+)?$', part) and part.startswith('1.'):
                            mc_version = part
                            parts.pop(i)
                            break

                    # 剩下的就是模组版本
                    if len(parts) == 3:  # 如果找到了MC版本
                        mod_version = parts[2] if parts[2] != parts[1] else parts[1]
                    elif len(parts) == 2:  # 如果没找到MC版本
                        mod_version = parts[1]

                    return FileInfo(
                        original_name=filename,
                        mod_name=mod_name,
                        version_str=mod_version,
                        minecraft_version=mc_version,
                        full_path=""
                    )

        # 如果所有模式都不匹配，尝试简单分割
        parts = base_name.split('-')
        if len(parts) >= 2:
            return FileInfo(
                original_name=filename,
                mod_name=parts[0],
                version_str='-'.join(parts[1:]),
                minecraft_version=None,
                full_path=""
            )

        return None

    def compare_versions(self, v1: str, v2: str) -> int:
        """比较两个版本号，返回1表示v1更新，-1表示v2更新，0表示相同"""
        try:
            # 清理版本字符串
            v1_clean = re.sub(r'[a-zA-Z]', '', v1)
            v2_clean = re.sub(r'[a-zA-Z]', '', v2)

            # 尝试使用packaging.version进行比较
            try:
                return 1 if version.parse(v1_clean) > version.parse(v2_clean) else \
                    -1 if version.parse(v1_clean) < version.parse(v2_clean) else 0
            except:
                # 如果packaging失败，使用简单比较
                v1_parts = [int(x) if x.isdigit() else 0 for x in v1_clean.split('.')]
                v2_parts = [int(x) if x.isdigit() else 0 for x in v2_clean.split('.')]

                # 补齐长度
                max_len = max(len(v1_parts), len(v2_parts))
                v1_parts.extend([0] * (max_len - len(v1_parts)))
                v2_parts.extend([0] * (max_len - len(v2_parts)))

                for i in range(max_len):
                    if v1_parts[i] > v2_parts[i]:
                        return 1
                    elif v1_parts[i] < v2_parts[i]:
                        return -1
                return 0
        except:
            # 如果都失败，按字符串比较
            return 1 if v1 > v2 else -1 if v1 < v2 else 0


def compare_mod_files(path1: str, path2: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    比较两个路径下的模组文件

    Returns:
        Tuple[List[str], List[str], List[str], List[str]]:
        (完全相同文件, 路径1独有文件, 路径2独有文件, 新版本文件列表)
        新版本文件格式: "filename.jar (较新版本来自路径1/2)"
    """
    comparator = ModFileComparator()

    # 获取文件列表
    files1 = [f for f in os.listdir(path1) if f.endswith('.jar')]
    files2 = [f for f in os.listdir(path2) if f.endswith('.jar')]

    # 解析文件信息
    mods1 = {}
    for f in files1:
        info = comparator.parse_filename(f)
        if info:
            info.full_path = os.path.join(path1, f)
            mods1[info.mod_name.lower()] = info

    mods2 = {}
    for f in files2:
        info = comparator.parse_filename(f)
        if info:
            info.full_path = os.path.join(path2, f)
            mods2[info.mod_name.lower()] = info

    # 分析结果
    common_exact = []  # 完全相同（包括版本）
    common_newer = []  # 有新版本的模组
    unique1 = []  # 路径1独有
    unique2 = []  # 路径2独有

    # 检查共有模组
    for mod_name in set(mods1.keys()) & set(mods2.keys()):
        mod1 = mods1[mod_name]
        mod2 = mods2[mod_name]

        if mod1.version_str == mod2.version_str:
            # 版本完全相同
            common_exact.append(mod1.original_name)
        else:
            # 比较版本
            comparison = comparator.compare_versions(mod1.version_str, mod2.version_str)
            if comparison == 1:
                # 路径1的版本较新
                common_newer.append(f"{mod1.original_name} (较新版本来自路径1)")
            elif comparison == -1:
                # 路径2的版本较新
                common_newer.append(f"{mod2.original_name} (较新版本来自路径2)")
            else:
                # 版本相同但字符串不同
                common_exact.append(mod1.original_name)

    # 找出独有文件
    for mod_name in set(mods1.keys()) - set(mods2.keys()):
        unique1.append(mods1[mod_name].original_name)

    for mod_name in set(mods2.keys()) - set(mods1.keys()):
        unique2.append(mods2[mod_name].original_name)

    return sorted(common_exact), sorted(unique1), sorted(unique2), sorted(common_newer)


def display_comparison_results(path1: str, path2: str):
    """显示比较结果"""
    try:
        common_exact, unique1, unique2, newer_versions = compare_mod_files(path1, path2)

        print("=== 模组文件比对结果 ===")
        print(f"路径1: {path1}")
        print(f"路径2: {path2}")
        print()

        print("完全相同文件:")
        if common_exact:
            for file in common_exact:
                print(f"  {file}")
        else:
            print("  无")
        print()

        print("路径1独有文件:")
        if unique1:
            for file in unique1:
                print(f"  {file}")
        else:
            print("  无")
        print()

        print("路径2独有文件:")
        if unique2:
            for file in unique2:
                print(f"  {file}")
        else:
            print("  无")
        print()

        print("共有但版本不同的文件（显示较新版本）:")
        if newer_versions:
            for file in newer_versions:
                print(f"  {file}")
        else:
            print("  无")

        return common_exact, unique1, unique2, newer_versions

    except Exception as e:
        print(f"错误: {e}")
        return [], [], [], []


# 使用示例
if __name__ == "__main__":
    # 示例用法
    path1 = "E:\\pcl2\\.minecraft\\versions\\GregTech Leisure\\mods"
    path2 = "E:\\pcl2\\.minecraft\\versions\\GregTech-Leisure-1.4.5.0\\mods"

    display_comparison_results(path1, path2)

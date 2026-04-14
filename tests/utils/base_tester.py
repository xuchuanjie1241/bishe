import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


class BaseTester:
    """测试基类，统一报告格式"""

    def __init__(self, module_name, output_dir='test_results'):
        self.module_name = module_name
        self.output_dir = os.path.join(output_dir, module_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        self.artifacts = []  # 保存生成的文件路径

    def log_result(self, test_name, data, passed=True):
        """记录测试结果"""
        result = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'passed': passed,
            'data': data
        }
        self.results.append(result)
        return result

    def save_artifact(self, name, data, subdir=''):
        """保存测试产物（图片、数据等）"""
        path = os.path.join(self.output_dir, subdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if name.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif name.endswith('.png') or name.endswith('.jpg'):
            if isinstance(data, plt.Figure):
                data.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(data)
            else:
                import cv2
                cv2.imwrite(path, data)

        self.artifacts.append(path)
        return path

    def generate_report(self):
        """生成统一格式的测试报告"""
        report = {
            'module': self.module_name,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for r in self.results if r['passed']),
                'failed': sum(1 for r in self.results if not r['passed'])
            },
            'results': self.results,
            'artifacts': self.artifacts
        }

        path = os.path.join(self.output_dir, 'report.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        # 打印摘要
        print(f"\n{'=' * 60}")
        print(f"{self.module_name} 测试报告")
        print(f"{'=' * 60}")
        print(f"通过: {report['summary']['passed']}/{report['summary']['total_tests']}")
        print(f"报告: {path}")

        return report
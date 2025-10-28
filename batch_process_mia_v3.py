#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mia_V3 批量处理ComfyUI工作流脚本
支持多图片输入和智能跳过空值
"""

import json
import requests
import time
import os
import sys
from pathlib import Path
import re

class MiaV3BatchProcessor:
    def __init__(self, comfyui_url="http://127.0.0.1:8000"):
        self.comfyui_url = comfyui_url
        self.config = {}
        self.workflow_template = {}
        self.batches = []
        
    def load_config(self, config_file="image_prompts.txt"):
        """从配置文件加载参数"""
        print(f"正在加载配置文件: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"错误: 配置文件 {config_file} 不存在")
            return False
            
        current_batch = {}
        with open(config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key == 'api_key':
                        self.config['api_key'] = value
                    else:
                        current_batch[key] = value
                elif line.startswith('Batch'):
                    # 如果已经有批次数据，先保存
                    if current_batch:
                        self.batches.append(current_batch)
                    current_batch = {}
                    print(f"  发现批次标识: {line}")
                    
        # 保存最后一个批次
        if current_batch:
            self.batches.append(current_batch)
                    
        print(f"配置加载完成:")
        print(f"  API Key: {self.config.get('api_key', 'N/A')[:10]}...")
        print(f"  发现 {len(self.batches)} 个批次")
        
        for i, batch in enumerate(self.batches, 1):
            print(f"  批次 {i}: {batch}")
            valid_images = [k for k, v in batch.items() if k in ['image1', 'image2', 'image3'] and v.strip()]
            print(f"    有效图片: {len(valid_images)} 个")
            
        return True
        
    def load_workflow_template(self, workflow_file="Mia_V3_api.json"):
        """加载工作流模板"""
        print(f"正在加载工作流模板: {workflow_file}")
        
        if not os.path.exists(workflow_file):
            print(f"错误: 工作流文件 {workflow_file} 不存在")
            return False
            
        with open(workflow_file, 'r', encoding='utf-8') as f:
            self.workflow_template = json.load(f)
            
        print("工作流模板加载完成")
        return True
        
    def check_comfyui_server(self):
        """检查ComfyUI服务器是否运行"""
        try:
            response = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            if response.status_code == 200:
                print("ComfyUI服务器连接正常")
                return True
            else:
                print(f"ComfyUI服务器响应异常: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"无法连接到ComfyUI服务器: {e}")
            print(f"请确保ComfyUI正在运行在 {self.comfyui_url}")
            return False
            
    def check_queue_status(self):
        """检查ComfyUI队列状态"""
        try:
            response = requests.get(f"{self.comfyui_url}/queue", timeout=5)
            if response.status_code == 200:
                queue_data = response.json()
                print(f"队列状态: {queue_data}")
                return queue_data
            else:
                print(f"获取队列状态失败: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"检查队列状态时发生错误: {e}")
            return None
            
    def create_workflow_for_batch(self, batch, batch_num):
        """根据批次数据创建工作流"""
        workflow = json.loads(json.dumps(self.workflow_template))  # 深拷贝
        
        # 替换API key
        workflow["1"]["inputs"]["api_key"] = self.config["api_key"]
        
        # 获取提示词 - 每个批次都使用prompt
        prompt_key = "prompt"
        if prompt_key in batch:
            workflow["1"]["inputs"]["prompt"] = batch[prompt_key]
            print(f"  使用提示词: {batch[prompt_key][:100]}...")
        else:
            print(f"  警告: 未找到提示词 {prompt_key}")
            return None
        
        # 处理图片输入 - 使用格式2: image1, image2, image3
        image_keys = ["image1", "image2", "image3"]
        image_nodes = ["4", "5", "6"]
        
        print(f"  批次数据: {batch}")
        print(f"  查找图片键: {image_keys}")
        
        # 检查哪些图片有值
        valid_images = []
        for i, (img_key, node_id) in enumerate(zip(image_keys, image_nodes)):
            print(f"  检查 {img_key}: ", end="")
            if img_key in batch:
                value = batch[img_key]
                print(f"找到值 '{value}'")
                if value and value.strip():
                    # 有图片，设置图片文件名
                    image_filename = value.strip()
                    workflow[node_id]["inputs"]["image"] = image_filename
                    valid_images.append((i, node_id, image_filename))
                    print(f"    ✓ 设置节点{node_id}图片: {image_filename}")
                else:
                    # 没有图片，设置为空字符串
                    workflow[node_id]["inputs"]["image"] = ""
                    print(f"    ✗ 节点{node_id}图片为空，跳过")
            else:
                # 没有图片，设置为空字符串
                workflow[node_id]["inputs"]["image"] = ""
                print(f"未找到键")
                print(f"    ✗ 节点{node_id}图片为空，跳过")
        
        # 根据有效图片数量调整节点1的输入连接
        if len(valid_images) == 0:
            print("  警告: 没有有效的图片输入")
            return None
        elif len(valid_images) == 1:
            # 只有一个图片，只连接第一个输入，其他留空
            workflow["1"]["inputs"]["input_image"] = [valid_images[0][1], 0]
            # 不连接其他输入，让Gemini只处理一张图片
            if "input_image_2" in workflow["1"]["inputs"]:
                del workflow["1"]["inputs"]["input_image_2"]
            if "input_image_3" in workflow["1"]["inputs"]:
                del workflow["1"]["inputs"]["input_image_3"]
        elif len(valid_images) == 2:
            # 有两个图片
            workflow["1"]["inputs"]["input_image"] = [valid_images[0][1], 0]
            workflow["1"]["inputs"]["input_image_2"] = [valid_images[1][1], 0]
            # 不连接第三个输入
            if "input_image_3" in workflow["1"]["inputs"]:
                del workflow["1"]["inputs"]["input_image_3"]
        else:
            # 有三个图片，全部连接
            workflow["1"]["inputs"]["input_image"] = [valid_images[0][1], 0]
            workflow["1"]["inputs"]["input_image_2"] = [valid_images[1][1], 0]
            workflow["1"]["inputs"]["input_image_3"] = [valid_images[2][1], 0]
        
        # 更新输出文件名前缀
        workflow["2"]["inputs"]["filename_prefix"] = f"MiaV3_Batch{batch_num}"
        
        print(f"工作流配置完成:")
        print(f"  API Key: {workflow['1']['inputs']['api_key'][:10]}...")
        print(f"  提示词: {workflow['1']['inputs']['prompt'][:50]}...")
        print(f"  有效图片数量: {len(valid_images)}")
        print(f"  输出前缀: {workflow['2']['inputs']['filename_prefix']}")
        
        return workflow
        
    def submit_workflow(self, workflow):
        """提交工作流到ComfyUI"""
        try:
            # ComfyUI API 需要特定的格式
            payload = {"prompt": workflow}
            response = requests.post(
                f"{self.comfyui_url}/prompt",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                if prompt_id:
                    print(f"工作流提交成功，Prompt ID: {prompt_id}")
                    return prompt_id
                else:
                    print("错误: 响应中没有找到prompt_id")
                    print(f"响应内容: {result}")
                    return None
            else:
                print(f"提交工作流失败: HTTP {response.status_code}")
                print(f"响应内容: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"提交工作流时发生错误: {e}")
            return None
            
    def wait_for_completion(self, prompt_id, timeout=300):
        """等待工作流处理完成"""
        print(f"等待工作流 {prompt_id} 处理完成...")
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.comfyui_url}/history/{prompt_id}", timeout=10)
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        status = history[prompt_id]
                        current_status = status.get("status", {})
                        
                        # 显示当前状态
                        if current_status != last_status:
                            print(f"当前状态: {current_status}")
                            last_status = current_status
                        
                        if current_status.get("completed", False):
                            print("工作流处理完成!")
                            return True
                        elif current_status.get("error"):
                            print(f"工作流处理出错: {current_status.get('error')}")
                            return False
                        elif current_status.get("status_str"):
                            print(f"处理状态: {current_status.get('status_str')}")
                    else:
                        print(f"提示: 工作流 {prompt_id} 尚未开始处理")
                else:
                    print(f"获取历史记录失败: HTTP {response.status_code}")
                    print(f"响应内容: {response.text}")
                            
                time.sleep(3)  # 增加等待时间
                
            except requests.exceptions.RequestException as e:
                print(f"检查状态时发生错误: {e}")
                time.sleep(3)
                
        print(f"等待超时 ({timeout}秒)")
        return False
        
    def process_batch(self, batch, batch_num):
        """处理单个批次"""
        print(f"\n{'='*50}")
        print(f"正在处理批次 {batch_num}")
        print(f"{'='*50}")
        
        # 创建工作流
        workflow = self.create_workflow_for_batch(batch, batch_num)
        if workflow is None:
            print(f"错误: 无法创建工作流，跳过批次 {batch_num}")
            return False
        
        # 检查队列状态
        print("检查队列状态...")
        self.check_queue_status()
        
        # 提交工作流
        prompt_id = self.submit_workflow(workflow)
        if not prompt_id:
            return False
            
        # 再次检查队列状态
        print("提交后的队列状态...")
        self.check_queue_status()
            
        # 等待完成
        return self.wait_for_completion(prompt_id)
        
    def run_batch_processing(self):
        """运行批量处理"""
        print("开始Mia_V3批量处理工作流...")
        
        # 检查服务器
        if not self.check_comfyui_server():
            return False
            
        # 处理所有批次
        success_count = 0
        total_count = len(self.batches)
        
        for i, batch in enumerate(self.batches, 1):
            print(f"\n处理批次 {i}")
            if self.process_batch(batch, i):
                success_count += 1
            else:
                print(f"批次 {i} 处理失败")
                
        print(f"\n{'='*50}")
        print(f"批量处理完成!")
        print(f"成功: {success_count}/{total_count}")
        print(f"{'='*50}")
        
        return success_count == total_count

def main():
    """主函数"""
    print("Mia_V3 ComfyUI 批量工作流处理器")
    print("=" * 50)
    
    # 创建处理器实例
    processor = MiaV3BatchProcessor()
    
    # 加载配置
    if not processor.load_config():
        sys.exit(1)
        
    # 加载工作流模板
    if not processor.load_workflow_template():
        sys.exit(1)
        
    # 运行批量处理
    success = processor.run_batch_processing()
    
    if success:
        print("\n所有批次处理成功!")
    else:
        print("\n部分批次处理失败，请检查日志")
        
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()

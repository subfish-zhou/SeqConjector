#!/usr/bin/env python3
"""
测试模型在data_gen数据集上的表现，并统计各种原语程序的成功率
"""

import json
import time
import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import List, Dict, Any
import multiprocessing as mp

# 添加项目路径
sys.path.insert(0, os.getcwd())

from oeis.beam_egd import egd_beam_search
from oeis.checker import check_program_on_pair, check_program_moonshine
from oeis.torch_model import enhanced_features
from main import TorchAdapter, try_templates_moonshine

# 全局worker模型
_worker_model = None
_worker_config = None

def init_worker(ckpt_path: str, device: str):
    """初始化worker进程，加载模型"""
    global _worker_model
    print(f"[Worker {mp.current_process().name}] Loading model from {ckpt_path} on {device}")
    try:
        _worker_model = TorchAdapter(ckpt_path, device=device)
        print(f"[Worker {mp.current_process().name}] Model loaded successfully")
    except Exception as e:
        print(f"[Worker {mp.current_process().name}] Failed to load model: {e}")
        _worker_model = None

def test_single_sample(task):
    """测试单个样本"""
    global _worker_model
    
    sample, beam_width, time_limit, max_steps = task
    
    A = sample['A']
    B = sample['B']
    ground_truth_toks = sample['toks']
    is_moon = sample.get('is_moon', False)
    
    # 使用前8个作为输入，后续作为验证
    min_len = min(len(A), len(B))
    n_in = min(8, min_len)
    n_chk = max(0, min_len - n_in)
    
    if n_in < 1:
        return {
            'success': False,
            'ground_truth': ground_truth_toks,
            'predicted': [],
            'time': 0,
            'error': 'sequence_too_short'
        }
    
    A_vis = A[:n_in]
    B_vis = B[:n_in]
    
    start_time = time.time()
    predicted_toks = []
    method_used = "none"
    
    try:
        # 1. 首先尝试模板匹配（快速路径）
        k_strict = 3
        tau0 = 2e-3
        tau1 = 1e-3
        
        template_toks, template_rep = try_templates_moonshine(
            A, B, n_in, n_chk,
            k_strict=k_strict, tau0=tau0, tau1=tau1
        )
        
        if template_rep and getattr(template_rep, "ok", False):
            # 模板匹配成功
            predicted_toks = template_toks
            method_used = "template"
            elapsed = time.time() - start_time
            
            return {
                'success': True,
                'ground_truth': ground_truth_toks,
                'predicted': predicted_toks,
                'time': elapsed,
                'is_moon': is_moon,
                'method': method_used,
                'error': None
            }
        
        # 2. 模板匹配失败，使用Beam搜索
        feat = enhanced_features(A_vis, B_vis)
        
        predicted_toks = egd_beam_search(
            _worker_model, 
            A_vis, B_vis, feat,
            beam=beam_width,
            max_steps=max_steps,
            use_ratio=True,
            k_strict=3,
            err_thr_lo=2e-3,
            err_thr_hi=0.10,
            time_limit=time_limit
        )
        
        method_used = "beam_search"
        elapsed = time.time() - start_time
        
        # 验证结果
        if predicted_toks:
            rep = check_program_on_pair(
                predicted_toks, 
                A_full=A, 
                B_full=B,
                n_in=n_in, 
                n_chk=n_chk
            )
            success = rep.ok
        else:
            success = False
        
        return {
            'success': success,
            'ground_truth': ground_truth_toks,
            'predicted': predicted_toks if predicted_toks else [],
            'time': elapsed,
            'is_moon': is_moon,
            'method': method_used,
            'error': None if success else 'verification_failed'
        }
        
    except Exception as e:
        return {
            'success': False,
            'ground_truth': ground_truth_toks,
            'predicted': [],
            'time': time.time() - start_time,
            'is_moon': is_moon,
            'method': method_used,
            'error': str(e)
        }

def extract_primitives(toks: List[str]) -> List[str]:
    """从token列表中提取原语（非数字token）"""
    primitives = []
    for tok in toks:
        # 跳过数字和负数
        if tok.lstrip('-').isdigit():
            continue
        primitives.append(tok)
    return primitives

def analyze_by_primitives(results: List[Dict]) -> Dict[str, Any]:
    """按原语类型分析成功率"""
    primitive_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})
    program_length_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    
    for res in results:
        ground_truth = res['ground_truth']
        success = res['success']
        
        # 统计原语
        primitives = extract_primitives(ground_truth)
        
        # 程序长度统计
        prog_len = len(ground_truth)
        program_length_stats[prog_len]['total'] += 1
        if success:
            program_length_stats[prog_len]['success'] += 1
        
        # 每个原语独立统计
        for prim in primitives:
            primitive_stats[prim]['total'] += 1
            if success:
                primitive_stats[prim]['success'] += 1
            else:
                primitive_stats[prim]['fail'] += 1
        
        # 如果有多个原语，也统计组合
        if len(primitives) > 1:
            combo_key = '+'.join(sorted(set(primitives)))
            primitive_stats[combo_key]['total'] += 1
            if success:
                primitive_stats[combo_key]['success'] += 1
            else:
                primitive_stats[combo_key]['fail'] += 1
    
    # 计算成功率
    primitive_success_rates = {}
    for prim, stats in primitive_stats.items():
        if stats['total'] > 0:
            primitive_success_rates[prim] = {
                'total': stats['total'],
                'success': stats['success'],
                'fail': stats['fail'],
                'success_rate': stats['success'] / stats['total']
            }
    
    # 排序：按总数排序
    sorted_primitives = sorted(
        primitive_success_rates.items(), 
        key=lambda x: x[1]['total'], 
        reverse=True
    )
    
    # 程序长度统计
    length_success_rates = {}
    for length, stats in sorted(program_length_stats.items()):
        if stats['total'] > 0:
            length_success_rates[length] = {
                'total': stats['total'],
                'success': stats['success'],
                'success_rate': stats['success'] / stats['total']
            }
    
    return {
        'by_primitive': sorted_primitives,
        'by_length': length_success_rates
    }

def main():
    parser = argparse.ArgumentParser(description='测试模型在data_gen数据集上的表现')
    parser.add_argument('--data', default='data_gen/easy_part_000.jsonl', 
                        help='测试数据文件')
    parser.add_argument('--ckpt', default='ckpt.pt', 
                        help='模型checkpoint路径')
    parser.add_argument('--beam', type=int, default=16, 
                        help='Beam搜索宽度')
    parser.add_argument('--timeout', type=float, default=5.0, 
                        help='超时时间（秒）')
    parser.add_argument('--max-steps', type=int, default=96, 
                        help='最大搜索步数')
    parser.add_argument('--workers', type=int, default=0, 
                        help='并行worker数量（0=自动检测）')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='运行设备')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制测试样本数量（用于快速测试）')
    parser.add_argument('--output', default='test_results.json',
                        help='输出结果文件')
    
    args = parser.parse_args()
    
    # 加载测试数据
    print(f"加载测试数据: {args.data}")
    samples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if args.limit:
        samples = samples[:args.limit]
    
    print(f"总共 {len(samples)} 个测试样本")
    
    # 准备任务
    tasks = [(sample, args.beam, args.timeout, args.max_steps) for sample in samples]
    
    # 确定worker数量
    num_workers = args.workers if args.workers > 0 else os.cpu_count()
    print(f"使用 {num_workers} 个worker进程")
    print(f"配置: beam={args.beam}, timeout={args.timeout}s, max_steps={args.max_steps}")
    
    # 运行测试
    results = []
    print("\n开始测试...")
    
    with mp.Pool(processes=num_workers, initializer=init_worker, 
                 initargs=(args.ckpt, args.device)) as pool:
        for i, result in enumerate(pool.imap_unordered(test_single_sample, tasks), 1):
            results.append(result)
            if i % 10 == 0 or i == len(tasks):
                success_count = sum(1 for r in results if r['success'])
                print(f"进度: {i}/{len(tasks)} ({i/len(tasks)*100:.1f}%) | "
                      f"成功: {success_count}/{i} ({success_count/i*100:.1f}%)")
    
    # 统计结果
    print("\n" + "="*80)
    print("测试完成！统计结果：")
    print("="*80)
    
    total = len(results)
    success = sum(1 for r in results if r['success'])
    failed = total - success
    
    # 统计方法使用情况
    template_success = sum(1 for r in results if r.get('method') == 'template' and r['success'])
    beam_success = sum(1 for r in results if r.get('method') == 'beam_search' and r['success'])
    
    print(f"\n总体统计:")
    print(f"  总样本数: {total}")
    print(f"  成功: {success} ({success/total*100:.2f}%)")
    print(f"  失败: {failed} ({failed/total*100:.2f}%)")
    print(f"\n方法统计:")
    print(f"  模板匹配成功: {template_success} ({template_success/total*100:.2f}%)")
    print(f"  Beam搜索成功: {beam_success} ({beam_success/total*100:.2f}%)")
    print(f"  总成功率提升: {success/total*100:.2f}% (vs 之前无模板)")
    
    # 按is_moon分类统计
    moon_results = [r for r in results if r.get('is_moon', False)]
    non_moon_results = [r for r in results if not r.get('is_moon', False)]
    
    if moon_results:
        moon_success = sum(1 for r in moon_results if r['success'])
        print(f"\n  Moon样本: {len(moon_results)}")
        print(f"    成功: {moon_success} ({moon_success/len(moon_results)*100:.2f}%)")
    
    if non_moon_results:
        non_moon_success = sum(1 for r in non_moon_results if r['success'])
        print(f"\n  非Moon样本: {len(non_moon_results)}")
        print(f"    成功: {non_moon_success} ({non_moon_success/len(non_moon_results)*100:.2f}%)")
    
    # 时间统计
    times = [r['time'] for r in results]
    avg_time = sum(times) / len(times) if times else 0
    print(f"\n时间统计:")
    print(f"  平均时间: {avg_time:.3f}s")
    print(f"  最小时间: {min(times):.3f}s")
    print(f"  最大时间: {max(times):.3f}s")
    
    # 按原语分析
    print("\n" + "="*80)
    print("原语程序成功率分析:")
    print("="*80)
    
    primitive_analysis = analyze_by_primitives(results)
    
    print("\n按原语类型（前30个最常见）:")
    print(f"{'原语':<30} {'总数':>8} {'成功':>8} {'失败':>8} {'成功率':>10}")
    print("-" * 80)
    
    for prim, stats in primitive_analysis['by_primitive'][:30]:
        print(f"{prim:<30} {stats['total']:>8} {stats['success']:>8} "
              f"{stats['fail']:>8} {stats['success_rate']*100:>9.2f}%")
    
    print("\n按程序长度:")
    print(f"{'长度':>6} {'总数':>8} {'成功':>8} {'成功率':>10}")
    print("-" * 50)
    
    for length in sorted(primitive_analysis['by_length'].keys()):
        stats = primitive_analysis['by_length'][length]
        print(f"{length:>6} {stats['total']:>8} {stats['success']:>8} "
              f"{stats['success_rate']*100:>9.2f}%")
    
    # 找出最擅长的原语（成功率高且样本数>=5）
    print("\n最擅长的原语（成功率>80%且样本数>=5）:")
    best_primitives = [
        (prim, stats) for prim, stats in primitive_analysis['by_primitive']
        if stats['success_rate'] > 0.8 and stats['total'] >= 5
    ]
    best_primitives.sort(key=lambda x: x[1]['success_rate'], reverse=True)
    
    for prim, stats in best_primitives[:20]:
        print(f"  {prim:<30} 成功率: {stats['success_rate']*100:.2f}% "
              f"({stats['success']}/{stats['total']})")
    
    # 找出最困难的原语（成功率低且样本数>=5）
    print("\n最困难的原语（成功率<50%且样本数>=5）:")
    hard_primitives = [
        (prim, stats) for prim, stats in primitive_analysis['by_primitive']
        if stats['success_rate'] < 0.5 and stats['total'] >= 5
    ]
    hard_primitives.sort(key=lambda x: x[1]['success_rate'])
    
    for prim, stats in hard_primitives[:20]:
        print(f"  {prim:<30} 成功率: {stats['success_rate']*100:.2f}% "
              f"({stats['success']}/{stats['total']})")
    
    # 保存详细结果
    output_data = {
        'config': {
            'data': args.data,
            'ckpt': args.ckpt,
            'beam': args.beam,
            'timeout': args.timeout,
            'max_steps': args.max_steps,
            'with_template': True
        },
        'summary': {
            'total': total,
            'success': success,
            'failed': failed,
            'success_rate': success / total if total > 0 else 0,
            'avg_time': avg_time,
            'template_success': template_success,
            'beam_success': beam_success,
            'template_rate': template_success / total if total > 0 else 0,
            'beam_rate': beam_success / total if total > 0 else 0
        },
        'by_moon': {
            'moon': {
                'total': len(moon_results),
                'success': sum(1 for r in moon_results if r['success']) if moon_results else 0,
            },
            'non_moon': {
                'total': len(non_moon_results),
                'success': sum(1 for r in non_moon_results if r['success']) if non_moon_results else 0,
            }
        },
        'primitive_analysis': {
            'by_primitive': [
                {'primitive': prim, **stats} 
                for prim, stats in primitive_analysis['by_primitive']
            ],
            'by_length': primitive_analysis['by_length']
        },
        'best_primitives': [
            {'primitive': prim, **stats} for prim, stats in best_primitives
        ],
        'hard_primitives': [
            {'primitive': prim, **stats} for prim, stats in hard_primitives
        ],
        'detailed_results': results
    }
    
    print(f"\n保存详细结果到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n测试完成！")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()


"""
正式实验脚本：批量数列关系发现

功能：
1. 从两个数据文件中导入数列（A和B）
2. 计算所有A与每条B之间的关系
3. Pipeline: Moonshine模式 -> 精确模式
4. 每种模式：模板匹配 -> 神经网络预测
5. 支持并行处理和超时控制
"""

import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
import multiprocessing as mp
from tqdm import tqdm

from oeis.program import Node, Program
from oeis.parser import parse_prefix
from oeis.interpreter import Interpreter, ExecConfig
from oeis.checker import check_program_on_pair, check_program_moonshine
from oeis.beam_egd import egd_beam_search
from oeis.torch_model import Cfg, TransDecoder, TOKENS, enhanced_features
from oeis.split_utils import compute_split
from oeis.template_matcher import try_feature_templates
from oeis.logging_config import setup_logger


# 设置日志
logger = setup_logger("experiment", level="INFO")


@dataclass
class ExperimentResult:
    """单个实验结果"""
    A_id: str
    B_id: str
    success: bool
    mode: str  # "moonshine_template", "moonshine_beam", "exact_template", "exact_beam", "failed"
    program: List[str]
    time_total: float
    time_template: float
    time_beam: float
    n_in: int
    n_chk: int
    error_msg: Optional[str] = None


def load_sequences_from_jsonl(filepath: str, max_count: int = None) -> List[Dict]:
    """
    从JSONL文件加载数列
    
    Args:
        filepath: JSONL文件路径
        max_count: 最大加载数量（None表示全部）
    
    Returns:
        数列列表，每个元素为字典，包含id和seq字段
    """
    sequences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_count is not None and i >= max_count:
                break
            data = json.loads(line.strip())
            sequences.append({
                'id': data['id'],
                'seq': data['seq'],
                'description': data.get('description', '')
            })
    return sequences


def try_templates_moonshine(A, B, n_in, n_chk, k_strict=3, tau0=2e-3, tau1=1e-3):
    """
    Moonshine模式模板匹配
    尝试三个模板：SCAN_ADD A, INSERT1 B[1] (SCAN_ADD A), INSERT2 B[2] (SCAN_ADD A)
    """
    N2 = n_in + n_chk
    cands = []
    
    toks1 = ["SCAN_ADD", "A"]
    cands.append(("SCAN_ADD", toks1))
    
    if N2 >= 2 and len(B) >= 2:
        c = B[1]
        toks2 = ["INSERT1", str(c), "SCAN_ADD", "A"]
        cands.append(("INS1", toks2))
    
    if N2 >= 3 and len(B) >= 3:
        c = B[2]
        toks3 = ["INSERT2", str(c), "SCAN_ADD", "A"]
        cands.append(("INS2", toks3))
    
    for tag, toks in cands:
        rep = check_program_moonshine(
            toks, A_full=A, B_full=B,
            n_in=n_in, n_chk=n_chk,
            k_strict=k_strict, tau0=tau0, tau1=tau1
        )
        if rep.ok:
            return toks, rep
    
    return None, None


def try_templates_exact(A, B, n_in, n_chk):
    """
    精确模式模板匹配
    尝试常见的精确转换模板
    """
    templates = [
        ["SCALE", "2", "A"],
        ["SCALE", "-1", "A"],
        ["OFFSET", "1", "A"],
        ["OFFSET", "-1", "A"],
        ["SCAN_ADD", "A"],
        ["DIFF_FWD", "1", "A"],
        ["POLY", "1", "0", "0", "A"],  # x^2
    ]
    
    for toks in templates:
        rep = check_program_on_pair(
            toks, A_full=A, B_full=B,
            n_in=n_in, n_chk=n_chk
        )
        if rep.ok:
            return toks, rep
    
    return None, None


class ModelAdapter:
    """模型适配器（用于beam search）"""
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = Cfg(**ck["cfg"])
        self.model = TransDecoder(cfg, vocab=len(TOKENS)).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device
        self.cfg = type("C", (), {"ctx_len": cfg.ctx_len})
    
    def predict_logits(self, ctx_ids, feat_vec):
        x = torch.tensor([ctx_ids], dtype=torch.long, device=self.device)
        f = torch.tensor([feat_vec.tolist() if hasattr(feat_vec, "tolist") else feat_vec], 
                        dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x, f)[0, -1, :].detach().cpu().numpy()
        return logits


def search_relation_single(A_data: Dict, B_data: Dict, 
                           ckpt_path: str,
                           beam_width: int = 16,
                           max_steps: int = 96,
                           time_limit: float = 5.0,
                           k_strict: int = 3,
                           relerr0: float = 2e-3,
                           relerr_step: float = 1e-3,
                           relerr_hi: float = 0.10,
                           device: str = "cpu") -> ExperimentResult:
    """
    搜索单个A到B的关系
    
    Pipeline:
    1. Moonshine模式：
       a. 特征驱动的模板匹配
       b. Moonshine特定模板匹配
       c. 神经网络beam search
    2. 精确模式（如果moonshine失败）：
       a. 特征驱动的模板匹配
       b. 精确模板匹配
       c. 神经网络beam search
    """
    A = A_data['seq']
    B = B_data['seq']
    A_id = A_data['id']
    B_id = B_data['id']
    
    t_start = time.time()
    
    # 计算分割点
    min_len = min(len(A), len(B))
    n_in, n_chk = compute_split(min_len)
    
    if n_in < 1 or n_chk < 1:
        return ExperimentResult(
            A_id=A_id, B_id=B_id, success=False, mode="failed",
            program=[], time_total=0, time_template=0, time_beam=0,
            n_in=n_in, n_chk=n_chk,
            error_msg="序列太短，无法分割"
        )
    
    # 计算特征（用于特征驱动的模板匹配）
    A_vis, B_vis = A[:n_in], B[:n_in]
    feat = enhanced_features(A_vis, B_vis)
    
    # ========== Phase 1: 精确模式（优先）==========
    
    best_result = None  # 记录最佳结果
    total_time_template = 0
    total_time_beam = 0
    
    # 1.1 精确特征模板匹配
    t_feat_tpl_exact = time.time()
    toks_feat_exact, rep_feat_exact = try_feature_templates(
        A, B, feat, n_in, n_chk,
        checker_mode="exact",
        max_templates=None  # 尝试所有模板
    )
    time_feat_tpl_exact = time.time() - t_feat_tpl_exact
    total_time_template += time_feat_tpl_exact
    
    if toks_feat_exact and rep_feat_exact and rep_feat_exact.ok:
        # 找到精确匹配，直接返回
        return ExperimentResult(
            A_id=A_id, B_id=B_id, success=True, mode="exact_feature_template",
            program=toks_feat_exact, time_total=time.time() - t_start,
            time_template=total_time_template, time_beam=total_time_beam,
            n_in=n_in, n_chk=n_chk
        )
    
    # 1.2 精确模板匹配
    t_tpl_exact = time.time()
    toks_tpl_exact, rep_tpl_exact = try_templates_exact(A, B, n_in, n_chk)
    time_tpl_exact = time.time() - t_tpl_exact
    total_time_template += time_tpl_exact
    
    if toks_tpl_exact and rep_tpl_exact and rep_tpl_exact.ok:
        # 找到精确匹配，直接返回
        return ExperimentResult(
            A_id=A_id, B_id=B_id, success=True, mode="exact_template",
            program=toks_tpl_exact, time_total=time.time() - t_start,
            time_template=total_time_template, time_beam=total_time_beam,
            n_in=n_in, n_chk=n_chk
        )
    
    # 1.3 精确beam search
    try:
        model = ModelAdapter(ckpt_path, device=device)
        
        t_beam_exact = time.time()
        toks_beam_exact = egd_beam_search(
            model, A_vis, B_vis, feat,
            beam=beam_width,
            max_steps=max_steps,
            use_ratio=False,  # 精确模式不使用ratio
            k_strict=k_strict,
            err_thr_lo=relerr0,
            err_thr_hi=relerr_hi,
            time_limit=time_limit
        )
        time_beam_exact = time.time() - t_beam_exact
        total_time_beam += time_beam_exact
        
        # 检查beam search结果
        if toks_beam_exact:
            rep_beam_exact = check_program_on_pair(
                toks_beam_exact, A_full=A, B_full=B,
                n_in=n_in, n_chk=n_chk
            )
            
            if rep_beam_exact.ok:
                # 找到精确匹配，直接返回
                return ExperimentResult(
                    A_id=A_id, B_id=B_id, success=True, mode="exact_beam",
                    program=toks_beam_exact, time_total=time.time() - t_start,
                    time_template=total_time_template, time_beam=total_time_beam,
                    n_in=n_in, n_chk=n_chk
                )
    except Exception as e:
        logger.warning(f"精确beam search失败: {e}")
    
    # ========== Phase 2: Moonshine模式（精确失败后的备选）==========
    
    # 2.1 Moonshine特征模板匹配
    t_feat_tpl_moon = time.time()
    toks_feat_moon, rep_feat_moon = try_feature_templates(
        A, B, feat, n_in, n_chk,
        checker_mode="moonshine",
        k_strict=k_strict,
        tau0=relerr0,
        tau1=relerr_step,
        max_templates=None  # 尝试所有模板
    )
    time_feat_tpl_moon = time.time() - t_feat_tpl_moon
    total_time_template += time_feat_tpl_moon
    
    if toks_feat_moon and rep_feat_moon and rep_feat_moon.ok:
        # 找到Moonshine匹配
        return ExperimentResult(
            A_id=A_id, B_id=B_id, success=True, mode="moonshine_feature_template",
            program=toks_feat_moon, time_total=time.time() - t_start,
            time_template=total_time_template, time_beam=total_time_beam,
            n_in=n_in, n_chk=n_chk
        )
    
    # 2.2 Moonshine特定模板匹配
    t_tpl_moon = time.time()
    toks_tpl_moon, rep_tpl_moon = try_templates_moonshine(
        A, B, n_in, n_chk,
        k_strict=k_strict, tau0=relerr0, tau1=relerr_step
    )
    time_tpl_moon = time.time() - t_tpl_moon
    total_time_template += time_tpl_moon
    
    if toks_tpl_moon and rep_tpl_moon and rep_tpl_moon.ok:
        # 找到Moonshine匹配
        return ExperimentResult(
            A_id=A_id, B_id=B_id, success=True, mode="moonshine_template",
            program=toks_tpl_moon, time_total=time.time() - t_start,
            time_template=total_time_template, time_beam=total_time_beam,
            n_in=n_in, n_chk=n_chk
        )
    
    # 2.3 Moonshine beam search
    try:
        if 'model' not in locals():
            model = ModelAdapter(ckpt_path, device=device)
        
        t_beam_moon = time.time()
        toks_beam_moon = egd_beam_search(
            model, A_vis, B_vis, feat,
            beam=beam_width,
            max_steps=max_steps,
            use_ratio=True,  # Moonshine使用ratio
            k_strict=k_strict,
            err_thr_lo=relerr0,
            err_thr_hi=relerr_hi,
            time_limit=time_limit
        )
        time_beam_moon = time.time() - t_beam_moon
        total_time_beam += time_beam_moon
        
        # 检查beam search结果
        if toks_beam_moon:
            rep_beam_moon = check_program_moonshine(
                toks_beam_moon, A_full=A, B_full=B,
                n_in=n_in, n_chk=n_chk,
                k_strict=k_strict, tau0=relerr0, tau1=relerr_step
            )
            
            if rep_beam_moon.ok:
                # 找到Moonshine匹配
                return ExperimentResult(
                    A_id=A_id, B_id=B_id, success=True, mode="moonshine_beam",
                    program=toks_beam_moon, time_total=time.time() - t_start,
                    time_template=total_time_template, time_beam=total_time_beam,
                    n_in=n_in, n_chk=n_chk
                )
    except Exception as e:
        logger.warning(f"Moonshine beam search失败: {e}")
    
    # 所有方法都失败
    return ExperimentResult(
        A_id=A_id, B_id=B_id, success=False, mode="failed",
        program=[], time_total=time.time() - t_start,
        time_template=total_time_template, time_beam=total_time_beam,
        n_in=n_in, n_chk=n_chk,
        error_msg="所有搜索方法都未找到有效程序"
    )


def worker_wrapper(args):
    """
    Worker包装函数（用于并行处理）
    """
    A_data, B_data, config = args
    try:
        result = search_relation_single(
            A_data, B_data,
            ckpt_path=config['ckpt_path'],
            beam_width=config['beam_width'],
            max_steps=config['max_steps'],
            time_limit=config['time_limit'],
            k_strict=config['k_strict'],
            relerr0=config['relerr0'],
            relerr_step=config['relerr_step'],
            relerr_hi=config['relerr_hi'],
            device=config['device']
        )
        return result
    except Exception as e:
        return ExperimentResult(
            A_id=A_data['id'], B_id=B_data['id'],
            success=False, mode="failed",
            program=[], time_total=0, time_template=0, time_beam=0,
            n_in=0, n_chk=0,
            error_msg=f"异常: {str(e)}"
        )


def run_batch_experiment(A_sequences: List[Dict],
                        B_sequences: List[Dict],
                        ckpt_path: str,
                        output_file: str,
                        num_workers: int = 1,
                        beam_width: int = 256,
                        max_steps: int = 96,
                        time_limit: float = 5.0,
                        k_strict: int = 3,
                        relerr0: float = 2e-3,
                        relerr_step: float = 1e-3,
                        relerr_hi: float = 0.10,
                        device: str = "cpu",
                        progress_interval: int = 5000):
    """
    批量运行实验
    
    Args:
        A_sequences: A数列列表
        B_sequences: B数列列表
        ckpt_path: 模型checkpoint路径
        output_file: 输出文件路径
        num_workers: 并行worker数量
        beam_width: beam search宽度
        max_steps: 最大搜索步数
        time_limit: 单次搜索超时时间（秒）
        k_strict: moonshine严格头部长度
        relerr0: moonshine初始相对误差阈值
        relerr_step: moonshine误差步长
        relerr_hi: beam search误差上限
        device: 计算设备
        progress_interval: 进度报告间隔（任务数）
    """
    
    total_tasks = len(A_sequences) * len(B_sequences)
    
    logger.info(f"========== Batch Experiment Started ==========")
    logger.info(f"A sequences: {len(A_sequences)}")
    logger.info(f"B sequences: {len(B_sequences)}")
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Beam width: {beam_width}")
    logger.info(f"Timeout: {time_limit}s")
    logger.info(f"Device: {device}")
    logger.info(f"Progress report every: {progress_interval} tasks")
    logger.info(f"==============================================")
    
    # 准备任务列表
    tasks = []
    config = {
        'ckpt_path': ckpt_path,
        'beam_width': beam_width,
        'max_steps': max_steps,
        'time_limit': time_limit,
        'k_strict': k_strict,
        'relerr0': relerr0,
        'relerr_step': relerr_step,
        'relerr_hi': relerr_hi,
        'device': device
    }
    
    for A_data in A_sequences:
        for B_data in B_sequences:
            tasks.append((A_data, B_data, config))
    
    # 运行实验
    results = []
    success_count = 0
    start_time = time.time()
    last_report_time = start_time
    
    if num_workers <= 1 or device == "cuda":
        # GPU模式或单线程：使用串行处理避免GPU竞争
        if device == "cuda":
            logger.info(f"GPU mode: using sequential processing with {num_workers} as batch hint")
        else:
            logger.info("CPU single-threaded mode")
        
        for idx, task in enumerate(tasks, 1):
            result = worker_wrapper(task)
            results.append(result)
            
            if result.success:
                success_count += 1
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
            
            # Progress report
            if idx % progress_interval == 0 or idx == total_tasks:
                elapsed = time.time() - start_time
                interval_time = time.time() - last_report_time
                speed = progress_interval / interval_time if idx % progress_interval == 0 else idx / elapsed
                eta = (total_tasks - idx) / speed if speed > 0 else 0
                success_rate = success_count / idx * 100
                
                print(f"\n{'='*70}")
                print(f"Progress Report: {idx}/{total_tasks} tasks ({idx/total_tasks*100:.1f}%)")
                print(f"  Success: {success_count} ({success_rate:.1f}%)")
                print(f"  Failed: {idx - success_count}")
                print(f"  Speed: {speed:.1f} tasks/sec")
                print(f"  Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
                print(f"{'='*70}\n")
                
                last_report_time = time.time()
    else:
        # CPU多进程并行模式
        logger.info(f"CPU parallel mode: {num_workers} workers")
        
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for result in executor.map(worker_wrapper, tasks):
                completed += 1
                results.append(result)
                
                if result.success:
                    success_count += 1
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')
                
                # Progress report
                if completed % progress_interval == 0 or completed == total_tasks:
                    elapsed = time.time() - start_time
                    interval_time = time.time() - last_report_time
                    speed = progress_interval / interval_time if completed % progress_interval == 0 else completed / elapsed
                    eta = (total_tasks - completed) / speed if speed > 0 else 0
                    success_rate = success_count / completed * 100
                    
                    print(f"\n{'='*70}")
                    print(f"Progress Report: {completed}/{total_tasks} tasks ({completed/total_tasks*100:.1f}%)")
                    print(f"  Success: {success_count} ({success_rate:.1f}%)")
                    print(f"  Failed: {completed - success_count}")
                    print(f"  Speed: {speed:.1f} tasks/sec")
                    print(f"  Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
                    print(f"{'='*70}\n")
                    
                    last_report_time = time.time()
    
    # 统计结果
    total = len(results)
    success = sum(1 for r in results if r.success)
    failed = total - success
    
    mode_stats = {}
    for r in results:
        mode_stats[r.mode] = mode_stats.get(r.mode, 0) + 1
    
    avg_time_total = np.mean([r.time_total for r in results]) if results else 0
    avg_time_template = np.mean([r.time_template for r in results]) if results else 0
    avg_time_beam = np.mean([r.time_beam for r in results if r.time_beam > 0]) if any(r.time_beam > 0 for r in results) else 0
    
    total_elapsed = time.time() - start_time
    
    # 打印统计信息
    print("\n" + "="*80)
    print("Experiment Statistics")
    print("="*80)
    print(f"Total tasks: {total}")
    print(f"Success: {success} ({success/total*100:.2f}%)")
    print(f"Failed: {failed} ({failed/total*100:.2f}%)")
    print(f"\nResults saved to: {output_file}")
    print(f"  (Only {success} successful records saved, failures not saved)")
    print(f"\nMode breakdown:")
    for mode, count in sorted(mode_stats.items(), key=lambda x: x[1], reverse=True):
        emoji = "✓" if mode != "failed" else "✗"
        print(f"  {emoji} {mode:30s}: {count:6d} ({count/total*100:.2f}%)")
    print(f"\nAverage time:")
    print(f"  Total: {avg_time_total:.3f}s")
    print(f"  Template matching: {avg_time_template:.3f}s")
    if avg_time_beam > 0:
        print(f"  Beam search: {avg_time_beam:.3f}s (only tasks using beam)")
    print(f"\nTotal execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Average throughput: {total/total_elapsed:.2f} tasks/sec")
    print("="*80)
    
    # 保存统计信息
    stats_file = output_file.replace('.jsonl', '_stats.json')
    stats = {
        'total': total,
        'success': success,
        'failed': failed,
        'success_rate': success / total if total > 0 else 0,
        'mode_stats': mode_stats,
        'avg_time_total': avg_time_total,
        'avg_time_template': avg_time_template,
        'avg_time_beam': avg_time_beam,
        'total_execution_time': total_elapsed,
        'throughput_tasks_per_sec': total / total_elapsed if total_elapsed > 0 else 0,
        'note': f'Results file contains only {success} successful records, failures not saved',
        'config': config
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Statistics saved to: {stats_file}")
    
    return results


def estimate_search_breadth(time_limit: float, beam_width: int, max_steps: int) -> Dict:
    """
    估计搜索广度
    
    Args:
        time_limit: 超时时间（秒）
        beam_width: beam宽度
        max_steps: 最大步数
    
    Returns:
        估计信息字典
    """
    # 经验估计：每步beam search大约耗时 0.01-0.02秒（CPU），0.001-0.002秒（GPU）
    # 这里假设CPU模式
    time_per_step = 0.015  # 秒
    
    estimated_steps = int(time_limit / time_per_step)
    actual_steps = min(estimated_steps, max_steps)
    
    # 搜索空间大小估计
    vocab_size = len(TOKENS)
    search_space_per_step = beam_width * vocab_size
    total_search_space = search_space_per_step * actual_steps
    
    return {
        'time_limit': time_limit,
        'beam_width': beam_width,
        'max_steps': max_steps,
        'time_per_step_estimate': time_per_step,
        'estimated_steps': estimated_steps,
        'actual_steps': actual_steps,
        'vocab_size': vocab_size,
        'search_space_per_step': search_space_per_step,
        'total_search_space': total_search_space,
        'note': f'在{time_limit}秒内，预计可以执行{actual_steps}步，每步探索{search_space_per_step}个状态'
    }


def estimate_parallel_capacity(model_size_mb: float, gpu_memory_gb: float) -> Dict:
    """
    估计并行容量
    
    Args:
        model_size_mb: 模型大小（MB）
        gpu_memory_gb: GPU显存大小（GB）
    
    Returns:
        估计信息字典
    """
    # 推理时，显存占用约为模型大小的2-2.5倍（包括activations和中间结果）
    memory_per_worker_mb = model_size_mb * 2.5
    
    # 留出20%显存作为缓冲
    available_memory_mb = gpu_memory_gb * 1024 * 0.8
    
    # 估计并行worker数
    max_workers = int(available_memory_mb / memory_per_worker_mb)
    
    # 保守建议：取估计值的80%
    recommended_workers = max(1, int(max_workers * 0.8))
    
    return {
        'model_size_mb': model_size_mb,
        'gpu_memory_gb': gpu_memory_gb,
        'memory_per_worker_mb': memory_per_worker_mb,
        'available_memory_mb': available_memory_mb,
        'max_workers_estimate': max_workers,
        'recommended_workers': recommended_workers,
        'note': f'使用{gpu_memory_gb}GB显存，模型大小{model_size_mb}MB，推荐{recommended_workers}路并行'
    }


def main():
    parser = argparse.ArgumentParser(description='批量数列关系发现实验')
    
    # 数据参数
    parser.add_argument('--A-file', type=str,
                       help='A数列文件路径（JSONL格式）')
    parser.add_argument('--B-file', type=str,
                       help='B数列文件路径（JSONL格式）')
    parser.add_argument('--A-count', type=int, default=None,
                       help='从A文件中抽取的数列数量（None表示全部）')
    parser.add_argument('--B-count', type=int, default=None,
                       help='从B文件中抽取的数列数量（None表示全部）')
    
    # 模型参数
    parser.add_argument('--ckpt', type=str, default='ckpt.pt',
                       help='模型checkpoint路径')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='计算设备')
    
    # 搜索参数
    parser.add_argument('--beam', type=int, default=256,
                       help='Beam search宽度')
    parser.add_argument('--max-steps', type=int, default=96,
                       help='最大搜索步数')
    parser.add_argument('--time-limit', type=float, default=5.0,
                       help='单次搜索超时时间（秒）')
    
    # Moonshine参数
    parser.add_argument('--k-strict', type=int, default=3,
                       help='Moonshine严格头部长度')
    parser.add_argument('--relerr0', type=float, default=2e-3,
                       help='Moonshine初始相对误差阈值')
    parser.add_argument('--relerr-step', type=float, default=1e-3,
                       help='Moonshine误差步长')
    parser.add_argument('--relerr-hi', type=float, default=0.10,
                       help='Beam search误差上限')
    
    # 并行参数
    parser.add_argument('--workers', type=int, default=1,
                       help='并行worker数量（1表示单线程）')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='results.jsonl',
                       help='输出文件路径（JSONL格式）')
    
    # 估计功能
    parser.add_argument('--estimate-breadth', action='store_true',
                       help='估计搜索广度')
    parser.add_argument('--estimate-parallel', action='store_true',
                       help='估计并行容量（需要--model-size和--gpu-memory）')
    parser.add_argument('--model-size', type=float, default=123,
                       help='模型大小（MB），用于估计并行容量')
    parser.add_argument('--gpu-memory', type=float, default=6,
                       help='GPU显存大小（GB），用于估计并行容量')
    
    args = parser.parse_args()
    
    # 估计模式
    if args.estimate_breadth:
        info = estimate_search_breadth(args.time_limit, args.beam, args.max_steps)
        print("\n搜索广度估计：")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return
    
    if args.estimate_parallel:
        info = estimate_parallel_capacity(args.model_size, args.gpu_memory)
        print("\n并行容量估计：")
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return
    
    # 检查必需参数
    if not args.A_file or not args.B_file:
        parser.error("实验模式需要 --A-file 和 --B-file 参数")
    
    # 加载数据
    logger.info(f"从 {args.A_file} 加载A数列...")
    A_sequences = load_sequences_from_jsonl(args.A_file, args.A_count)
    logger.info(f"加载了 {len(A_sequences)} 条A数列")
    
    logger.info(f"从 {args.B_file} 加载B数列...")
    B_sequences = load_sequences_from_jsonl(args.B_file, args.B_count)
    logger.info(f"加载了 {len(B_sequences)} 条B数列")
    
    # 清空输出文件
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        pass  # 清空文件
    
    # 运行实验
    results = run_batch_experiment(
        A_sequences=A_sequences,
        B_sequences=B_sequences,
        ckpt_path=args.ckpt,
        output_file=args.output,
        num_workers=args.workers,
        beam_width=args.beam,
        max_steps=args.max_steps,
        time_limit=args.time_limit,
        k_strict=args.k_strict,
        relerr0=args.relerr0,
        relerr_step=args.relerr_step,
        relerr_hi=args.relerr_hi,
        device=args.device
    )


if __name__ == '__main__':
    main()


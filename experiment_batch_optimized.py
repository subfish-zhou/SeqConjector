"""
优化版实验脚本：worker进程预加载模型，避免重复加载

关键优化：
1. 每个worker进程只加载一次模型（在进程初始化时）
2. 使用全局变量缓存模型，所有任务复用
3. 减少序列化开销
"""

import sys
sys.path.insert(0, '.')

# 导入原始模块的所有功能
from experiment_batch import *

# 全局变量：每个worker进程的模型缓存
_worker_model = None
_worker_config = None

def init_worker(config):
    """
    Worker进程初始化函数
    在每个worker进程启动时调用一次，预加载模型
    """
    global _worker_model, _worker_config
    _worker_config = config
    
    # 预加载模型
    if config['device'] != 'cpu' or True:  # 总是预加载
        try:
            _worker_model = ModelAdapter(
                config['ckpt_path'], 
                device=config['device']
            )
            print(f"Worker {mp.current_process().name}: Model loaded successfully")
        except Exception as e:
            print(f"Worker {mp.current_process().name}: Failed to load model: {e}")
            _worker_model = None


def worker_wrapper_optimized(task_data):
    """
    优化的worker函数：使用预加载的模型
    """
    global _worker_model, _worker_config
    
    A_data, B_data = task_data
    config = _worker_config
    
    try:
        # 使用优化版本的搜索函数
        result = search_relation_single_optimized(
            A_data, B_data,
            model=_worker_model,  # 使用预加载的模型
            beam_width=config['beam_width'],
            max_steps=config['max_steps'],
            time_limit=config['time_limit'],
            k_strict=config['k_strict'],
            relerr0=config['relerr0'],
            relerr_step=config['relerr_step'],
            relerr_hi=config['relerr_hi'],
            device=config['device'],
            ckpt_path=config['ckpt_path']
        )
        return result
    except Exception as e:
        return ExperimentResult(
            A_id=A_data['id'], B_id=B_data['id'],
            success=False, mode="failed",
            program=[], time_total=0, time_template=0, time_beam=0,
            n_in=0, n_chk=0,
            error_msg=f"Worker exception: {str(e)}"
        )


def search_relation_single_optimized(A_data, B_data, model=None, 
                                     beam_width=16, max_steps=96,
                                     time_limit=5.0, k_strict=3,
                                     relerr0=2e-3, relerr_step=1e-3,
                                     relerr_hi=0.10, device="cpu",
                                     ckpt_path="ckpt.pt"):
    """
    优化版：接受预加载的模型，避免重复加载
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
            error_msg="Sequence too short"
        )
    
    # 计算特征
    A_vis, B_vis = A[:n_in], B[:n_in]
    feat = enhanced_features(A_vis, B_vis)
    
    best_result = None
    total_time_template = 0
    total_time_beam = 0
    
    # Phase 1: 精确模式模板匹配
    t_feat_tpl_exact = time.time()
    toks_feat_exact, rep_feat_exact = try_feature_templates(
        A, B, feat, n_in, n_chk,
        checker_mode="exact",
        max_templates=None
    )
    time_feat_tpl_exact = time.time() - t_feat_tpl_exact
    total_time_template += time_feat_tpl_exact
    
    if toks_feat_exact and rep_feat_exact and rep_feat_exact.ok:
        return ExperimentResult(
            A_id=A_id, B_id=B_id, success=True, mode="exact_feature_template",
            program=toks_feat_exact, time_total=time.time() - t_start,
            time_template=total_time_template, time_beam=total_time_beam,
            n_in=n_in, n_chk=n_chk
        )
    
    # Phase 2: Beam search（使用预加载的模型）
    if model is None:
        # 如果没有预加载模型，临时加载（回退方案）
        model = ModelAdapter(ckpt_path, device=device)
    
    try:
        t_beam_exact = time.time()
        toks_beam_exact = egd_beam_search(
            model, A_vis, B_vis, feat,
            beam=beam_width,
            max_steps=max_steps,
            use_ratio=False,
            k_strict=k_strict,
            err_thr_lo=relerr0,
            err_thr_hi=relerr_hi,
            time_limit=time_limit
        )
        time_beam_exact = time.time() - t_beam_exact
        total_time_beam += time_beam_exact
        
        if toks_beam_exact:
            rep_beam_exact = check_program_on_pair(
                toks_beam_exact, A_full=A, B_full=B,
                n_in=n_in, n_chk=n_chk
            )
            
            if rep_beam_exact.ok:
                return ExperimentResult(
                    A_id=A_id, B_id=B_id, success=True, mode="exact_beam",
                    program=toks_beam_exact, time_total=time.time() - t_start,
                    time_template=total_time_template, time_beam=total_time_beam,
                    n_in=n_in, n_chk=n_chk
                )
    except Exception as e:
        pass
    
    # 失败
    return ExperimentResult(
        A_id=A_id, B_id=B_id, success=False, mode="failed",
        program=[], time_total=time.time() - t_start,
        time_template=total_time_template, time_beam=total_time_beam,
        n_in=n_in, n_chk=n_chk,
        error_msg="All methods failed"
    )


def run_batch_experiment_optimized(A_sequences, B_sequences, ckpt_path, output_file,
                                   num_workers=1, beam_width=256, max_steps=96,
                                   time_limit=5.0, k_strict=3, relerr0=2e-3,
                                   relerr_step=1e-3, relerr_hi=0.10, device="cpu",
                                   progress_interval=10000):
    """
    优化版批量实验：每个worker预加载模型
    """
    total_tasks = len(A_sequences) * len(B_sequences)
    
    logger.info(f"========== Optimized Batch Experiment Started ==========")
    logger.info(f"A sequences: {len(A_sequences)}")
    logger.info(f"B sequences: {len(B_sequences)}")
    logger.info(f"Total tasks: {total_tasks}")
    logger.info(f"Parallel workers: {num_workers}")
    logger.info(f"Optimization: Pre-loaded model in each worker")
    logger.info(f"Progress report interval: {progress_interval}")
    logger.info(f"=======================================================")
    
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
    
    # 准备任务列表（只传递数据，不传递config）
    # 同时建立ID到数据的映射，方便后续查找related_seq
    tasks = []
    A_map = {a['id']: a for a in A_sequences}
    B_map = {b['id']: b for b in B_sequences}
    
    for A_data in A_sequences:
        for B_data in B_sequences:
            tasks.append((A_data, B_data))
    
    # 运行实验
    results = []
    success_count = 0
    moonshine_count = 0
    exact_count = 0
    start_time = time.time()
    last_report_time = start_time
    
    # CPU多进程并行模式（带模型预加载）
    logger.info(f"CPU parallel mode: {num_workers} workers with pre-loaded models")
    
    completed = 0
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,  # 关键：预加载模型
        initargs=(config,)
    ) as executor:
        for result in executor.map(worker_wrapper_optimized, tasks):
            completed += 1
            results.append(result)
            
            if result.success:
                success_count += 1
                
                # 判断是moonshine还是exact
                is_moonshine = "moonshine" in result.mode.lower()
                is_exact = "exact" in result.mode.lower()
                
                if is_moonshine:
                    moonshine_count += 1
                    match_type = "Moonshine"
                elif is_exact:
                    exact_count += 1
                    match_type = "Exact"
                else:
                    match_type = "Unknown"
                
                # 检查B是否在A的related_seq中
                in_related = "Unknown"
                if result.A_id in A_map:
                    A_data = A_map[result.A_id]
                    if 'related_seq' in A_data:
                        in_related = "Yes" if result.B_id in A_data['related_seq'] else "No"
                    else:
                        in_related = "No_field"  # A没有related_seq字段
                
                # 实时打印成功发现
                program_str = " ".join(result.program[:10]) + ("..." if len(result.program) > 10 else "")
                print(f"\n[{match_type}] {result.A_id} -> {result.B_id} | Program: {program_str} | In_related: {in_related}")
                
                # 保存结果
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
                print(f"  Total Success: {success_count} ({success_rate:.1f}%)")
                print(f"    - Moonshine: {moonshine_count}")
                print(f"    - Exact: {exact_count}")
                print(f"  Failed: {completed - success_count}")
                print(f"  Speed: {speed:.1f} tasks/sec")
                print(f"  Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s ({eta/3600:.1f}h)")
                print(f"{'='*70}\n")
                
                last_report_time = time.time()
    
    # 统计结果
    total = len(results)
    success = sum(1 for r in results if r.success)
    total_elapsed = time.time() - start_time
    
    # 按模式分类统计
    mode_stats = {}
    for r in results:
        mode_stats[r.mode] = mode_stats.get(r.mode, 0) + 1
    
    print("\n" + "="*80)
    print("Experiment Statistics")
    print("="*80)
    print(f"Total tasks: {total}")
    print(f"Total Success: {success} ({success/total*100:.2f}%)")
    print(f"  - Moonshine matches: {moonshine_count}")
    print(f"  - Exact matches: {exact_count}")
    print(f"Failed: {total - success} ({(total-success)/total*100:.2f}%)")
    print(f"\nMode breakdown:")
    for mode, count in sorted(mode_stats.items(), key=lambda x: x[1], reverse=True):
        emoji = "✓" if mode != "failed" else "✗"
        print(f"  {emoji} {mode:30s}: {count:6d} ({count/total*100:.2f}%)")
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min, {total_elapsed/3600:.1f} h)")
    print(f"Throughput: {total/total_elapsed:.2f} tasks/sec")
    print("="*80)
    
    return results


if __name__ == '__main__':
    # 使用优化版本替换原始main函数
    parser = argparse.ArgumentParser(description='Optimized batch experiment')
    
    parser.add_argument('--A-file', type=str)
    parser.add_argument('--B-file', type=str)
    parser.add_argument('--A-count', type=int, default=None)
    parser.add_argument('--B-count', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default='ckpt.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--beam', type=int, default=256)
    parser.add_argument('--max-steps', type=int, default=96)
    parser.add_argument('--time-limit', type=float, default=5.0)
    parser.add_argument('--k-strict', type=int, default=3)
    parser.add_argument('--relerr0', type=float, default=2e-3)
    parser.add_argument('--relerr-step', type=float, default=1e-3)
    parser.add_argument('--relerr-hi', type=float, default=0.10)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=str, default='results.jsonl')
    
    args = parser.parse_args()
    
    if not args.A_file or not args.B_file:
        parser.error("Need --A-file and --B-file")
    
    logger.info(f"Loading A sequences from {args.A_file}...")
    A_sequences = load_sequences_from_jsonl(args.A_file, args.A_count)
    logger.info(f"Loaded {len(A_sequences)} A sequences")
    
    logger.info(f"Loading B sequences from {args.B_file}...")
    B_sequences = load_sequences_from_jsonl(args.B_file, args.B_count)
    logger.info(f"Loaded {len(B_sequences)} B sequences")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        pass
    
    results = run_batch_experiment_optimized(
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


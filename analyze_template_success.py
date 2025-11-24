#!/usr/bin/env python3
"""
分析模板匹配成功的案例
"""

import json

def main():
    # 加载结果
    with open('test_easy_with_template.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 找出模板匹配成功的样本
    template_successes = [
        r for r in data['detailed_results']
        if r.get('method') == 'template' and r['success']
    ]
    
    # 找出Beam搜索成功的样本
    beam_successes = [
        r for r in data['detailed_results']
        if r.get('method') == 'beam_search' and r['success']
    ]
    
    print("=" * 80)
    print("模板匹配 vs Beam搜索对比分析")
    print("=" * 80)
    
    print(f"\n总样本数: {data['summary']['total']}")
    print(f"总成功数: {data['summary']['success']}")
    print(f"总失败数: {data['summary']['failed']}")
    
    print(f"\n模板匹配成功: {len(template_successes)} ({len(template_successes)/data['summary']['total']*100:.2f}%)")
    print(f"Beam搜索成功: {len(beam_successes)} ({len(beam_successes)/data['summary']['total']*100:.2f}%)")
    
    if template_successes:
        print("\n" + "=" * 80)
        print(f"模板匹配成功的 {len(template_successes)} 个样本详情:")
        print("=" * 80)
        
        # 按时间排序
        template_successes.sort(key=lambda x: x['time'])
        
        for i, r in enumerate(template_successes, 1):
            print(f"\n{i}. 程序: {' '.join(r['ground_truth'])}")
            print(f"   预测: {' '.join(r['predicted'])}")
            print(f"   匹配: {'✓' if r['ground_truth'] == r['predicted'] else '✗'}")
            print(f"   时间: {r['time']:.6f}s")
            print(f"   Moon: {r['is_moon']}")
        
        # 统计模板成功的程序类型
        print("\n" + "=" * 80)
        print("模板匹配成功的程序模式分析:")
        print("=" * 80)
        
        from collections import Counter
        
        # 提取原语
        primitives = []
        for r in template_successes:
            tokens = r['ground_truth']
            prims = [t for t in tokens if not t.lstrip('-').isdigit()]
            primitives.extend(prims)
        
        prim_counter = Counter(primitives)
        print("\n原语频率:")
        for prim, count in prim_counter.most_common():
            print(f"  {prim}: {count}")
        
        # 程序长度分布
        lengths = [len(r['ground_truth']) for r in template_successes]
        length_counter = Counter(lengths)
        print("\n程序长度分布:")
        for length in sorted(length_counter.keys()):
            print(f"  长度{length}: {length_counter[length]}个")
        
        # 时间统计
        times = [r['time'] for r in template_successes]
        print(f"\n时间统计:")
        print(f"  平均: {sum(times)/len(times):.6f}s")
        print(f"  最小: {min(times):.6f}s")
        print(f"  最大: {max(times):.6f}s")
    
    # 对比Beam搜索的时间
    if beam_successes:
        beam_times = [r['time'] for r in beam_successes]
        print("\n" + "=" * 80)
        print("Beam搜索成功案例的时间统计:")
        print("=" * 80)
        print(f"  平均: {sum(beam_times)/len(beam_times):.6f}s")
        print(f"  最小: {min(beam_times):.6f}s")
        print(f"  最大: {max(beam_times):.6f}s")
        
        if template_successes:
            template_avg = sum(times)/len(times)
            beam_avg = sum(beam_times)/len(beam_times)
            print(f"\n模板匹配比Beam搜索快: {beam_avg/template_avg:.1f}倍")
    
    # 分析失败的案例中，哪些可能适合加入模板
    failures = [
        r for r in data['detailed_results']
        if not r['success']
    ]
    
    if failures:
        print("\n" + "=" * 80)
        print("失败案例中的简单模式（可能适合加入模板）:")
        print("=" * 80)
        
        # 找出长度<=3且失败的案例
        simple_failures = [r for r in failures if len(r['ground_truth']) <= 3]
        print(f"\n长度<=3的失败案例: {len(simple_failures)}")
        
        if simple_failures:
            fail_prims = []
            for r in simple_failures[:10]:  # 只显示前10个
                tokens = r['ground_truth']
                prims = [t for t in tokens if not t.lstrip('-').isdigit()]
                fail_prims.extend(prims)
                print(f"  - {' '.join(tokens)}")
            
            fail_prim_counter = Counter(fail_prims)
            print("\n这些失败案例中最常见的原语:")
            for prim, count in fail_prim_counter.most_common(10):
                print(f"  {prim}: {count}")

if __name__ == '__main__':
    main()


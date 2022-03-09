from ql.analyzer import JavaAnalyzer, analyze_str, analyze_str_context, decode_bqrs_all, merge_csv

if __name__ == '__main__':
    base = '/home/rain/program/tmp'
    analyze_str_context(base, f'{base}/pass.csv', 'java')
    decode_bqrs_all(base, 'context_to', 'java')
    # context_to = merge_csv(base, cmd='context_to', language='java')
    # context_to.to_csv(f'{base}/pass_context_to.csv')
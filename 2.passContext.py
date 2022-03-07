from ql.analyzer import JavaAnalyzer, analyze_str, analyze_str_context

if __name__ == '__main__':
    base = '/home/rain/program/tmp'
    context_to = analyze_str_context(base, f'{base}/pass.csv', 'java')
    context_to.to_csv(f'{base}/pass_context_to.csv')
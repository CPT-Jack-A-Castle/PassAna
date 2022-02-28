from ql.analyzer import JavaAnalyzer, analyze_str, analyze_str_context
#for z in *.zip; do unzip $z; done
if __name__ == '__main__':
    base = '/home/rain/program/java'
    analyze_str_context(base, f'{base}/string.csv', 'java')
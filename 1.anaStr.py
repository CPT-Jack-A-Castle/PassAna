from ql.analyzer import analyze_str, merge_csv

if __name__ == '__main__':
    base = '/home/rain/program/java'
    analyze_str(base, 'findString', skip=False)
    data = merge_csv(base)
    data.to_csv(f'{base}/string.csv')
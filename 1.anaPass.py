from ql.analyzer import analyze_str, merge_data

if __name__ == '__main__':
    base = '/home/rain/program/java'
    analyze_str(base, 'findPass', skip=True)
    data = merge_data(base)
    data.to_csv(f'{base}/string.csv')
from ql.analyzer import analyze_str, merge_csv, decode_bqrs_all

if __name__ == '__main__':
    base = '/home/rain/program/tmp'
    analyze_str(base, 'findString', skip=False)
    decode_bqrs_all(base, 'findString')
    data = merge_csv(base, 'findString')
    data.to_csv(f'{base}/string.csv')
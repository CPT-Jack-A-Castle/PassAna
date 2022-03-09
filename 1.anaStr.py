from ql.analyzer import analyze_str, merge_csv, decode_bqrs_all

if __name__ == '__main__':
    language = 'cpp'
    base = '/home/rain/program/cpp'
    analyze_str(base, 'findString', language, skip=False)
    data = merge_csv(base, 'findString', language)
    data.to_csv(f'{base}/string.csv')
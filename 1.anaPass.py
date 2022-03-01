from ql.analyzer import analyze_str, merge_csv, decode_bqrs_all

if __name__ == '__main__':
    base = '/home/rain/program/tmp'
    analyze_str(base, 'findPass', skip=True)
    decode_bqrs_all(base, 'findPass')
    data = merge_csv(base, 'findPass')
    data.to_csv(f'{base}/pass.csv')
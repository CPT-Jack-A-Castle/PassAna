from ql.analyzer import analyze_str, merge_data

if __name__ == '__main__':
    analyze_str('/home/rain/program/java', 'findString', skip=True)
    data = merge_data('/home/rain/program/java')
    data.to_csv('/home/rain/program/java/string.csv')
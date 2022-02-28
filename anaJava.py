from ql.analyzer import process_all_dataset, merge_data

if __name__ == '__main__':
    process_all_dataset('/home/rain/program/java', 'findString', skip=True)
    data = merge_data('/home/rain/program/java')
    data.to_csv('/home/rain/program/java/string.csv')
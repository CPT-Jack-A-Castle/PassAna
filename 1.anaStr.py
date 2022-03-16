from ql.analyzer import init_analyzer

if __name__ == '__main__':
    language = 'python'
    base = '/home/gonghuihui/pwd_proj/python_database'
    analyzer = init_analyzer(language)
    analyzer.get_str_from_projects(base, threads=8)
    data = analyzer.merge_csv(base, "findString")
    data.to_csv(f'{base}/string.csv')
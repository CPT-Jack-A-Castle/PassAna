from ql.analyzer import init_analyzer

if __name__ == '__main__':
    language = 'javascript'
    base = '/media/rain/data/js_zip'
    analyzer = init_analyzer(language)
    analyzer.get_str_from_projects(base, threads=8, skip=True)
    data = analyzer.merge_csv(base, "findString")
    data.to_csv(f'csv/{language}/string.csv')
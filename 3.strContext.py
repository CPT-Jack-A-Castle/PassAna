from ql.analyzer import init_analyzer

if __name__ == '__main__':
    base = '/home/rain/program/tmp'
    language = 'java'
    analyzer = init_analyzer(language)
    analyzer.get_context_for_strs(base, f'csv/{language}/string.csv')
    context_to = analyzer.merge_csv(base, 'context_to')
    context_to.to_csv(f'csv/{language}/str_context.csv')

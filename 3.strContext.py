from ql.analyzer import init_analyzer
import pandas as pd


if __name__ == '__main__':
    language = 'java'
    base = f'/home/gonghuihui/pwd_proj/{language}_database'
    analyzer = init_analyzer(language)
    analyzer.get_context_for_strs(base, f'csv/{language}/string.csv', skip=False)
    context_to = analyzer.merge_csv(base, 'context_str')
    context_to = context_to.drop(columns="project")
    source = pd.read_csv(f'csv/{language}/string.csv', index_col=0)
    out = pd.merge(source, context_to, on=['var', 'location'])
    out.to_csv(f'csv/{language}/str_context.csv')

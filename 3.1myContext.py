from ql.analyzer import init_analyzer
import pandas as pd


if __name__ == '__main__':
    language = 'cpp'
    base = f'/home/gonghuihui/pwd_proj/{language}_database'
    for str_label in ['pass', 'string']:
        analyzer = init_analyzer(language)
        analyzer.get_context_for_strs(base, f'csv/{language}/{str_label}.csv', skip=True)
        if str_label == "string":
            context_to = analyzer.merge_csv(base, 'context_str')
        else:
            context_to = analyzer.merge_csv(base, 'context_pass')
        context_to = context_to.drop(columns="project")
        source = pd.read_csv(f'csv/{language}/{str_label}.csv', index_col=0)

        out = pd.merge(source, context_to, on=['var', 'location'])
        if str_label == "string":
            out.to_csv(f'csv/{language}/mycontext_string.csv')
        else:
            out.to_csv(f'csv/{language}/mycontext_pass.csv')


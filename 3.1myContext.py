from context.contextTool import split_context_csv_by_project
from ql.analyzer import init_analyzer
import pandas as pd


if __name__ == '__main__':
    language = 'java'
    base = '/media/rain/data/java_zip'
    for str_label in ['pass']:
        analyzer = init_analyzer(language)

        if str_label == "string":
            analyzer.get_context_for_strs(base, f'csv/{language}/{str_label}.csv', skip=False)
            context_to = analyzer.merge_csv(base, 'context_str')
        else:
            analyzer.get_context_for_passs(base, f'csv/{language}/{str_label}2.csv', skip=True)
            context_to = analyzer.merge_csv(base, 'context_pass')
        context_to = context_to.drop(columns="project")
        source = pd.read_csv(f'csv/{language}/{str_label}.csv', index_col=0)
        try:
            out = split_context_csv_by_project(source, context_to)
        except Exception as e:
            print(f"error with {e}")
            continue
        if str_label == "string":
            out.to_csv(f'csv/{language}/mycontext_string.csv')
        else:
            out.to_csv(f'csv/{language}/mycontext_pass2.csv')


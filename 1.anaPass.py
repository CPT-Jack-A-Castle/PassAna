from passwd.passTool import process_found_pass
from ql.analyzer import init_analyzer

if __name__ == '__main__':
    language = 'java'
    base = '/media/rain/data/java_zip'
    analyzer = init_analyzer(language)
    analyzer.get_pass_from_projects(base, threads=8, skip=True)
    data = analyzer.merge_csv(base, "findPass")

    data.to_csv(f'csv/{language}/pass.csv')

    process_found_pass(f'csv/{language}')
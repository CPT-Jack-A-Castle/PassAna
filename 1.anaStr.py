# from passwd.passTool import process_found_pass
from ql.analyzer import init_analyzer

if __name__ == '__main__':
    language = 'cpp'
    base = '/home/gonghuihui/pwd_proj/cpp_database'
    analyzer = init_analyzer(language)
    analyzer.get_str_from_projects(base, threads=8)
    data = analyzer.merge_csv(base, "findString")
    data.to_csv(f'csv/{language}//string.csv')
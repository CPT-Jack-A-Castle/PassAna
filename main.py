from ql.analyzer import JavaAnalyzer, PythonAnalyzer

if __name__ == '__main__':
    analyzer = JavaAnalyzer(True)
    # # analyzer.analyze_create('/home/rain/program/java/mall')
    analyzer.set_cmd('findString')
    analyzer.ql_find_str('/home/rain/program/java/mall')
    analyzer.decode_bqrs2csv('/home/rain/program/java/mall')
    data = JavaAnalyzer.load_project_csv('/home/rain/program/java/mall')
    JavaAnalyzer.create_dfg_from_csv(data)
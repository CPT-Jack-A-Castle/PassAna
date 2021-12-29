from analyzer import CppAnalyzer, JavaAnalyzer

if __name__ == '__main__':
    analyzer = JavaAnalyzer(True)
    # analyzer.analyze_create('/home/rain/program/java/mall')
    analyzer.ql_analyze('/home/rain/program/java/mall')
    out = analyzer.decode_bqrs2csv('/home/rain/program/java/mall')
    print()
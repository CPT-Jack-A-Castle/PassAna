from context.contextTool import merge_my_context
from context.passFinderContext import merge_passfinder_context

if __name__ == '__main__':
    data = merge_my_context('/home/rain/PassAna/csv', 'pass')
    data.to_csv("raw_dataset/mycontext_pass.csv", index=False)

    data = merge_my_context('/home/rain/PassAna/csv', 'string')
    data.to_csv("raw_dataset/mycontext_str.csv", index=False)
from context.passFinderContext import merge_passfinder_context

if __name__ == '__main__':
    data = merge_passfinder_context('/home/rain/PassAna/csv', 'pass')
    data.to_csv("raw_dataset/passfindercontext_pass.csv", index=False)

    data = merge_passfinder_context('/home/rain/PassAna/csv', 'string')
    data.to_csv("raw_dataset/passfindercontext_str.csv", index=False)
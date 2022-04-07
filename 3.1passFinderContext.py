from context.passFinderContext import extract_context_passfinder_all, merge_passfinder_context

if __name__ == '__main__':
    language = 'java'
    base = '/media/rain/data/java_zip'
    extract_context_passfinder_all(
        base,
        language,
        "pass"
    )
    extract_context_passfinder_all(
        base,
        language,
        "string"
    )


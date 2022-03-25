from context.passFinderContext import extract_context_passfinder_all, merge_passfinder_context

if __name__ == '__main__':
    language = 'javascript'
    base = '/media/rain/data/js_zip'
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

